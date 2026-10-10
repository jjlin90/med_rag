"""用真实MySQL临时实例和隔离Redis键验证FAQ取答与缓存。

需要Docker Desktop、可连接的项目Redis及有权使用的FAQ CSV。
MySQL仅绑定本机随机端口，使用随机密码和内存数据目录，结束后删除。
不调用HTTP API或供应商LLM，不修改现有数据库或项目默认配置。
"""
import argparse
from datetime import datetime, timezone
import copy
import csv
import hashlib
import json
import logging
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pymysql
from src.config.settings import Config
from src.online_service.cache_manager import CacheManager, faq_cache_key
from src.online_service.faq_search import FAQSearch
from scripts.verify_faq_guard import SUBSTITUTIONS

logging.basicConfig(level=logging.WARNING)


def docker(*args):
    result = subprocess.run(['docker', *args], capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Docker command failed: ' + args[0] + ': ' + result.stderr.strip())
    return result.stdout.strip()


class ScopedCache:
    def __init__(self, manager, prefix):
        self.manager, self.prefix = manager, prefix
        self.keys = set()

    def get(self, key):
        return self.manager.get(self.prefix + key)

    def set(self, key, value, ttl=None):
        scoped = self.prefix + key
        self.keys.add(scoped)
        return self.manager.set(scoped, value, ttl=ttl)


class CountingCursor:
    def __init__(self, cursor):
        self.cursor, self.selects = cursor, 0

    def execute(self, statement, parameters=None):
        self.selects += statement.startswith('SELECT')
        return self.cursor.execute(statement, parameters)

    def fetchone(self):
        return self.cursor.fetchone()

    def __getattr__(self, name):
        return getattr(self.cursor, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/msd_faq_clean.csv',
                        help='FAQ CSV，需含question和answer列')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/faq_services_verification.json',
                        help='通过验收及资源清理后写入的JSON摘要')
    parser.add_argument('--mysql-image', default='mysql:8.0', help='临时验收实例使用的MySQL镜像')
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_file():
        parser.error('FAQ CSV不存在：' + str(dataset))
    with dataset.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        if not {'question', 'answer'}.issubset(reader.fieldnames or []):
            parser.error('FAQ CSV必须包含question和answer列')
        rows = [{'question':r['question'].strip(), 'answer':r.get('answer','').strip()}
                for r in reader if r.get('question','').strip()]
    if not rows:
        parser.error('FAQ CSV没有有效标准问题')
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    config = Config()
    label = secrets.token_hex(5)
    container = 'med-rag-faq-check-' + label
    password = secrets.token_urlsafe(32)
    cache = None
    faq = None
    created = False
    env_path = None
    report = {}
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         suffix='.env', dir=output.parent,
                                         delete=False) as env_file:
            env_path = Path(env_file.name)
            env_file.write('MYSQL_ROOT_PASSWORD=' + password + '\n')
        docker('run', '--rm', '-d', '--name', container,
               '-p', '127.0.0.1::3306', '--tmpfs', '/var/lib/mysql:rw,size=1073741824',
               '--env-file', str(env_path), args.mysql_image)
        created = True
        env_path.unlink()
        env_path = None
        binding = json.loads(docker('inspect', '--format', '{{json .NetworkSettings.Ports}}', container))
        port = int(binding['3306/tcp'][0]['HostPort'])
        connection = None
        for attempt in range(45):
            try:
                connection = pymysql.connect(host='127.0.0.1', port=port, user='root',
                                             password=password, connect_timeout=2,
                                             charset='utf8mb4')
                break
            except pymysql.Error:
                time.sleep(2)
        if connection is None:
            raise RuntimeError('Temporary MySQL did not become ready')
        with connection.cursor() as cursor:
            cursor.execute('CREATE DATABASE med_rag_verification CHARACTER SET utf8mb4')
        connection.close()

        config = copy.copy(config)
        config.MYSQL_HOST, config.MYSQL_PORT = '127.0.0.1', port
        config.MYSQL_USER, config.MYSQL_PASSWORD = 'root', password
        config.MYSQL_DATABASE = 'med_rag_verification'
        manager = CacheManager(config)
        if manager.client is None or not manager.client.ping():
            raise RuntimeError('Real Redis unavailable')
        cache = ScopedCache(manager, 'med-rag-live-check:' + label + ':')
        faq = FAQSearch(config, cache)
        if faq.connection is None:
            raise RuntimeError('Production FAQ MySQL initialization failed')
        faq.cursor.executemany('INSERT INTO faq (question, answer) VALUES (%s, %s)',
                               [(r['question'], r['answer']) for r in rows])
        faq.connection.commit()
        faq._init_bm25_index()
        tracked = CountingCursor(faq.cursor)
        faq.cursor = tracked

        controls = 0
        chosen = None
        for row in rows:
            answer, need_rag = faq.search_faq(row['question'], use_cache=False)
            if answer and not need_rag:
                controls += 1
                if chosen is None:
                    chosen = row
                if answer != row['answer']:
                    raise RuntimeError('MySQL answer differs from CSV')
        if chosen is None:
            raise RuntimeError('No exact FAQ control accepted')

        probes = 0
        opposite = 0
        nonexact = 0
        for row in rows:
            for old, new in SUBSTITUTIONS:
                if old in row['question'] and new not in row['question']:
                    query = row['question'].replace(old, new, 1)
                    scores = faq.bm25_index.search_normalized(query, k=5)
                    matched = faq.bm25_index.documents[scores[0][1]] if scores else ''
                    answer, need_rag = faq.search_faq(query, use_cache=False)
                    probes += 1
                    opposite += bool(answer and not need_rag and old in matched)
                    nonexact += bool(answer and not need_rag and matched != query)
                    break
        if opposite or nonexact:
            raise RuntimeError('Live FAQ guard accepted an incompatible question')

        before = tracked.selects
        first = faq.search_faq(chosen['question'], use_cache=True)
        if first != (chosen['answer'], False) or tracked.selects != before + 1:
            raise RuntimeError('Cache miss did not query actual MySQL')
        payload = cache.get(faq_cache_key(chosen['question']))
        if not payload or payload['question'] != chosen['question'] or payload['answer'] != chosen['answer']:
            raise RuntimeError('Actual Redis payload missing acceptance evidence')
        before = tracked.selects
        second = faq.search_faq(chosen['question'], use_cache=True)
        if second != first or tracked.selects != before:
            raise RuntimeError('Actual Redis hit did not bypass MySQL')
        if manager.ttl(cache.prefix + faq_cache_key(chosen['question'])) <= 0:
            raise RuntimeError('Actual Redis entry has no expiry')

        if not cache.set(faq_cache_key(chosen['question']), {**payload, 'question':'other standard question'}):
            raise RuntimeError('Could not seed actual Redis question-mismatch case')
        before = tracked.selects
        repaired = faq.search_faq(chosen['question'], use_cache=True)
        if repaired != first or tracked.selects != before + 1:
            raise RuntimeError('Mismatched cache evidence was accepted')
        if not cache.set(faq_cache_key(chosen['question']), {**payload, 'answer':'   '}):
            raise RuntimeError('Could not seed actual Redis empty-answer case')
        before = tracked.selects
        if faq.search_faq(chosen['question'], use_cache=True) != first or tracked.selects != before + 1:
            raise RuntimeError('Empty cached answer was accepted')
        legacy = faq_cache_key(chosen['question']).replace('faq:v3:', 'faq:v2:', 1)
        if not cache.set(legacy, {'answer':'legacy wrong answer', 'type':'faq'}):
            raise RuntimeError('Could not seed legacy Redis case')
        manager.delete(cache.prefix + faq_cache_key(chosen['question']))
        before = tracked.selects
        if (faq.search_faq(chosen['question'], use_cache=True) != first
                or tracked.selects != before + 1):
            raise RuntimeError('Legacy Redis namespace leaked')

        report = {'scope':'Actual production FAQSearch and BM25 with isolated real MySQL, '
                           'existing real Redis under a private key prefix; no mocked IO or LLM/API calls.',
                  'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
                  'faq_count':len(rows), 'original_questions_accepted':controls,
                  'perturbation_count':probes, 'opposite_acceptances':opposite,
                  'nonexact_acceptances':nonexact, 'mysql_cache_miss_verified':True,
                  'redis_hit_bypasses_mysql':True, 'cache_question_mismatch_rejected':True,
                  'empty_cached_answer_rejected':True, 'legacy_v2_namespace_ignored':True,
                  'redis_ttl_verified':True, 'sql_select_calls':tracked.selects}
    finally:
        # Every cleanup action is attempted even if an earlier one fails.
        cleanup_errors = []
        if faq is not None and faq.connection is not None:
            try:
                faq.connection.close()
            except Exception as exc:
                cleanup_errors.append('MySQL connection close: ' + type(exc).__name__)
        if cache is not None:
            try:
                # Use the strict client here: convenience helpers convert IO
                # failures to False, which cannot prove resource removal.
                if cache.keys:
                    cache.manager.client.delete(*cache.keys)
                if cache.keys and cache.manager.client.exists(*cache.keys):
                    cleanup_errors.append('Temporary Redis keys remain')
            except Exception as exc:
                cleanup_errors.append('Redis cleanup: ' + type(exc).__name__)
        if env_path is not None:
            try:
                env_path.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_errors.append('Temporary env file cleanup: ' + type(exc).__name__)
        if created:
            try:
                docker('stop', '-t', '10', container)
                if docker('ps', '-a', '--filter', 'name=^/' + container + '$', '--format', '{{.Names}}'):
                    cleanup_errors.append('Temporary MySQL container remains')
            except Exception as exc:
                cleanup_errors.append('Temporary MySQL cleanup: ' + str(exc))
        if cleanup_errors:
            raise RuntimeError('; '.join(cleanup_errors))
    report['temporary_resources_removed'] = True
    report['generated_at'] = datetime.now(timezone.utc).isoformat()
    output.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

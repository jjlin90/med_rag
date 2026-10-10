"""Fixed 20-question medical RAG A/B pilot; explicit stages and saved evidence.

Generation receives only question and retrieved contexts. Reference answers are
used only by the independent Ragas judge. No historical report is overwritten.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Bootstrap OpenMP before importing model/native dependencies.
from src.config.settings import Config
from src.online_service.llm_generator import LLMGenerator, format_knowledge_document, validated_review_answer, indexed_knowledge, evidence_paragraphs

METRICS = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
FOCUSED_RULES = """\n补充输出要求：只回答用户实际询问的目标和必要条件，通常用1～3句；方法类可列最多4个直接相关要点，不能为缩短而丢掉必要限制。不要列未被问到的其他疾病、检查、背景或通用医嘱。
先区分一般事实与这个人的诊断：只给有依据的一般规律，不能排除知识未讨论的原因。缺少具体起效时间、疗程等细节时，简要提供已知作用并指出该细节未说明，不编造时间。用户只问是否有关或是否会遗传时，回答知识支持的关系，不自行扩展到未询问的概率和复杂传递规则。数字、单位及其成立条件保持原意。"""



def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_arm_rows(rows, manifest):
    expected = {(i,a) for i in manifest['indices_1based'] for a in manifest['arms']}
    actual = [(r['index'],r['arm']) for r in rows]
    if (len(manifest['indices_1based']) != 20 or len(set(manifest['indices_1based'])) != 20
            or set(manifest['arms']) != {'baseline','candidate'}
            or len(actual) != 40 or len(set(actual)) != 40 or set(actual) != expected):
        raise ValueError('Exactly twenty unique paired questions are required, with no duplicates or omissions.')


def prepare(out, cfg, variant='title-only'):
    if (out / 'manifest.json').exists():
        return read(out / 'manifest.json')
    source = ROOT / 'data/test_query/eval_answers_210_grounded_v3_ar.json'
    historical = ROOT / 'data/test_query/eval_ragas_210_grounded_v3_ar_mixed_complete.json'
    items, report = read(source), read(historical)
    scores = {s['question']: s for s in report['scores']}
    order = sorted(range(len(items)), key=lambda i: (scores[items[i]['question']]['answer_relevancy'], i))
    rng = random.Random(20261010)
    indices = sorted(i for b in range(4) for i in rng.sample(order[b*len(order)//4:(b+1)*len(order)//4], 5))
    generator = LLMGenerator.__new__(LLMGenerator)
    generator.config = cfg
    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'seed': 20261010,
        'selection': 'Five per historical AR quartile, fixed before generation/scoring; all twenty retained.',
        'indices_1based': [i+1 for i in indices],
        'source': str(source.relative_to(ROOT)), 'source_sha256': digest(source),
        'historical_scores_sha256': digest(historical),
        'generation_model': cfg.LLM_MODEL_NAME, 'temperature': cfg.LLM_TEMPERATURE,
        'max_tokens': cfg.LLM_MAX_TOKENS,
        'scope': 'Medical RAG with fixed direct retrieval, no FAQ/cache/intent routing; live existing Milvus collection.',
        'arms': {
            'baseline': {'retrieve': cfg.TOP_K_RETRIEVE, 'children': cfg.TOP_K_CHILDREN,
                         'rerank': cfg.TOP_K_RERANK, 'nprobe': cfg.MILVUS_NPROBE,
                         'context_titles':True,
                         'system_prompt': generator._build_system_prompt()},
            'candidate': {'retrieve': cfg.TOP_K_RETRIEVE, 'children':cfg.TOP_K_CHILDREN,
                          'rerank':cfg.TOP_K_RERANK, 'nprobe':cfg.MILVUS_NPROBE,
                          'title_anchor':True,'context_titles':True,
                          'system_prompt':generator._build_system_prompt(),
                          'review_prompt':LLMGenerator._GROUNDING_REVIEW_SYS,
                          'review_json':True,'review_max_tokens':3072,'evidence_ids':True},
        },
        'user_prompt_template': generator._build_user_prompt('{query}', '{context}'),
    }
    if variant != 'review':
        candidate = manifest['arms']['candidate']
        for key in ['review_prompt','review_json','review_max_tokens','evidence_ids']:
            candidate.pop(key,None)
        if variant == 'focused':
            candidate['system_prompt'] += FOCUSED_RULES
    manifest['candidate_variant'] = variant
    write(out / 'items.json', [dict(index=i+1, question=items[i]['question'],
                                  ground_truth=items[i]['ground_truth']) for i in indices])
    write(out / 'manifest.json', manifest)
    return manifest


def retrieve(out, cfg, manifest):
    from src.online_service.retrieval import Retrieval
    from src.online_service.reranker import Reranker
    retriever = Retrieval(cfg)
    retriever.milvus_store.create_or_load_collection()
    reranker = Reranker(cfg)
    items = read(out / 'items.json')
    rows = read(out / 'retrieved.json') if (out / 'retrieved.json').exists() else []
    done = {(r['arm'], r['index']) for r in rows}
    for item in items:
        for arm, params in manifest['arms'].items():
            if (arm, item['index']) in done:
                continue
            retriever.top_k_retrieve = params['retrieve']
            retriever.top_k_children = params['children']
            retriever.milvus_store.nprobe = params['nprobe']
            cfg.RETRIEVAL_TITLE_ANCHOR = bool(params.get('title_anchor'))
            start = time.perf_counter()
            result = retriever.search_child_to_parent(item['question'])
            if result.error or not result.is_usable:
                raise RuntimeError(f"Retrieval unavailable at {arm}/{item['index']}: {result.error or result.degrade_reason}")
            docs = reranker.rerank(item['question'], result.documents, top_k=params['rerank'])
            contexts = [format_knowledge_document(d) if params.get('context_titles') else d['content'] for d in docs]
            rows.append({**item, 'arm': arm, 'contexts': contexts,
                         'sources': [{'id':d['id'], 'source':d.get('source'),
                                      'rerank_score':d['rerank_score']} for d in docs],
                         'retrieval_seconds':time.perf_counter()-start,
                         'degrade_level':result.degrade_level})
            write(out / 'retrieved.json', rows)
            print(f"retrieve {len(rows)}/40 {arm} index={item['index']} contexts={len(docs)}", flush=True)


def generate(out, cfg, manifest, workers):
    from openai import OpenAI
    if cfg.LLM_MODEL_NAME != manifest['generation_model']:
        raise ValueError('Generation model changed since manifest; use a new output directory.')
    rows = read(out / 'retrieved.json')
    validate_arm_rows(rows,manifest)
    saved = read(out / 'answers.json') if (out / 'answers.json').exists() else []
    done = {(r['arm'],r['index']) for r in saved}
    client = OpenAI(api_key=cfg.LLM_API_KEY, base_url=cfg.LLM_BASE_URL, timeout=90, max_retries=1)

    def one(row):
        context = '\n\n'.join(f'【知识来源{i+1}】\n{c}\n' for i,c in enumerate(row['contexts']))
        prompt = manifest['user_prompt_template'].replace('{query}',row['question']).replace('{context}',context)
        # Ground truth deliberately never enters generation messages.
        answer = row.get('_frozen_draft')
        if not answer:
            response = client.chat.completions.create(model=manifest['generation_model'],
                messages=[{'role':'system','content':manifest['arms'][row['arm']]['system_prompt']},
                          {'role':'user','content':prompt}],
                temperature=manifest['temperature'], max_tokens=manifest['max_tokens'])
            answer = (response.choices[0].message.content or '').strip()
            if not answer or response.choices[0].finish_reason != 'stop':
                raise RuntimeError(f"Empty/truncated generation {row['arm']}/{row['index']}")
        draft = answer
        review_prompt = manifest['arms'][row['arm']].get('review_prompt')
        if review_prompt:
            review_config = manifest['arms'][row['arm']]
            review_user = prompt
            if review_config.get('evidence_ids'):
                review_user = manifest['user_prompt_template'].replace('{query}',row['question']).replace('{context}',indexed_knowledge(context))
            review = client.chat.completions.create(model=manifest['generation_model'],
                messages=[{'role':'system','content':review_prompt},
                          {'role':'user','content':review_user+f'\n【待核查初稿（不是事实依据）】\n{draft}\n请核查并输出最终答案正文：'}],
                temperature=0.0,max_tokens=review_config.get('review_max_tokens',manifest['max_tokens']),
                **({'response_format':{'type':'json_object'}} if review_config.get('review_json') else {}))
            raw_review = (review.choices[0].message.content or '').strip()
            write(out/'reviews'/f"{row['arm']}_{row['index']}.json",{
                'index':row['index'],'raw':raw_review,'finish_reason':review.choices[0].finish_reason})
            answer = raw_review
            if review_config.get('review_json'):
                answer = validated_review_answer(raw_review,context)
            if not answer or review.choices[0].finish_reason != 'stop':
                raise RuntimeError(f"Empty/truncated review {row['arm']}/{row['index']}")
        return {**row, 'answer':answer, 'draft':draft, '_gen_model':manifest['generation_model'],
                'review_record':json.loads(raw_review) if review_prompt and review_config.get('review_json') else None}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one,r) for r in rows if (r['arm'],r['index']) not in done]
        failures = []
        for future in as_completed(futures):
            try:
                completed = future.result()
            except Exception as exc:
                failures.append(str(exc))
                print(f'Generation failed: {exc}',flush=True)
                continue
            saved.append(completed)
            saved.sort(key=lambda r:(r['index'],r['arm']))
            write(out / 'answers.json', saved)
            print(f"generate {len(saved)}/40 {completed['arm']} index={completed['index']}", flush=True)
        if failures:
            raise RuntimeError(f'{len(failures)} generations failed; valid answers retained. Resume generate. '+repr(failures))


def refine(out, manifest):
    """A second, separately recorded candidate; reuse frozen baseline answers."""
    target = out/'reviewed'
    if (target/'manifest.json').exists():
        raise ValueError('Refined manifest exists; resume its generate stage.')
    revised = copy.deepcopy(manifest)
    revised['created_at'] = datetime.now(timezone.utc).isoformat()
    revised['development_iteration'] = 2
    revised['arms']['candidate']['rerank'] = 2
    revised['arms']['candidate']['context_titles'] = True
    revised['arms']['candidate']['system_prompt'] = manifest['arms']['baseline']['system_prompt']
    revised['arms']['candidate']['review_prompt'] = LLMGenerator._GROUNDING_REVIEW_SYS
    revised['arms']['candidate'].update(review_json=True,review_max_tokens=3072,evidence_ids=True)
    rows = read(out/'retrieved.json')
    for row in rows:
        if row['arm'] == 'candidate':
            row['sources'] = row['sources'][:2]
            row['contexts'] = [f"【文档标题：{s['source']}】\n{ctx}"
                               for s,ctx in zip(row['sources'],row['contexts'][:2])]
    write(target/'retrieved.json',rows)
    write(target/'items.json',read(out/'items.json'))
    write(target/'answers.json',[r for r in read(out/'answers.json') if r['arm']=='baseline'])
    write(target/'manifest.json',revised)
    print(f'Refined pilot prepared: {target}',flush=True)


def anchor_pilot(out, manifest):
    target = out/'anchored'
    if (target/'manifest.json').exists():
        raise ValueError('Anchored manifest exists; resume its retrieve stage.')
    revised = copy.deepcopy(manifest)
    revised['created_at'] = datetime.now(timezone.utc).isoformat()
    revised['development_iteration'] = 3
    revised['arms']['candidate'] = {**manifest['arms']['baseline'],
        'title_anchor':True,'context_titles':True,'review_prompt':LLMGenerator._GROUNDING_REVIEW_SYS,
        'review_json':True,'review_max_tokens':3072,'evidence_ids':True}
    write(target/'manifest.json',revised)
    write(target/'items.json',read(out/'items.json'))
    write(target/'retrieved.json',[r for r in read(out/'retrieved.json') if r['arm']=='baseline'])
    write(target/'answers.json',[r for r in read(out/'answers.json') if r['arm']=='baseline'])
    print(f'Anchored pilot prepared: {target}',flush=True)


def judge_config(cfg, judge):
    judge_cfg = copy.copy(cfg)
    judge_cfg.LLM_MODEL_NAME = judge
    os.environ['LLM_JUDGE_TEMPERATURE'] = '0.0'
    os.environ['LLM_JUDGE_DISABLE_THINKING'] = 'false'
    os.environ.setdefault('LLM_JUDGE_REQUESTS_PER_SECOND', '0.25')
    return judge_cfg


def focused_pilot(out, manifest):
    target = out.parent/'focused'
    if (target/'manifest.json').exists():
        raise ValueError('Focused manifest exists; resume generate.')
    revised = copy.deepcopy(manifest)
    revised['created_at'] = datetime.now(timezone.utc).isoformat()
    revised['development_iteration'] = 7
    candidate = revised['arms']['candidate']
    for key in ['review_prompt','review_json','review_max_tokens','evidence_ids']:
        candidate.pop(key,None)
    candidate['system_prompt'] = revised['arms']['baseline']['system_prompt']+FOCUSED_RULES
    rows = read(out/'retrieved.json')
    for row in rows:
        row.pop('_frozen_draft',None)
    write(target/'manifest.json',revised)
    write(target/'items.json',read(out/'items.json'))
    write(target/'retrieved.json',rows)
    write(target/'answers.json',[r for r in read(out/'answers.json') if r['arm']=='baseline'])
    print(f'Focused pilot prepared: {target}',flush=True)


def revise_review(out, manifest, structured=False, evidence_ids=False):
    """Freeze an improved review prompt, keeping the same retrieval and drafts."""
    target = out.parent/('paragraph_review' if evidence_ids else 'evidence_review' if structured else 'final')
    if (target/'manifest.json').exists():
        raise ValueError('Final manifest exists; resume its generate stage.')
    revised = copy.deepcopy(manifest)
    revised['created_at'] = datetime.now(timezone.utc).isoformat()
    revised['development_iteration'] = 6 if evidence_ids else 5 if structured else 4
    revised['review_drafts_source_sha256'] = digest(out/'answers.json')
    revised['arms']['candidate']['review_prompt'] = LLMGenerator._GROUNDING_REVIEW_SYS
    revised['arms']['candidate'].update(review_json=True,review_max_tokens=3072,evidence_ids=True)
    if structured:
        revised['arms']['candidate'].update(review_json=True,review_max_tokens=3072)
    if evidence_ids:
        revised['arms']['candidate']['evidence_ids'] = True
    answers = read(out/'answers.json')
    validate_arm_rows(answers,manifest)
    rows = copy.deepcopy(answers)
    for row in rows:
        if row['arm'] == 'candidate':
            row['_frozen_draft'] = row['draft']
    write(target/'retrieved.json',rows)
    write(target/'items.json',read(out/'items.json'))
    write(target/'answers.json',[r for r in answers if r['arm']=='baseline'])
    write(target/'manifest.json',revised)
    print(f'Final review pilot prepared: {target}',flush=True)


def summarize(out, result, manifest):
    from src.online_service.rag_evaluator import valid_metric_score
    pairs = {}
    for s in result['scores']:
        pairs.setdefault(s['index'],{})[s['arm']] = s
    complete = [p for p in pairs.values() if all(a in p and valid_metric_score(p[a].get(k))
                for a in manifest['arms'] for k in METRICS)]
    summary = {'total_pairs':20, 'complete_pairs':len(complete),
               'target':{'faithfulness':[.85,.90], 'answer_relevancy':[.80,.90],
                         'context_recall':[.75,.85]},
               'context_precision_note':'Context precision is a ranked precision metric, not the screenshot context relevance metric.',
               'average':{},'delta':{},'paired_bootstrap_95ci':{}}
    for arm in manifest['arms']:
        summary['average'][arm] = {k:sum(p[arm][k] for p in complete)/len(complete)
                                  if complete else None for k in METRICS}
    rng = random.Random(42)
    for k in METRICS:
        if not complete:
            summary['delta'][k] = None
            continue
        differences = [p['candidate'][k]-p['baseline'][k] for p in complete]
        summary['delta'][k] = sum(differences)/len(differences)
        boots = sorted(sum(rng.choices(differences,k=len(differences)))/len(differences) for _ in range(5000))
        summary['paired_bootstrap_95ci'][k] = [boots[125], boots[4874]]
    write(out / 'comparison.json', summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if len(complete) != 20:
        raise RuntimeError('Incomplete pairs retained; run repair-score before claiming a complete comparison.')


def score(out, cfg, manifest, judge):
    from src.online_service.rag_evaluator import RAGEvaluator
    if judge == manifest['generation_model']:
        raise ValueError('Use a judge distinct from the generation model.')
    rows = read(out / 'answers.json')
    validate_arm_rows(rows,manifest)
    rows.sort(key=lambda r:(r['index'],r['arm']))
    if (out / 'scores.json').exists():
        raise ValueError('Scores already exist. Use repair-score to fill only missing cells.')
    evaluator = RAGEvaluator(judge_config(cfg,judge))
    if not evaluator._ragas:
        raise RuntimeError('Ragas unavailable; this comparison never substitutes a fallback judge.')
    # Call the Ragas path directly: any backend failure stays explicit.
    result = evaluator._evaluate_ragas(rows, show_progress=True)
    result['evaluation_config'] = {'judge_model':judge, 'generation_model':manifest['generation_model'],
        'judge_temperature':0.0, 'ragas_version':'0.2.6', 'embedding':'local BGE-M3 CUDA/fp16',
        'strictness':3, 'scope':manifest['scope'], 'provider_host':urlsplit(cfg.LLM_BASE_URL).hostname,
        'answer_relevancy_language':os.getenv('RAGAS_ANSWER_RELEVANCY_LANGUAGE','default')}
    for row, scored in zip(rows,result['scores']):
        scored.update(index=row['index'], arm=row['arm'])
    write(out / 'scores.json', result)
    summarize(out,result,manifest)


def adapt_score(out, cfg, manifest, judge):
    """Chinese AR for both arms; reuse only identical baseline F/CP/CR cells."""
    from src.online_service.rag_evaluator import valid_metric_score
    if (out/'scores.json').exists():
        raise ValueError('Adapted scores exist; use repair-score to resume missing cells.')
    original_out = out.parent
    original = read(original_out/'scores.json')
    if original['evaluation_config']['judge_model'] != judge:
        raise ValueError('The independent judge must remain unchanged.')
    original_answers = {(r['index'],r['arm']):r for r in read(original_out/'answers.json')}
    original_scores = {(r['index'],r['arm']):r for r in original['scores']}
    chinese_baseline = {}
    chinese_path = original_out/'final/scores.json'
    if out.name in ('evidence_review','paragraph_review','focused') and chinese_path.exists():
        chinese_result = read(chinese_path)
        chinese_answers = {r['index']:r for r in read(original_out/'final/answers.json') if r['arm']=='baseline'}
        if (chinese_result['evaluation_config']['judge_model'] == judge
                and chinese_result['evaluation_config']['answer_relevancy_language'] == 'chinese'):
            for index, row in chinese_answers.items():
                old = original_answers[(index,'baseline')]
                if any(old[key] != row[key] for key in ['question','answer','contexts','ground_truth']):
                    raise ValueError('Chinese baseline reuse requires identical baseline evidence.')
            chinese_baseline = {r['index']:r for r in chinese_result['scores'] if r['arm']=='baseline'}
    rows = read(out/'answers.json')
    validate_arm_rows(rows,manifest)
    context_scores = {}
    context_path = original_out/'paragraph_review/scores.json'
    if out.name == 'focused' and context_path.exists():
        context_result = read(context_path)
        if context_result['evaluation_config']['judge_model'] != judge:
            raise ValueError('Context scoring reuse requires the same independent judge.')
        context_answers = {(r['index'],r['arm']):r for r in read(original_out/'paragraph_review/answers.json')}
        for row in rows:
            if row['arm'] == 'candidate':
                old = context_answers[(row['index'],'candidate')]
                if any(old[key] != row[key] for key in ['question','contexts','ground_truth']):
                    raise ValueError('Context metric reuse requires identical question, contexts and reference.')
        context_scores = {r['index']:r for r in context_result['scores'] if r['arm']=='candidate'}
    scored = []
    for row in sorted(rows,key=lambda r:(r['index'],r['arm'])):
        score_row = {'index':row['index'],'arm':row['arm'],'question':row['question']}
        for key in METRICS:
            score_row[key] = None
        if row['arm'] == 'baseline':
            old = original_answers[(row['index'],'baseline')]
            if any(old[key] != row[key] for key in ['question','answer','contexts','ground_truth']):
                raise ValueError('Baseline evidence changed; previous baseline scores cannot be reused.')
            source_score = original_scores[(row['index'],'baseline')]
            for key in METRICS:
                if key != 'answer_relevancy':
                    if not valid_metric_score(source_score.get(key)):
                        raise ValueError('Complete original baseline scores are required.')
                    score_row[key] = source_score[key]
            score_row['original_default_answer_relevancy'] = source_score['answer_relevancy']
            adapted = chinese_baseline.get(row['index'],{}).get('answer_relevancy')
            if valid_metric_score(adapted):
                score_row['answer_relevancy'] = adapted
        else:
            for key in ['context_precision','context_recall']:
                prior = context_scores.get(row['index'],{}).get(key)
                if valid_metric_score(prior):
                    score_row[key] = prior
        scored.append(score_row)
    result = {'engine':'ragas','total':40,'metrics':METRICS,'has_ground_truth':True,
              'scores':scored,'evaluation_config':{
        **original['evaluation_config'],'answer_relevancy_language':'chinese',
        'provider_host':urlsplit(cfg.LLM_BASE_URL).hostname,
        'baseline_reuse':'Only identical baseline question/answer/contexts/reference F, CP and CR; AR rerun for both arms.',
        'original_scores_sha256':digest(original_out/'scores.json'),
        'answers_sha256':digest(out/'answers.json'),
        'manifest_sha256':digest(out/'manifest.json')}}
    if chinese_baseline:
        result['evaluation_config']['chinese_baseline_source_sha256'] = digest(chinese_path)
    if context_scores:
        result['evaluation_config']['context_only_reuse_source_sha256'] = digest(context_path)
        result['evaluation_config']['context_only_reuse'] = 'CP/CR depend only on identical question, contexts and reference; F/AR newly evaluated.'
    write(out/'scores.json',result)
    repair_score(out,cfg,manifest,judge)


def repair_score(out, cfg, manifest, judge):
    """Retry missing metric cells with the original judge; keep every valid cell."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall
    from ragas.run_config import RunConfig
    from src.online_service.rag_evaluator import RAGEvaluator, valid_metric_score, answer_relevancy_metric
    result = read(out/'scores.json')
    if result['evaluation_config']['judge_model'] != judge:
        raise ValueError('Repair requires the original judge.')
    answers = {(r['index'],r['arm']):r for r in read(out/'answers.json')}
    validate_arm_rows(list(answers.values()),manifest)
    validate_arm_rows(result['scores'],manifest)
    recorded_hash = result['evaluation_config'].get('answers_sha256')
    if recorded_hash and digest(out/'answers.json') != recorded_hash:
        raise ValueError('Answers changed after scoring began; use a new experiment directory.')
    language = result['evaluation_config'].get('answer_relevancy_language','default')
    groups = {}
    for s in result['scores']:
        missing = tuple(k for k in METRICS if not valid_metric_score(s.get(k)))
        if missing: groups.setdefault(missing,[]).append(s)
    evaluator = RAGEvaluator(judge_config(cfg,judge))
    embeddings = evaluator._build_embeddings() if groups else None
    factories = dict(zip(METRICS,[Faithfulness,lambda:answer_relevancy_metric(language),ContextPrecision,ContextRecall]))
    for missing, scored_rows in groups.items():
        # Short reverse-question prompts fit a faster request rate; long context
        # judgments retain 0.25 rps to respect the provider token quota.
        rate = 1.0 if missing == ('answer_relevancy',) and language == 'chinese' else 0.25
        os.environ['LLM_JUDGE_REQUESTS_PER_SECOND'] = str(rate)
        llm = evaluator._build_llm()
        rows = [answers[(s['index'],s['arm'])] for s in scored_rows]
        dataset = Dataset.from_list([{k:r[k] for k in ['question','answer','contexts','ground_truth']} for r in rows])
        print(f'Repair {len(rows)} rows: {missing}',flush=True)
        repaired = evaluate(dataset=dataset, metrics=[factories[k]() for k in missing],
            llm=llm, embeddings=embeddings, run_config=RunConfig(timeout=180,max_retries=2,
                max_wait=30,max_workers=2,seed=42),show_progress=True,raise_exceptions=False)
        _, values = evaluator._extract(repaired)
        if len(values) != len(scored_rows): raise RuntimeError('Repair row count mismatch')
        for s,v in zip(scored_rows,values):
            for k in missing:
                if valid_metric_score(v.get(k)): s[k] = round(float(v[k]),4)
        write(out/'scores.json',result)
    result['valid_counts'] = {k:sum(valid_metric_score(s.get(k)) for s in result['scores']) for k in METRICS}
    result['complete_count'] = sum(all(valid_metric_score(s.get(k)) for k in METRICS) for s in result['scores'])
    result['average'] = {k:sum(s[k] for s in result['scores'] if valid_metric_score(s.get(k)))/result['valid_counts'][k]
                          if result['valid_counts'][k] else None for k in METRICS}
    result['repair'] = {'judge_model':judge,'policy':'Only missing cells filled; previous valid scores unchanged',
                        'requests_per_second':float(os.environ['LLM_JUDGE_REQUESTS_PER_SECOND'])}
    write(out/'scores.json',result)
    summarize(out,result,manifest)


def reference_summary():
    """Public optimization targets are separate from measured experiment evidence."""
    return {
        'kind': 'reference_targets',
        'measured': False,
        'description': '参考目标值；用于优化规划，不代表项目实测结果。',
        'source': 'User-provided reference ranges',
        'reference_targets': {
            'context_relevance': {'label': '上下文相关性', 'range': [0.65, 0.80],
                                  'target': 0.79, 'computed_by_current_evaluator': False},
            'context_recall': {'label': '上下文召回率', 'range': [0.75, 0.85], 'target': 0.84},
            'faithfulness': {'label': '忠实度', 'range': [0.85, 0.90], 'target': 0.89},
            'answer_relevancy': {'label': '答案相关性', 'range': [0.80, 0.90], 'target': 0.89},
        },
        'current_evaluator_metrics': METRICS,
        'definition_note': 'Context Precision 是上下文排序精确度，与上下文相关性分别定义；不互换数值。',
        'measured_records': 'Preserved in private experiment artifacts; public values are targets only.',
    }


def publish_summary(out, manifest):
    """Archive measured evidence privately and publish only labeled reference targets."""
    scored = read(out/'scores.json')
    validate_arm_rows(scored['scores'],manifest)
    comparison = read(out/'comparison.json')
    from src.online_service.rag_evaluator import valid_metric_score
    if (comparison['complete_pairs'] != 20
            or not all(valid_metric_score(row.get(k)) for row in scored['scores'] for k in METRICS)):
        raise ValueError('Only a complete twenty-pair comparison can be published.')
    summary = {'created_at':datetime.now(timezone.utc).isoformat(),
        'sample_size':20,'indices_1based':manifest['indices_1based'],'seed':manifest['seed'],
        'scope':manifest['scope'],'evaluation_config':scored['evaluation_config'],
        'arms':{arm:{**{k:v for k,v in params.items() if k not in ['system_prompt','review_prompt']},
                     'system_prompt_sha256':hashlib.sha256(params['system_prompt'].encode('utf-8')).hexdigest()}
                for arm,params in manifest['arms'].items()},
        'comparison':comparison,'manifest_sha256':digest(out/'manifest.json'),
        'answers_sha256':digest(out/'answers.json'),'scores_sha256':digest(out/'scores.json'),
        'development_sample':True,'full_210_evaluation':False}
    raw_path = out.parent/'comparison.json'
    review_path = out.parent/'paragraph_review/comparison.json'
    if raw_path.exists():
        summary['original_default_prompt_comparison'] = read(raw_path)
    if review_path.exists() and review_path.parent != out:
        summary['structured_review_comparison'] = read(review_path)
    write(out/'measured_summary.json',summary)
    write(ROOT/'docs/quality_pilot_20261010.json',reference_summary())
    print('Archived measured summary privately; published reference targets to docs/quality_pilot_20261010.json',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','retrieve','generate','score','repair-score','adapt-score','refine','anchor','revise-review','audit-review','paragraph-review','focused','publish'],required=True)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/quality_20261010')
    parser.add_argument('--workers',type=int,choices=[1,2,3,4],default=3)
    parser.add_argument('--judge-model',default='Qwen/Qwen3-30B-A3B-Instruct-2507')
    parser.add_argument('--ar-language',choices=['default','chinese'],default='chinese')
    parser.add_argument('--variant',choices=['focused','title-only','review'],default='title-only',
                        help='New manifests only; existing experiment prompts remain frozen.')
    args = parser.parse_args()
    cfg = Config()
    manifest = prepare(args.out,cfg,args.variant)
    if args.stage == 'retrieve': retrieve(args.out,cfg,manifest)
    elif args.stage == 'generate': generate(args.out,cfg,manifest,args.workers)
    elif args.stage == 'score':
        os.environ['RAGAS_ANSWER_RELEVANCY_LANGUAGE'] = args.ar_language
        score(args.out,cfg,manifest,args.judge_model)
    elif args.stage == 'repair-score': repair_score(args.out,cfg,manifest,args.judge_model)
    elif args.stage == 'adapt-score': adapt_score(args.out,cfg,manifest,args.judge_model)
    elif args.stage == 'refine': refine(args.out,manifest)
    elif args.stage == 'anchor': anchor_pilot(args.out,manifest)
    elif args.stage == 'revise-review': revise_review(args.out,manifest)
    elif args.stage == 'audit-review': revise_review(args.out,manifest,structured=True)
    elif args.stage == 'paragraph-review': revise_review(args.out,manifest,structured=True,evidence_ids=True)
    elif args.stage == 'focused': focused_pilot(args.out,manifest)
    elif args.stage == 'publish': publish_summary(args.out,manifest)


if __name__ == '__main__':
    main()

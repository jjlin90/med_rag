import json, glob

M = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']

def valid(r):
    return any((r.get(k) or 0) > 0 for k in M)

q2scores = {}
for f in sorted(glob.glob('data/test_query/chunks/chunk_*.json')):
    d = json.load(open(f, encoding='utf-8'))
    for x in d.get('scores', []):
        if not valid(x):
            continue
        q = x['question']
        if q not in q2scores:
            q2scores[q] = {k: x.get(k, 0) for k in M}

ans = json.load(open('data/test_query/eval_answers_210.json', encoding='utf-8'))
st = json.load(open('data/test_query/eval_set_210.json', encoding='utf-8'))
gt = {x['question']: x.get('reference_answer', '') for x in st}
ans_map = {a['question']: a for a in ans}

# 取 F=0 且为真实低分的样本（按 answer 长度排序，看长答案是否更易 0）
zero = [q for q, s in q2scores.items() if s['faithfulness'] == 0]
print(f'F=0 样本数: {len(zero)}')
print('=' * 80)

for q in zero[:4]:
    a = ans_map[q]
    answer = a.get('answer', '')
    ctxs = a.get('contexts', [])
    score = q2scores[q]
    print(f'\n【Q】{q[:80]}')
    print(f'【评分】F={score["faithfulness"]} AR={score["answer_relevancy"]:.2f} CP={score["context_precision"]:.2f} CR={score["context_recall"]:.2f}')
    print(f'【答案长度】{len(answer)} 字 | 上下文条数 {len(ctxs)}，总长 {sum(len(c) for c in ctxs)} 字')
    print(f'--- 答案（前400字）---')
    print(answer[:400])
    print(f'--- 第1条上下文（前300字）---')
    if ctxs:
        print(ctxs[0][:300])
    print('-' * 80)

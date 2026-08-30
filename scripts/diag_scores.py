import json, glob
from collections import defaultdict

M = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']

def valid(r):
    return any((r.get(k) or 0) > 0 for k in M)

q2scores = {}
q2judge = {}
for f in sorted(glob.glob('data/test_query/chunks/chunk_*.json')):
    d = json.load(open(f, encoding='utf-8'))
    m = d.get('_meta', {}).get('model')
    for x in d.get('scores', []):
        if not valid(x):
            continue
        q = x['question']
        if q not in q2scores:
            q2scores[q] = {k: x.get(k, 0) for k in M}
            q2judge[q] = m

st = json.load(open('data/test_query/eval_set_210.json', encoding='utf-8'))
order = [x['question'] for x in st]
present = [q for q in order if q in q2scores]
first44 = present[:44]
rest = [q for q in present if q not in set(first44)]

def mean(sub, key):
    vs = [q2scores[q][key] for q in sub]
    return sum(vs) / len(vs) if vs else 0

print('=== 样本顺序效应：前44 vs 后165 ===')
print(f'{"":<22}{"前44":>10}{"后165":>10}')
for k in M:
    print(f'{k:<22}{mean(first44,k):>10.3f}{mean(rest,k):>10.3f}')
print(f'{"weighted":<22}{sum(mean(first44,k) for k in M)/4:>10.3f}{sum(mean(rest,k) for k in M)/4:>10.3f}')

print()
print('=== faithfulness 分布（209条）===')
fs = [v['faithfulness'] for v in q2scores.values()]
buckets = {'=0': 0, '0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1': 0}
for v in fs:
    if v == 0:
        buckets['=0'] += 1
    elif v < 0.2:
        buckets['0-0.2'] += 1
    elif v < 0.4:
        buckets['0.2-0.4'] += 1
    elif v < 0.6:
        buckets['0.4-0.6'] += 1
    elif v < 0.8:
        buckets['0.6-0.8'] += 1
    else:
        buckets['0.8-1'] += 1
for b, c in buckets.items():
    print(f'  F {b:<8}: {c:>3} ({100*c/len(fs):.0f}%)')
low = buckets['=0'] + buckets['0-0.2'] + buckets['0.2-0.4']
print(f'  F<0.4 合计: {low} ({100*low/len(fs):.0f}%)')

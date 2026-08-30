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

items = list(q2scores.values())
n = len(items)

# F=0 中，CP=0 且 CR=0 的比例
f0 = [s for s in items if s['faithfulness'] == 0]
f0_retrieval_bad = [s for s in f0 if s['context_precision'] == 0 and s['context_recall'] == 0]
print(f'F=0 总数: {len(f0)}')
print(f'  其中 CP=0 且 CR=0 (检索完全错位): {len(f0_retrieval_bad)} ({100*len(f0_retrieval_bad)/len(f0):.0f}%)')
print(f'  其中 CP>0 或 CR>0 (检索还行但答案不忠实): {len(f0)-len(f0_retrieval_bad)} ({100*(len(f0)-len(f0_retrieval_bad))/len(f0):.0f}%)')

# F>0.6 的检索质量
fhi = [s for s in items if s['faithfulness'] > 0.6]
if fhi:
    print(f'\nF>0.6 的样本 ({len(fhi)}): CP均={sum(s["context_precision"] for s in fhi)/len(fhi):.3f}  CR均={sum(s["context_recall"] for s in fhi)/len(fhi):.3f}  AR均={sum(s["answer_relevancy"] for s in fhi)/len(fhi):.3f}')

# 整体 AR 与 CP/CR 对比 —— 答案相关性 vs 检索质量
print(f'\n=== 全局均值对比 ===')
for k in M:
    print(f'  {k:<20}{sum(s[k] for s in items)/n:.3f}')

# AR 高但 F 低（答案相关但被判不忠实）的占比
ar_hi_f_lo = [s for s in items if s['answer_relevancy'] > 0.7 and s['faithfulness'] < 0.4]
print(f'\nAR>0.7 且 F<0.4（答案相关但被认为不忠实）: {len(ar_hi_f_lo)} ({100*len(ar_hi_f_lo)/n:.0f}%)')

# CP/CR 至少一项>0 的样本里，F 的分布
good_retrieval = [s for s in items if s['context_precision'] > 0 or s['context_recall'] > 0]
if good_retrieval:
    print(f'\n检索至少部分命中 ({len(good_retrieval)}) 的 F 均值: {sum(s["faithfulness"] for s in good_retrieval)/len(good_retrieval):.3f}')
bad_retrieval = [s for s in items if s['context_precision'] == 0 and s['context_recall'] == 0]
if bad_retrieval:
    print(f'检索完全未命中 ({len(bad_retrieval)}) 的 F 均值: {sum(s["faithfulness"] for s in bad_retrieval)/len(bad_retrieval):.3f}')

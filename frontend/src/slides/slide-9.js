window.slideDataMap.set(9, `
<div class="w-[1440px] h-[810px] shadow-2xl relative overflow-hidden slide-bg">
  <div class="absolute inset-0 opacity-12" style="background-image: linear-gradient(rgba(129,140,248,0.12) 1px, transparent 1px), linear-gradient(90deg, rgba(129,140,248,0.12) 1px, transparent 1px); background-size: 44px 44px;"></div>
  <div class="absolute -top-16 right-1/4 w-80 h-80 bg-indigo-500 rounded-full blur-3xl opacity-[0.06]"></div>
  <div class="relative z-10 w-[1350px] h-[720px] mx-auto my-[45px] flex flex-col">
    <div class="mb-5 flex items-center gap-4">
      <div class="w-12 h-1 bg-cyan-400"></div>
      <h1 class="font-title text-4xl font-bold text-white">在线检索流程</h1>
      <span class="ml-2 text-slate-400 text-lg font-light">七步在线链路 · 命中即返回</span>
    </div>
    <div class="grid grid-cols-4 gap-4 flex-1">
      <div class="bg-slate-800/50 border border-cyan-500/30 rounded-xl p-5 flex flex-col">
        <div class="flex items-center gap-3 mb-2"><span class="font-mono text-2xl font-bold text-cyan-300">01</span><h3 class="text-lg font-bold text-white">查询缓存</h3></div>
        <p class="text-slate-400 text-sm leading-relaxed">Redis 查询缓存区分问题、来源、策略与历史；启用且可用时，复用有答案且降级等级低于 L2 的缓存结果。</p>
      </div>
      <div class="bg-slate-800/50 border border-cyan-500/30 rounded-xl p-5 flex flex-col">
        <div class="flex items-center gap-3 mb-2"><span class="font-mono text-2xl font-bold text-cyan-300">02</span><h3 class="text-lg font-bold text-white">条件 FAQ</h3></div>
        <p class="text-slate-400 text-sm leading-relaxed">无历史/来源/策略限制时查询；先查 FAQ 缓存，再用 BM25 匹配题库并从 MySQL 取答案，命中即返回。</p>
      </div>
      <div class="bg-slate-800/50 border border-cyan-500/30 rounded-xl p-5 flex flex-col">
        <div class="flex items-center gap-3 mb-2"><span class="font-mono text-2xl font-bold text-cyan-300">03</span><h3 class="text-lg font-bold text-white">意图分类</h3></div>
        <p class="text-slate-400 text-sm leading-relaxed">未命中或跳过快通道后由 BERT 分类；<span class="text-cyan-300">general</span> 用通用提示词直答，<span class="text-cyan-300">medical</span> 继续检索链路。</p>
      </div>
      <div class="bg-slate-800/50 border border-indigo-500/30 rounded-xl p-5 flex flex-col">
        <div class="flex items-center gap-3 mb-2"><span class="font-mono text-2xl font-bold text-indigo-300">04</span><h3 class="text-lg font-bold text-white">策略选择</h3></div>
        <p class="text-slate-400 text-sm leading-relaxed">显式策略优先；否则由 LLM 选择 direct / hyde / subquery / backtracking。</p>
      </div>
      <div class="bg-slate-800/50 border border-indigo-500/30 rounded-xl p-5 flex flex-col">
        <div class="flex items-center gap-3 mb-2"><span class="font-mono text-2xl font-bold text-indigo-300">05</span><h3 class="text-lg font-bold text-white">检索合并</h3></div>
        <p class="text-slate-400 text-sm leading-relaxed">Milvus 融合 Top16，取前5子块并回溯父块。</p>
      </div>
      <div class="bg-slate-800/50 border border-purple-500/30 rounded-xl p-5 flex flex-col">
        <div class="flex items-center gap-3 mb-2"><span class="font-mono text-2xl font-bold text-purple-300">06</span><h3 class="text-lg font-bold text-white">重排序</h3></div>
        <p class="text-slate-400 text-sm leading-relaxed">BGE-reranker 按子证据重排，最终最多 Top2。</p>
      </div>
      <div class="col-span-2 bg-slate-800/50 border border-purple-500/30 rounded-xl p-5 flex flex-col">
        <div class="flex items-center gap-3 mb-2"><span class="font-mono text-2xl font-bold text-purple-300">07</span><h3 class="text-lg font-bold text-white">生成</h3></div>
        <p class="text-slate-400 text-sm leading-relaxed">医疗 Prompt + 多轮历史 → {答案, 意图, 策略, 来源}。</p>
      </div>
    </div>
  </div>
</div>
`);

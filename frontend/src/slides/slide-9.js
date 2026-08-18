window.slideDataMap.set(9, `
<div class="w-[1440px] h-[810px] shadow-2xl relative overflow-hidden slide-bg">
  <div class="absolute inset-0 opacity-12" style="background-image: linear-gradient(rgba(129,140,248,0.12) 1px, transparent 1px), linear-gradient(90deg, rgba(129,140,248,0.12) 1px, transparent 1px); background-size: 44px 44px;"></div>
  <div class="absolute -top-16 right-1/4 w-80 h-80 bg-indigo-500 rounded-full blur-3xl opacity-12"></div>
  <div class="relative z-10 w-[1350px] h-[720px] mx-auto my-[45px] flex flex-col">
    <div class="mb-5 flex items-center gap-4">
      <div class="w-12 h-1 bg-cyan-400"></div>
      <h1 class="font-title text-4xl font-bold text-white">在线检索流程</h1>
      <span class="ml-2 text-slate-400 text-lg font-light">六步 → 可信答案</span>
    </div>
    <div class="grid grid-cols-3 gap-4 flex-1">
      <div class="bg-slate-800/50 border border-cyan-500/30 rounded-xl p-5 flex flex-col">
        <div class="flex items-center gap-3 mb-2"><span class="font-mono text-2xl font-bold text-cyan-300">01</span><h3 class="text-lg font-bold text-white">FAQ 缓存</h3></div>
        <p class="text-slate-400 text-sm leading-relaxed">Redis 命中即直答；未命中查 MySQL BM25。</p>
      </div>
      <div class="bg-slate-800/50 border border-cyan-500/30 rounded-xl p-5 flex flex-col">
        <div class="flex items-center gap-3 mb-2"><span class="font-mono text-2xl font-bold text-cyan-300">02</span><h3 class="text-lg font-bold text-white">意图分类</h3></div>
        <p class="text-slate-400 text-sm leading-relaxed">BERT 区分 <span class="text-cyan-300">general</span> / <span class="text-cyan-300">medical</span>。</p>
      </div>
      <div class="bg-slate-800/50 border border-indigo-500/30 rounded-xl p-5 flex flex-col">
        <div class="flex items-center gap-3 mb-2"><span class="font-mono text-2xl font-bold text-indigo-300">03</span><h3 class="text-lg font-bold text-white">策略选择</h3></div>
        <p class="text-slate-400 text-sm leading-relaxed">LLM 自动选 direct / hyde / subquery / backtracking。</p>
      </div>
      <div class="bg-slate-800/50 border border-indigo-500/30 rounded-xl p-5 flex flex-col">
        <div class="flex items-center gap-3 mb-2"><span class="font-mono text-2xl font-bold text-indigo-300">04</span><h3 class="text-lg font-bold text-white">检索合并</h3></div>
        <p class="text-slate-400 text-sm leading-relaxed">Milvus 混合检索，粗排 Top8 候选。</p>
      </div>
      <div class="bg-slate-800/50 border border-purple-500/30 rounded-xl p-5 flex flex-col">
        <div class="flex items-center gap-3 mb-2"><span class="font-mono text-2xl font-bold text-purple-300">05</span><h3 class="text-lg font-bold text-white">重排序</h3></div>
        <p class="text-slate-400 text-sm leading-relaxed">BGE-reranker 交叉编码精排 Top4。</p>
      </div>
      <div class="bg-slate-800/50 border border-purple-500/30 rounded-xl p-5 flex flex-col">
        <div class="flex items-center gap-3 mb-2"><span class="font-mono text-2xl font-bold text-purple-300">06</span><h3 class="text-lg font-bold text-white">生成</h3></div>
        <p class="text-slate-400 text-sm leading-relaxed">医疗 Prompt + 多轮历史 → {答案, 意图, 策略, 来源}。</p>
      </div>
    </div>
  </div>
</div>
`);

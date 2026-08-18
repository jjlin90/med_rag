window.slideDataMap.set(10, `
<div class="w-[1440px] h-[810px] shadow-2xl relative overflow-hidden slide-bg">
  <div class="absolute inset-0 opacity-12" style="background-image: linear-gradient(rgba(34,211,238,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,0.1) 1px, transparent 1px); background-size: 44px 44px;"></div>
  <div class="absolute -top-16 left-1/4 w-80 h-80 bg-cyan-500 rounded-full blur-3xl opacity-12"></div>
  <div class="relative z-10 w-[1350px] h-[720px] mx-auto my-[45px] flex flex-col">
    <div class="mb-6 flex items-center gap-4">
      <div class="w-12 h-1 bg-cyan-400"></div>
      <h1 class="font-title text-4xl font-bold text-white">策略与重排</h1>
      <span class="ml-2 text-slate-400 text-lg font-light">为什么检索得准</span>
    </div>
    <div class="grid grid-cols-2 gap-6 flex-1">
      <div class="bg-slate-800/40 border border-cyan-500/30 rounded-xl p-6">
        <h3 class="text-xl font-bold text-cyan-300 mb-4">四种检索策略（LLM 自动选）</h3>
        <div class="grid grid-cols-2 gap-4">
          <div class="bg-slate-900/40 rounded-lg p-4 border-l-4 border-cyan-400"><p class="text-white font-semibold">direct</p><p class="text-slate-400 text-sm mt-1">直接向量检索</p></div>
          <div class="bg-slate-900/40 rounded-lg p-4 border-l-4 border-cyan-400"><p class="text-white font-semibold">hyde</p><p class="text-slate-400 text-sm mt-1">假设性文档增强</p></div>
          <div class="bg-slate-900/40 rounded-lg p-4 border-l-4 border-cyan-400"><p class="text-white font-semibold">subquery</p><p class="text-slate-400 text-sm mt-1">子问题拆解</p></div>
          <div class="bg-slate-900/40 rounded-lg p-4 border-l-4 border-cyan-400"><p class="text-white font-semibold">backtracking</p><p class="text-slate-400 text-sm mt-1">回溯纠错</p></div>
        </div>
      </div>
      <div class="bg-slate-800/40 border border-indigo-500/30 rounded-xl p-6 flex flex-col">
        <h3 class="text-xl font-bold text-indigo-300 mb-4">重排序与混合检索</h3>
        <div class="space-y-4 flex-1">
          <div class="bg-slate-900/40 rounded-lg p-4 border-l-4 border-indigo-400">
            <p class="text-white font-semibold mb-1">BGE-reranker 精排</p>
            <p class="text-slate-400 text-sm">CrossEncoder 对 Top8 候选做精细相关性打分，输出 Top4。</p>
          </div>
          <div class="bg-slate-900/40 rounded-lg p-4 border-l-4 border-indigo-400">
            <p class="text-white font-semibold mb-1">混合检索</p>
            <p class="text-slate-400 text-sm">dense（语义）+ sparse（关键词）+ multi-vector 三者融合，提升医学同义表述召回。</p>
          </div>
          <div class="bg-slate-900/40 rounded-lg p-4 border-l-4 border-emerald-400">
            <p class="text-emerald-300 font-semibold">收益</p>
            <p class="text-slate-400 text-sm">相比单向量 + 无重排，相关性 / 可信度显著提升。</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
`);

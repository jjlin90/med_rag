window.slideDataMap.set(12, `
<div class="w-[1440px] h-[810px] shadow-2xl relative overflow-hidden slide-bg">
  <div class="absolute inset-0 opacity-12" style="background-image: linear-gradient(rgba(129,140,248,0.12) 1px, transparent 1px), linear-gradient(90deg, rgba(129,140,248,0.12) 1px, transparent 1px); background-size: 44px 44px;"></div>
  <div class="absolute -bottom-16 left-1/4 w-80 h-80 bg-purple-500 rounded-full blur-3xl opacity-12"></div>
  <div class="relative z-10 w-[1350px] h-[720px] mx-auto my-[45px] flex flex-col">
    <div class="mb-5 flex items-center gap-4">
      <div class="w-12 h-1 bg-cyan-400"></div>
      <h1 class="font-title text-4xl font-bold text-white">智能路由</h1>
      <span class="ml-2 text-slate-400 text-lg font-light">让 Naive RAG 具备判断力</span>
    </div>
    <div class="grid grid-cols-5 gap-5 flex-1">
      <div class="col-span-2 bg-slate-800/50 border border-cyan-500/30 rounded-xl p-6 flex flex-col justify-center">
        <h3 class="text-lg font-bold text-cyan-300 mb-4">路由决策</h3>
        <div class="space-y-3 text-base">
          <div class="bg-slate-900/40 rounded-lg px-4 py-3 text-white">用户提问</div>
          <div class="text-center text-cyan-400 font-mono">↓ BERT 意图分类</div>
          <div class="grid grid-cols-2 gap-3">
            <div class="bg-cyan-500/10 border border-cyan-400/50 rounded-lg px-3 py-2 text-center text-cyan-200">medical<br><span class="text-xs text-slate-400">→ 直走 RAG</span></div>
            <div class="bg-indigo-500/10 border border-indigo-400/50 rounded-lg px-3 py-2 text-center text-indigo-200">general<br><span class="text-xs text-slate-400">→ FAQ/直答</span></div>
          </div>
        </div>
      </div>
      <div class="col-span-3 bg-slate-800/40 border border-indigo-500/30 rounded-xl p-6 flex flex-col">
        <h3 class="text-lg font-bold text-indigo-300 mb-4">关键设计</h3>
        <div class="space-y-3 flex-1">
          <div class="bg-slate-900/40 rounded-lg p-4 border-l-4 border-cyan-400"><p class="text-white font-semibold">意图分类</p><p class="text-slate-400 text-sm">bert-base-chinese 微调，输出 general/medical 及置信度。</p></div>
          <div class="bg-slate-900/40 rounded-lg p-4 border-l-4 border-cyan-400"><p class="text-white font-semibold">路由规则</p><p class="text-slate-400 text-sm">无历史/来源/策略限制先走FAQ；未命中再分类并进入RAG。</p></div>
          <div class="bg-slate-900/40 rounded-lg p-4 border-l-4 border-emerald-400"><p class="text-emerald-300 font-semibold">FAQ 守卫</p><p class="text-slate-400 text-sm">原始BM25为零则回退RAG；正分仍需通过0.85相对阈值，并非正确率保证。</p></div>
        </div>
      </div>
    </div>
  </div>
</div>
`);

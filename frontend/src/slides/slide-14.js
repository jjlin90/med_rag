window.slideDataMap.set(14, `
<div class="w-[1440px] h-[810px] shadow-2xl relative overflow-hidden slide-bg">
  <div class="absolute inset-0 opacity-12" style="background-image: linear-gradient(rgba(129,140,248,0.12) 1px, transparent 1px), linear-gradient(90deg, rgba(129,140,248,0.12) 1px, transparent 1px); background-size: 44px 44px;"></div>
  <div class="absolute -bottom-16 right-1/4 w-80 h-80 bg-indigo-500 rounded-full blur-3xl opacity-12"></div>
  <div class="relative z-10 w-[1350px] h-[720px] mx-auto my-[45px] flex flex-col">
    <div class="mb-6 flex items-center gap-4">
      <div class="w-12 h-1 bg-cyan-400"></div>
      <h1 class="font-title text-4xl font-bold text-white">质量评估</h1>
      <span class="ml-2 text-slate-400 text-lg font-light">Ragas 四指标量化看板</span>
    </div>
    <div class="grid grid-cols-4 gap-5 flex-1">
      <div class="bg-slate-800/50 border border-cyan-500/30 rounded-xl p-6 flex flex-col">
        <div class="w-12 h-12 rounded-full bg-cyan-500/20 border border-cyan-400/50 flex items-center justify-center font-mono text-cyan-300 text-sm mb-4">F</div>
        <h3 class="text-base font-bold text-white mb-2">faithfulness</h3>
        <p class="text-slate-400 text-sm leading-relaxed flex-1">忠实度：答案不杜撰、不偏离来源。</p>
      </div>
      <div class="bg-slate-800/50 border border-indigo-500/30 rounded-xl p-6 flex flex-col">
        <div class="w-12 h-12 rounded-full bg-indigo-500/20 border border-indigo-400/50 flex items-center justify-center font-mono text-indigo-300 text-sm mb-4">R</div>
        <h3 class="text-base font-bold text-white mb-2">answer_relevancy</h3>
        <p class="text-slate-400 text-sm leading-relaxed flex-1">回答相关性：紧扣用户问题。</p>
      </div>
      <div class="bg-slate-800/50 border border-purple-500/30 rounded-xl p-6 flex flex-col">
        <div class="w-12 h-12 rounded-full bg-purple-500/20 border border-purple-400/50 flex items-center justify-center font-mono text-purple-300 text-sm mb-4">P</div>
        <h3 class="text-base font-bold text-white mb-2">context_precision</h3>
        <p class="text-slate-400 text-sm leading-relaxed flex-1">上下文精确度：召回片段相关。</p>
      </div>
      <div class="bg-slate-800/50 border border-cyan-500/30 rounded-xl p-6 flex flex-col">
        <div class="w-12 h-12 rounded-full bg-cyan-500/20 border border-cyan-400/50 flex items-center justify-center font-mono text-cyan-300 text-sm mb-4">C</div>
        <h3 class="text-base font-bold text-white mb-2">context_recall</h3>
        <p class="text-slate-400 text-sm leading-relaxed flex-1">上下文召回率：覆盖应检索内容。</p>
      </div>
    </div>
    <div class="mt-5 grid grid-cols-2 gap-5">
      <div class="bg-slate-900/40 border border-slate-700 rounded-lg px-6 py-4"><p class="text-slate-300 text-base"><span class="text-cyan-300 font-semibold">闭环：</span>离线评估 → 定位弱项 → 调参（分块/策略/重排）→ 复测。</p></div>
      <div class="bg-slate-900/40 border border-slate-700 rounded-lg px-6 py-4"><p class="text-slate-300 text-base"><span class="text-cyan-300 font-semibold">下一步：</span>持续采集评估分数，建立量化看板指导迭代。</p></div>
    </div>
  </div>
</div>
`);

window.slideDataMap.set(16, `
<div class="w-[1440px] h-[810px] shadow-2xl relative overflow-hidden slide-bg">
  <div class="absolute inset-0 opacity-12" style="background-image: linear-gradient(rgba(129,140,248,0.12) 1px, transparent 1px), linear-gradient(90deg, rgba(129,140,248,0.12) 1px, transparent 1px); background-size: 44px 44px;"></div>
  <div class="absolute -bottom-16 left-1/4 w-80 h-80 bg-indigo-500 rounded-full blur-3xl opacity-12"></div>
  <div class="relative z-10 w-[1350px] h-[720px] mx-auto my-[45px] flex flex-col">
    <div class="mb-6 flex items-center gap-4">
      <div class="w-12 h-1 bg-cyan-400"></div>
      <h1 class="font-title text-4xl font-bold text-white">后续计划</h1>
      <span class="ml-2 text-slate-400 text-lg font-light">从可用 Demo 走向成熟系统</span>
    </div>
    <div class="grid grid-cols-2 gap-5 flex-1">
      <div class="bg-slate-800/50 border border-cyan-500/30 rounded-xl p-6 flex flex-col">
        <h3 class="text-lg font-bold text-cyan-300 mb-2">数据扩展</h3>
        <p class="text-slate-400 text-sm leading-relaxed flex-1">纳入更多专科（罕见病、营养 / 康复科普），提升覆盖广度。</p>
      </div>
      <div class="bg-slate-800/50 border border-indigo-500/30 rounded-xl p-6 flex flex-col">
        <h3 class="text-lg font-bold text-indigo-300 mb-2">评估闭环</h3>
        <p class="text-slate-400 text-sm leading-relaxed flex-1">落地 Ragas 量化看板，持续监控四指标，指导迭代。</p>
      </div>
      <div class="bg-slate-800/50 border border-purple-500/30 rounded-xl p-6 flex flex-col">
        <h3 class="text-lg font-bold text-purple-300 mb-2">前端增强</h3>
        <p class="text-slate-400 text-sm leading-relaxed flex-1">来源引用可视化、检索过程透明化，提升可解释性。</p>
      </div>
      <div class="bg-slate-800/50 border border-cyan-500/30 rounded-xl p-6 flex flex-col">
        <h3 class="text-lg font-bold text-cyan-300 mb-2">可复现性</h3>
        <p class="text-slate-400 text-sm leading-relaxed flex-1">venv 运行时自包含脚本、依赖固化（websockets 等）。</p>
      </div>
    </div>
    <div class="mt-5 bg-slate-900/40 border border-cyan-500/30 rounded-lg px-6 py-4 text-center">
      <p class="text-slate-300 text-lg"><span class="text-cyan-300 font-semibold">目标：</span>从"可用 Demo"走向"可评测、可维护"的成熟系统。</p>
    </div>
  </div>
</div>
`);

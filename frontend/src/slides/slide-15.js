window.slideDataMap.set(15, `
<div class="w-[1440px] h-[810px] shadow-2xl relative overflow-hidden slide-bg">
  <div class="absolute inset-0 opacity-12" style="background-image: linear-gradient(rgba(34,211,238,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,0.1) 1px, transparent 1px); background-size: 44px 44px;"></div>
  <div class="absolute -top-16 left-1/3 w-80 h-80 bg-emerald-500 rounded-full blur-3xl opacity-10"></div>
  <div class="relative z-10 w-[1350px] h-[720px] mx-auto my-[45px] flex flex-col">
    <div class="mb-6 flex items-center gap-4">
      <div class="w-12 h-1 bg-cyan-400"></div>
      <h1 class="font-title text-4xl font-bold text-white">工程成果</h1>
    </div>
    <div class="grid grid-cols-2 gap-5 flex-1">
      <div class="bg-slate-800/50 border border-emerald-500/30 rounded-xl p-5 flex items-start gap-4">
        <span class="font-mono text-emerald-300 text-xl">✓</span>
        <div><p class="text-white font-semibold mb-1">可运行</p><p class="text-slate-400 text-sm">离线建库 + 在线 API（8005）+ Streamlit 前端（8501）端到端打通。</p></div>
      </div>
      <div class="bg-slate-800/50 border border-emerald-500/30 rounded-xl p-5 flex items-start gap-4">
        <span class="font-mono text-emerald-300 text-xl">✓</span>
        <div><p class="text-white font-semibold mb-1">稳定性</p><p class="text-slate-400 text-sm">修复 Windows uv venv 缺 VC++ 运行时崩溃，自包含 8 个 DLL。</p></div>
      </div>
      <div class="bg-slate-800/50 border border-emerald-500/30 rounded-xl p-5 flex items-start gap-4">
        <span class="font-mono text-emerald-300 text-xl">✓</span>
        <div><p class="text-white font-semibold mb-1">可用性</p><p class="text-slate-400 text-sm">修复 FAQ 误答：医疗问题跳过 FAQ + 关键词守卫。</p></div>
      </div>
      <div class="bg-slate-800/50 border border-emerald-500/30 rounded-xl p-5 flex items-start gap-4">
        <span class="font-mono text-emerald-300 text-xl">✓</span>
        <div><p class="text-white font-semibold mb-1">中间件</p><p class="text-slate-400 text-sm">Milvus / Redis（Docker，密码 1234）/ MySQL 已联通并验证。</p></div>
      </div>
    </div>
    <div class="mt-5 bg-slate-900/40 border border-cyan-500/30 rounded-lg px-6 py-4 flex items-center justify-center gap-6 flex-wrap">
      <span class="text-slate-500 text-sm font-mono">关键数字</span>
      <span class="text-cyan-300 text-lg font-semibold font-mono">2570 篇</span>
      <span class="text-cyan-300 text-lg font-semibold font-mono">20816 块</span>
      <span class="text-cyan-300 text-lg font-semibold font-mono">BGE-M3 多向量</span>
      <span class="text-cyan-300 text-lg font-semibold font-mono">六步流程</span>
    </div>
  </div>
</div>
`);

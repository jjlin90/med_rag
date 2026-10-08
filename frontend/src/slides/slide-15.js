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
        <div><p class="text-white font-semibold mb-1">运行入口</p><p class="text-slate-400 text-sm">离线建库、在线 API（8005）、CLI 与 Streamlit 界面（8501），共用核心问答流程。</p></div>
      </div>
      <div class="bg-slate-800/50 border border-emerald-500/30 rounded-xl p-5 flex items-start gap-4">
        <span class="font-mono text-emerald-300 text-xl">✓</span>
        <div><p class="text-white font-semibold mb-1">独立启动</p><p class="text-slate-400 text-sm">Windows 预加载 PyArrow，提供净化环境启动器；独立进程回归覆盖 API 帮助与延迟导入。</p></div>
      </div>
      <div class="bg-slate-800/50 border border-emerald-500/30 rounded-xl p-5 flex items-start gap-4">
        <span class="font-mono text-emerald-300 text-xl">✓</span>
        <div><p class="text-white font-semibold mb-1">FAQ 路由</p><p class="text-slate-400 text-sm">零匹配进入 RAG；带历史、来源过滤或显式策略的请求直接进入深通道。</p></div>
      </div>
      <div class="bg-slate-800/50 border border-emerald-500/30 rounded-xl p-5 flex items-start gap-4">
        <span class="font-mono text-emerald-300 text-xl">✓</span>
        <div><p class="text-white font-semibold mb-1">故障处理</p><p class="text-slate-400 text-sm">检索、意图分类与重排故障显式返回 L2；回归验证拒答及跳过缓存的行为。</p></div>
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

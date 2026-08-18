window.slideDataMap.set(13, `
<div class="w-[1440px] h-[810px] shadow-2xl relative overflow-hidden slide-bg">
  <div class="absolute inset-0 opacity-12" style="background-image: linear-gradient(rgba(34,211,238,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,0.1) 1px, transparent 1px); background-size: 44px 44px;"></div>
  <div class="absolute -top-16 right-1/4 w-80 h-80 bg-cyan-500 rounded-full blur-3xl opacity-12"></div>
  <div class="relative z-10 w-[1350px] h-[720px] mx-auto my-[45px] flex flex-col">
    <div class="mb-6 flex items-center gap-4">
      <div class="w-12 h-1 bg-cyan-400"></div>
      <h1 class="font-title text-4xl font-bold text-white">多轮对话</h1>
      <span class="ml-2 text-slate-400 text-lg font-light">从一问一答到连续问诊式</span>
    </div>
    <div class="grid grid-cols-3 gap-6 flex-1">
      <div class="bg-slate-800/50 border border-cyan-500/30 rounded-xl p-6 flex flex-col">
        <div class="w-12 h-12 rounded-lg bg-cyan-500/20 border border-cyan-400/50 flex items-center justify-center font-mono text-cyan-300 mb-4">ID</div>
        <h3 class="text-lg font-bold text-white mb-2">会话管理</h3>
        <p class="text-slate-400 text-sm leading-relaxed flex-1">每次对话分配会话 ID，写入网址，<span class="text-cyan-300">刷新不丢失</span>。</p>
      </div>
      <div class="bg-slate-800/50 border border-indigo-500/30 rounded-xl p-6 flex flex-col">
        <div class="w-12 h-12 rounded-lg bg-indigo-500/20 border border-indigo-400/50 flex items-center justify-center font-mono text-indigo-300 mb-4">DB</div>
        <h3 class="text-lg font-bold text-white mb-2">历史持久化</h3>
        <p class="text-slate-400 text-sm leading-relaxed flex-1">多轮上下文存 MySQL，支持连贯追问（先问"1 型糖尿病"再问"饮食注意"）。</p>
      </div>
      <div class="bg-slate-800/50 border border-purple-500/30 rounded-xl p-6 flex flex-col">
        <div class="w-12 h-12 rounded-lg bg-purple-500/20 border border-purple-400/50 flex items-center justify-center font-mono text-purple-300 mb-4">↺</div>
        <h3 class="text-lg font-bold text-white mb-2">上下文窗口</h3>
        <p class="text-slate-400 text-sm leading-relaxed flex-1">最近若干轮拼入 Prompt，兼顾连贯与成本。</p>
      </div>
      <div class="col-span-3 bg-slate-900/40 border border-slate-700 rounded-lg px-6 py-4">
        <p class="text-slate-300 text-base"><span class="text-cyan-300 font-semibold">价值：</span>从"一问一答"升级为"连续问诊式"交互，医疗咨询场景体验更自然。</p>
      </div>
    </div>
  </div>
</div>
`);

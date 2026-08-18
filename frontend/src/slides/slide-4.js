window.slideDataMap.set(4, `
<div class="w-[1440px] h-[810px] shadow-2xl relative overflow-hidden slide-bg">
  <div class="absolute inset-0 opacity-12" style="background-image: linear-gradient(rgba(129,140,248,0.12) 1px, transparent 1px), linear-gradient(90deg, rgba(129,140,248,0.12) 1px, transparent 1px); background-size: 44px 44px;"></div>
  <div class="absolute -top-20 -left-16 w-80 h-80 bg-rose-500 rounded-full blur-3xl opacity-10"></div>
  <div class="absolute -bottom-20 -right-16 w-80 h-80 bg-cyan-500 rounded-full blur-3xl opacity-10"></div>
  <div class="relative z-10 w-[1350px] h-[720px] mx-auto my-[45px] flex flex-col">
    <div class="mb-7 flex items-center gap-4">
      <div class="w-12 h-1 bg-cyan-400"></div>
      <h1 class="font-title text-4xl font-bold text-white">背景与目标</h1>
    </div>
    <div class="grid grid-cols-2 gap-7 flex-1">
      <div class="bg-slate-800/40 border border-rose-500/30 rounded-xl p-7">
        <h3 class="text-xl font-bold text-rose-300 mb-5 flex items-center gap-2"><span class="font-mono text-rose-400">!</span> 痛点：通用大模型医学问答</h3>
        <div class="space-y-4">
          <div class="bg-slate-900/40 rounded-lg p-4 border-l-4 border-rose-400">
            <p class="text-white text-lg font-semibold mb-1">幻觉风险</p>
            <p class="text-slate-400 text-base leading-relaxed">医学回答易凭空编造，引用来源不可靠。</p>
          </div>
          <div class="bg-slate-900/40 rounded-lg p-4 border-l-4 border-rose-400">
            <p class="text-white text-lg font-semibold mb-1">无权威出处</p>
            <p class="text-slate-400 text-base leading-relaxed">答案难以回溯到可信医学来源，无法查证。</p>
          </div>
          <div class="bg-slate-900/40 rounded-lg p-4 border-l-4 border-rose-400">
            <p class="text-white text-lg font-semibold mb-1">信任缺失</p>
            <p class="text-slate-400 text-base leading-relaxed">用户无法判断回答是否可信，不敢依此行动。</p>
          </div>
        </div>
      </div>
      <div class="bg-slate-800/40 border border-cyan-500/30 rounded-xl p-7">
        <h3 class="text-xl font-bold text-cyan-300 mb-5 flex items-center gap-2"><span class="font-mono text-cyan-400">&gt;</span> 目标 · 解法 · 价值</h3>
        <div class="space-y-4">
          <div class="bg-slate-900/40 rounded-lg p-4 border-l-4 border-cyan-400">
            <p class="text-white text-lg font-semibold mb-1">目标</p>
            <p class="text-slate-400 text-base leading-relaxed">构建"答案有据可依、可追溯"的医疗科普问答系统。</p>
          </div>
          <div class="bg-slate-900/40 rounded-lg p-4 border-l-4 border-cyan-400">
            <p class="text-white text-lg font-semibold mb-1">解法（RAG）</p>
            <p class="text-slate-400 text-base leading-relaxed">先召回权威知识片段，再由大模型组织成自然语言。</p>
          </div>
          <div class="bg-slate-900/40 rounded-lg p-4 border-l-4 border-cyan-400">
            <p class="text-white text-lg font-semibold mb-1">价值</p>
            <p class="text-slate-400 text-base leading-relaxed">高频缓存直答省成本；医疗走检索保权威；多轮连贯。</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
`);

window.slideDataMap.set(8, `
<div class="w-[1440px] h-[810px] shadow-2xl relative overflow-hidden slide-bg">
  <div class="absolute inset-0 opacity-12" style="background-image: linear-gradient(rgba(34,211,238,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,0.1) 1px, transparent 1px); background-size: 44px 44px;"></div>
  <div class="absolute -bottom-20 right-1/4 w-80 h-80 bg-cyan-500 rounded-full blur-3xl opacity-12"></div>
  <div class="relative z-10 w-[1350px] h-[720px] mx-auto my-[45px] flex flex-col">
    <div class="mb-7 flex items-center gap-4">
      <div class="w-12 h-1 bg-cyan-400"></div>
      <h1 class="font-title text-4xl font-bold text-white">离线建库</h1>
      <span class="ml-2 text-slate-400 text-lg font-light">知识从哪来、怎么存</span>
    </div>
    <div class="grid grid-cols-4 gap-5 flex-1">
      <div class="relative bg-slate-800/50 border border-cyan-500/30 rounded-xl p-6 flex flex-col">
        <div class="w-12 h-12 rounded-full bg-cyan-500/20 border border-cyan-400/50 flex items-center justify-center font-mono text-xl font-bold text-cyan-300 mb-4">1</div>
        <h3 class="text-lg font-bold text-white mb-2">清洗</h3>
        <p class="text-slate-400 text-sm leading-relaxed flex-1">去除导航、广告、版权声明，得到约 <span class="text-cyan-300 font-semibold">2570 篇</span> 干净 Markdown。</p>
      </div>
      <div class="relative bg-slate-800/50 border border-indigo-500/30 rounded-xl p-6 flex flex-col">
        <div class="w-12 h-12 rounded-full bg-indigo-500/20 border border-indigo-400/50 flex items-center justify-center font-mono text-xl font-bold text-indigo-300 mb-4">2</div>
        <h3 class="text-lg font-bold text-white mb-2">父子分块</h3>
        <p class="text-slate-400 text-sm leading-relaxed flex-1">子块 <span class="text-indigo-300 font-semibold">400</span> 字符精准召回，父块 <span class="text-indigo-300 font-semibold">2000</span> 字符保上下文，共约 <span class="text-indigo-300 font-semibold">20816</span> 块。</p>
      </div>
      <div class="relative bg-slate-800/50 border border-purple-500/30 rounded-xl p-6 flex flex-col">
        <div class="w-12 h-12 rounded-full bg-purple-500/20 border border-purple-400/50 flex items-center justify-center font-mono text-xl font-bold text-purple-300 mb-4">3</div>
        <h3 class="text-lg font-bold text-white mb-2">向量化</h3>
        <p class="text-slate-400 text-sm leading-relaxed flex-1">BGE-M3 生成 <span class="text-purple-300 font-semibold">稠密 + 稀疏（关闭ColBERT）</span>，多粒度表达能力更强。</p>
      </div>
      <div class="relative bg-slate-800/50 border border-cyan-500/30 rounded-xl p-6 flex flex-col">
        <div class="w-12 h-12 rounded-full bg-cyan-500/20 border border-cyan-400/50 flex items-center justify-center font-mono text-xl font-bold text-cyan-300 mb-4">4</div>
        <h3 class="text-lg font-bold text-white mb-2">入库</h3>
        <p class="text-slate-400 text-sm leading-relaxed flex-1">写入 Milvus 集合 <span class="text-cyan-300 font-mono text-xs">med_msd_consumer_chunk</span>，建立混合索引。</p>
      </div>
    </div>
    <div class="mt-5 bg-slate-900/40 border border-slate-700 rounded-lg px-6 py-4">
      <p class="text-slate-300 text-base"><span class="text-cyan-300 font-semibold">成功标准：</span>检索延迟可控、召回率满足在线问答需求，是 RAG 答案质量的根基。</p>
    </div>
  </div>
</div>
`);

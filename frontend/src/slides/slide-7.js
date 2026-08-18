window.slideDataMap.set(7, `
<div class="w-[1440px] h-[810px] shadow-2xl relative overflow-hidden slide-bg">
  <div class="absolute inset-0 opacity-12" style="background-image: linear-gradient(rgba(129,140,248,0.12) 1px, transparent 1px), linear-gradient(90deg, rgba(129,140,248,0.12) 1px, transparent 1px); background-size: 44px 44px;"></div>
  <div class="absolute -top-16 left-1/3 w-80 h-80 bg-indigo-500 rounded-full blur-3xl opacity-12"></div>
  <div class="relative z-10 w-[1350px] h-[720px] mx-auto my-[45px] flex flex-col">
    <div class="mb-6 flex items-center gap-4">
      <div class="w-12 h-1 bg-cyan-400"></div>
      <h1 class="font-title text-4xl font-bold text-white">系统总体架构</h1>
    </div>
    <div class="grid grid-cols-3 gap-5 flex-1">
      <div class="bg-slate-800/50 border border-cyan-500/30 rounded-xl p-6 flex flex-col">
        <div class="flex items-center gap-3 mb-4">
          <div class="w-10 h-10 rounded-lg bg-cyan-500/20 border border-cyan-400/50 flex items-center justify-center font-mono text-cyan-300 text-sm">L1</div>
          <h3 class="text-xl font-bold text-cyan-300">知识层</h3>
        </div>
        <div class="space-y-2 text-slate-300 text-base">
          <p class="flex items-center gap-2"><span class="text-cyan-400">▸</span>默沙东手册</p>
          <p class="flex items-center gap-2"><span class="text-cyan-400">▸</span>清洗去噪</p>
          <p class="flex items-center gap-2"><span class="text-cyan-400">▸</span>父子分块</p>
          <p class="flex items-center gap-2"><span class="text-cyan-400">▸</span>BGE-M3 嵌入</p>
          <p class="flex items-center gap-2"><span class="text-cyan-400">▸</span>Milvus 向量库</p>
        </div>
      </div>
      <div class="bg-slate-800/50 border border-indigo-500/30 rounded-xl p-6 flex flex-col">
        <div class="flex items-center gap-3 mb-4">
          <div class="w-10 h-10 rounded-lg bg-indigo-500/20 border border-indigo-400/50 flex items-center justify-center font-mono text-indigo-300 text-sm">L2</div>
          <h3 class="text-xl font-bold text-indigo-300">服务层</h3>
        </div>
        <div class="space-y-2 text-slate-300 text-base">
          <p class="flex items-center gap-2"><span class="text-indigo-400">▸</span>FastAPI 服务</p>
          <p class="flex items-center gap-2"><span class="text-indigo-400">▸</span>FAQ 缓存</p>
          <p class="flex items-center gap-2"><span class="text-indigo-400">▸</span>意图分类</p>
          <p class="flex items-center gap-2"><span class="text-indigo-400">▸</span>检索策略</p>
          <p class="flex items-center gap-2"><span class="text-indigo-400">▸</span>重排 / 生成</p>
        </div>
      </div>
      <div class="bg-slate-800/50 border border-purple-500/30 rounded-xl p-6 flex flex-col">
        <div class="flex items-center gap-3 mb-4">
          <div class="w-10 h-10 rounded-lg bg-purple-500/20 border border-purple-400/50 flex items-center justify-center font-mono text-purple-300 text-sm">L3</div>
          <h3 class="text-xl font-bold text-purple-300">交互层</h3>
        </div>
        <div class="space-y-2 text-slate-300 text-base">
          <p class="flex items-center gap-2"><span class="text-purple-400">▸</span>Streamlit 前端（8501）</p>
          <p class="flex items-center gap-2"><span class="text-purple-400">▸</span>多轮会话</p>
          <p class="flex items-center gap-2"><span class="text-purple-400">▸</span>会话持久化</p>
          <p class="flex items-center gap-2"><span class="text-purple-400">▸</span>来源可追溯</p>
        </div>
      </div>
    </div>
    <div class="mt-5 bg-slate-900/40 border border-slate-700 rounded-lg px-6 py-4 flex items-center justify-center gap-3 flex-wrap">
      <span class="text-slate-500 text-sm font-mono">技术栈</span>
      <span class="text-cyan-300 text-sm font-mono">BGE-M3</span><span class="text-slate-600">·</span>
      <span class="text-cyan-300 text-sm font-mono">Milvus 混合检索</span><span class="text-slate-600">·</span>
      <span class="text-cyan-300 text-sm font-mono">BGE-reranker</span><span class="text-slate-600">·</span>
      <span class="text-cyan-300 text-sm font-mono">bert-base-chinese</span><span class="text-slate-600">·</span>
      <span class="text-cyan-300 text-sm font-mono">Redis</span><span class="text-slate-600">·</span>
      <span class="text-cyan-300 text-sm font-mono">MySQL</span>
    </div>
  </div>
</div>
`);

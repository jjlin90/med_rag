window.slideDataMap.set(5, `
<div class="w-[1440px] h-[810px] shadow-2xl relative overflow-hidden slide-bg">
  <div class="absolute inset-0 opacity-12" style="background-image: linear-gradient(rgba(34,211,238,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,0.1) 1px, transparent 1px); background-size: 44px 44px;"></div>
  <div class="absolute -top-16 right-16 w-80 h-80 bg-cyan-500 rounded-full blur-3xl opacity-12"></div>
  <div class="relative z-10 w-[1350px] h-[720px] mx-auto my-[45px] flex flex-col">
    <div class="mb-6 flex items-center gap-4">
      <div class="w-12 h-1 bg-cyan-400"></div>
      <h1 class="font-title text-4xl font-bold text-white">数据来源</h1>
      <span class="ml-2 text-cyan-300 text-lg font-light">— 默沙东诊疗手册（大众版）</span>
    </div>
    <div class="grid grid-cols-3 gap-6 flex-1">
      <div class="col-span-1 bg-indigo-500/10 border border-indigo-400/40 rounded-xl p-7 flex flex-col justify-center">
        <div class="w-12 h-12 rounded-lg bg-indigo-500/30 flex items-center justify-center font-mono text-2xl text-indigo-200 mb-4">MSD</div>
        <h3 class="text-xl font-bold text-white mb-3">医学科普 · 大众版</h3>
        <p class="text-slate-300 text-base leading-relaxed">本地语料按默沙东手册大众版组织，用于医疗科普技术学习与演示；来源授权由数据提供方材料确认。</p>
      </div>
      <div class="col-span-2 grid grid-cols-2 gap-5">
        <div class="bg-slate-800/50 border border-cyan-500/30 rounded-xl p-6 flex flex-col justify-center">
          <p class="font-mono text-4xl font-bold text-cyan-300">2570<span class="text-xl text-slate-400 ml-1">篇</span></p>
          <p class="text-slate-400 text-base mt-1">清洗后 Markdown 文档</p>
        </div>
        <div class="bg-slate-800/50 border border-cyan-500/30 rounded-xl p-6 flex flex-col justify-center">
          <p class="font-mono text-4xl font-bold text-cyan-300">20816<span class="text-xl text-slate-400 ml-1">块</span></p>
          <p class="text-slate-400 text-base mt-1">父子分块文本单元</p>
        </div>
        <div class="bg-slate-800/50 border border-purple-500/30 rounded-xl p-6 flex flex-col justify-center">
          <p class="font-mono text-4xl font-bold text-purple-300">522.90<span class="text-xl text-slate-400 ml-1">字符</span></p>
          <p class="text-slate-400 text-base mt-1">全部父子记录平均长度</p>
        </div>
        <div class="bg-slate-800/50 border border-purple-500/30 rounded-xl p-6 flex flex-col justify-center">
          <p class="font-mono text-4xl font-bold text-purple-300">2000<span class="text-xl text-slate-400 ml-1">字符</span></p>
          <p class="text-slate-400 text-base mt-1">全部父子记录最大长度</p>
        </div>
      </div>
    </div>
    <div class="mt-5 bg-slate-900/40 border border-slate-700 rounded-lg px-6 py-4 flex items-center gap-6 flex-wrap">
      <span class="text-slate-500 text-sm font-mono">覆盖</span>
      <span class="text-slate-300 text-sm">内分泌</span><span class="text-slate-600">·</span>
      <span class="text-slate-300 text-sm">心脑血管</span><span class="text-slate-600">·</span>
      <span class="text-slate-300 text-sm">呼吸</span><span class="text-slate-600">·</span>
      <span class="text-slate-300 text-sm">消化</span><span class="text-slate-600">·</span>
      <span class="text-slate-300 text-sm">骨科</span><span class="text-slate-600">·</span>
      <span class="text-slate-300 text-sm">皮肤</span><span class="text-slate-600">·</span>
      <span class="text-slate-300 text-sm">儿科</span><span class="text-slate-600">·</span>
      <span class="text-slate-300 text-sm">妇科</span>
      <span class="ml-auto text-emerald-300 text-sm">用途：技术学习与演示；数据使用依据来源材料确认</span>
    </div>
  </div>
</div>
`);

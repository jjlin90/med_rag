# PPT 演示前端

本目录不是医疗问答 Web 客户端；它是一个用 Vite 构建的静态 PPT 演示项目。医疗问答界面位于 `web_demo/`。

## 代码核验后的技术栈

- Vite `^7.2.2`
- Tailwind CSS `^3.4.1`
- 原生 JavaScript ES Modules
- PostCSS / Autoprefixer

依赖和版本以 `package.json`、锁文件为准。

## 实际目录

- `index.html`、`404.html`：页面入口
- `src/main.js`：主入口
- `src/js/ppt-controller.js`：翻页控制
- `src/js/route-handler.js`：路由处理
- `src/slides/slide-1.js` ～ `slide-17.js`：17 页内容
- `src/styles/`：样式
- `scripts/build-slides.js`：幻灯片构建脚本

共17页，支持按钮、方向键、Home/End和触摸翻页；页码与URL同步。

## 命令

```bash
cd frontend
npm install
npm run dev
npm run build
npm run preview
```

`npm run build` 写入 `frontend/dist/`，避免与项目文档、PPT 等共用 artifacts。`npm run build:slides` 直接运行 Node 脚本，已去除依赖 Unix rm 命令的前置步骤；输出 dist-slides。旧 build:export-pptx 引用了仓库不存在的 .codebuddy 导出器，已删除该无效命令；原有 PPT 文件保留。

2026-10-08 已验证17页渲染、桌面1600×1000、手机390×844、翻页和控制台。默认页面是静态汇报，问答界面位于 `web_demo/`。

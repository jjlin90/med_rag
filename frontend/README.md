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

旧文档写成 `src/data/slide-N.js`、10 页，均与当前目录不符，已删除。

## 命令

```bash
cd frontend
npm install
npm run dev
npm run build
npm run preview
```

`npm run build` 写入 `frontend/dist/`，避免与项目文档、PPT 等共用 artifacts。`npm run build:slides` 直接运行 Node 脚本，已去除依赖 Unix rm 命令的前置步骤；输出 dist-slides。旧 build:export-pptx 引用了仓库不存在的 .codebuddy 导出器，已删除该无效命令；原有 PPT 文件保留。

本文只陈述仓库代码和配置可证明的内容；浏览器版本范围、线上部署状态、许可证和性能均未由本仓库验证。

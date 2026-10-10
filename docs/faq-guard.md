# FAQ 直答一致性修复与验证（2026-10-10）

## 确认的问题

FAQ 的 BM25 + Top-5 softmax 只能衡量词项相关性与候选相对优势。问题只改了人群、方向、病程等关键字符时，错误候选仍可接近 1.0。例如“HDL 胆固醇降低”会命中“HDL 胆固醇升高”，“成人健康”可能命中儿童主题。提高 0.85 阈值不能可靠判断是否同义。

本地 2,416 条 FAQ 来自 `data/msd_faq_clean.csv`，主要是医学主题标题及答案。本次按固定替换表构造 165 条字符改写探针，旧接受策略有 82 条命中保留原类别词的 FAQ。探针包括人为改写，不能当作独立临床标注基准。

## 当前接受策略

无历史、来源过滤、显式策略限制的请求才尝试 FAQ。最高 BM25 原始分必须大于零，Top-5 softmax 必须达到 0.85，且输入与候选标准问题仅允许首尾空白不同。代码同时核对数据库当前 `question`，并要求答案是非空字符串。任何条件不满足都进入深通道。

不删除数字、标点、正负号、内部空格，不转换大小写，不把词项覆盖率或相似度当成医学等价性证明。此策略会让同义改写、拼写变体和追加条件进入深通道；在有独立等价性标注与可靠验收前，不开放模糊 FAQ 直答。深通道随后仍会分类，`general` 通用生成、`medical` 执行医学 RAG；进入深通道不等于已经完成检索或获得正确答案。

FAQ 缓存必须保存 `question` 并验证其与当前输入一致，数据库不可用时也不能绕过此条件。两层键统一升级为 `faq:v3:<MD5>` 与 `query:v3:<SHA-256>`，旧 v2 条目不再读取，按原 TTL 自然到期，无需清空共享 Redis。FAQ 数据更新后的自动版本失效策略仍未实现，现有缓存依赖 TTL；此次 v3 迁移仅隔离旧接受策略的结果。

## 验证记录

| 验证对象 | 修复前 | 修复后 |
|---|---:|---:|
| 165 条改写探针中的相反类别误接受 | 82 | 0 |
| 改写探针中问题不一致却直答 | — | 0 |
| 2,416 条原始标准问题的直答数 | 2,345 | 2,345 |
| 当前全量工程测试（含8项FAQ回归与14项质量试验回归） | 原 50 项 | 72 项通过 |

改写探针中 20 条仍直答，因为修改后的文本本身就是另一条标准问题；未直答的 145 条进入深通道。原始问题中 71 条仍受 softmax 候选门槛限制，不能声称标准问题全部命中。

新增回归覆盖反义、人群、病程、否定、数值与单位差异、未见词追加、数据库记录偏离索引、空答案、离线缓存证据、两层旧缓存隔离及 API 深通道分流。回归使用真实 BM25，数据库、缓存、深通道生成用隔离替身。[公开摘要](faq_guard_verification_20261010.json)不包含答案正文或密钥；详细本地报告与旧桌面文件备份保存在已忽略的 `artifacts/`。

```powershell
python -m unittest discover -s tests -v
# 完整本地报告，包含每条改写探针
python scripts/verify_faq_guard.py --output artifacts/faq_guard_verification.json
# 发布摘要：实测回归数、接受策略及缓存命名空间由脚本一起写出
python scripts/verify_faq_guard.py --with-tests --summary-only --output docs/faq_guard_verification_20261010.json
python scripts/audit_static.py
```

`acceptance_policy` 由验证脚本的当前策略说明生成，`cache_namespaces` 从实际缓存键函数取得；`--with-tests` 运行完整回归，按实际成功数写入 `automated_tests_passed`，测试失败则终止发布。`--summary-only` 仅省略逐条 `cases`。公开摘要与本地详细报告使用同一生成入口，时间戳随本次运行更新。

整库验证使用真实本地 CSV 与 BM25，CSV 游标替代 MySQL，关闭缓存；未调用真实 Redis/MySQL/Milvus 或供应商 LLM，不是完整线上验收，也不证明知识库答案的临床正确性。`data/` 未随仓库发布，复跑整库验证需自行准备有权使用的 FAQ CSV。

## 真实数据库与缓存验证（2026-10-10）

另以生产FAQSearch和BM25连接隔离的真实MySQL 8实例，将同一份2416条CSV写入验收库；Redis使用本机现有服务与独立测试键前缀。标准问题直答2345条，165条改写探针的相反类别误接受和问题不一致直答均为0，与上述隔离替身验证结果一致。

缓存未命中时实际执行MySQL SELECT取答并回填Redis；再次命中时SQL查询数不增加。缓存标准问题不一致或答案为空时重新查库，v2条目不影响v3查询，实际缓存具有有效TTL。测试后仅清理验收实例和测试键，保留现有数据库与缓存数据。此次覆盖FAQ组件的真实数据库与缓存路径，未调用API或供应商LLM；对应日志与摘要见`artifacts/quality_20261010/live_faq_check.log`和`live_faq_verification.json`。

联调同时发现MySQL 8默认认证需要PyMySQL的RSA依赖，现已在pyproject.toml与requirements.txt中声明`pymysql[rsa]==1.1.1`，补齐认证所需cryptography。

仓库提供[真实FAQ服务验证脚本](../scripts/verify_faq_services.py)，安装项目依赖、启动Docker Desktop与项目配置中的Redis，并准备有权使用的FAQ CSV后，从源码根目录执行：

```powershell
python scripts/verify_faq_services.py --dataset data/msd_faq_clean.csv --output artifacts/faq_services_verification.json
```

首次运行需要本机已有MySQL镜像或能够拉取`mysql:8.0`；可用`--mysql-image`指定兼容镜像。脚本自动创建随机密码、本机随机端口及内存数据目录的临时MySQL实例，Redis仅操作独立随机前缀的测试键。摘要只在验证通过、临时实例及测试键清理完成后生成。验证使用正式Config，因此本机CUDA也需满足项目启动检查。CSV与实际服务需自行准备，脚本随源码包与wheel分发，不依赖私有artifacts脚本。

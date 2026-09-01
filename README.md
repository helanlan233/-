# 基于 RAG 的科研文献智能分析助手

这是一个适合学习和面试展示的轻量 RAG（Retrieval-Augmented Generation，检索增强生成）项目。它支持一次上传多篇 PDF 论文，按论文分别检索证据，让模型严格依据原文回答，并提供文献概览、证据核验和结构化对比。

## 功能

- 多 PDF 上传与逐页文本提取
- PDF 中文空格、逐字母英文、全角字符和错误断行清洗
- 使用 SHA-256 检测并跳过重复 PDF
- 对可信论文标题做标准化完全匹配，跳过不同下载版本的疑似重复论文
- 保留 PDF 文件名、页码和原文元数据
- 标记 `main` / `references`，默认排除参考文献污染
- 页内重叠切块，避免来源页码含糊
- 硅基流动免费 Embedding 向量化
- FAISS 本地向量索引与按论文分别 Top-K 检索
- 面向研究方法、变量、内生性和机制问题的轻量查询扩展
- 多文献逐篇回答，并在综合比较中只归纳异同，不重复逐篇答案
- 主回答保持简洁；“回答依据”与“检索详情”分开折叠展示
- 每篇论文一次结构化提取，生成文献概览并缓存到会话状态
- 区分实证研究、理论研究、综述研究和其他，按类型抽取不同科研字段
- 严格区分“不适用”和“未找到足够信息”，避免把理论概念误判成 X/Y
- 概览与十一字段文献对比表复用同一份结果，避免重复调用模型
- 六个科研常用问题可一键执行

## 技术栈

- Python 3.9+（Windows 推荐 3.10-3.12；本项目已用 3.12 验证）
- Streamlit
- PyPDF（`pypdf` 包）
- OpenAI Python SDK（调用硅基流动的 OpenAI 兼容接口）
- 硅基流动 Chat Completions API + Embeddings API
- FAISS CPU
- NumPy
- python-dotenv

## RAG 流程

```text
上传多篇 PDF
    ↓
PDF 内容 hash 去重
    ↓
PyPDF 逐页提取文字（文件名 + 页码 + 正文）
    ↓
清洗异常空格与断行，标记 main / references
    ↓
页内重叠切分 Chunk（继续保留文件名 + 页码）
    ↓
仅将 main Chunk 交给硅基流动 Embedding 转成向量
    ↓
一次性识别论文类型、抽取结构化信息和可信标题
    ↓
标题标准化去重，只把保留论文合并进本地 FAISS 索引
    ↓
按问题补充少量科研检索词 → 问题向量化
    ↓
每篇论文分别检索相同数量的正文片段
    ↓
把检索结果作为唯一 Context 交给大模型
    ↓
展示简洁回答 + 引用页码
    ↓
按需展开回答依据或检索调试详情
```

FAISS 使用归一化向量和内积索引 `IndexFlatIP`，因此检索分数等价于余弦相似度。为了保证跨论文问题不会被一篇论文占满 Top-K，程序会取回全局排序结果，再按 PDF 内容 hash 分组，每篇保留相同数量的片段。索引只保存在当前 Streamlit 会话的内存中，刷新页面后可能需要重新处理 PDF。

## 安装方法

推荐在项目目录中创建虚拟环境。

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果 PowerShell 阻止激活脚本，也可以不激活，直接使用：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 配置硅基流动 API Key

1. 打开[硅基流动控制台](https://cloud.siliconflow.cn/)，注册并登录。
2. 在“API 密钥”页面新建一个 Key。
3. 不要把 Key 发到聊天或截图中，只粘贴到本机的 `.env` 文件。

如果项目中还没有 `.env`，先复制示例配置：

```powershell
Copy-Item .env.example .env
```

打开 `.env`，在等号后填写自己的硅基流动 Key：

```dotenv
SILICONFLOW_API_KEY=你的硅基流动Key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_CHAT_MODEL=Qwen/Qwen2.5-32B-Instruct
SILICONFLOW_EMBEDDING_MODEL=BAAI/bge-m3
```

不要把 `.env` 提交到 Git。项目已通过 `.gitignore` 忽略它。

当前默认选择付费聊天模型 `Qwen/Qwen2.5-32B-Instruct` 和免费向量模型 `BAAI/bge-m3`。32B 模型是非推理指令模型，中文信息抽取和结构化输出比免费 7B 模型更稳定，也不会因为长篇思考内容挤占 RAG 回答正文。模型价格和平台规则可能调整，请以硅基流动官网为准。如果模型名称发生变化，只需修改 `.env`，无需改 Python 代码。

## 运行方法

已激活虚拟环境时：

```powershell
streamlit run app.py
```

未激活虚拟环境时：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器通常会自动打开 `http://localhost:8501`。上传 PDF 后先点击“解析文献”，然后即可使用快捷问题、自由提问或生成文献对比表。

解析新论文时，每篇论文会调用一次聊天模型，先判断“实证研究 / 理论研究 / 综述研究 / 其他”，再按类型生成结构化档案并保存在 `paper_profiles` 中。文献概览、问答类型提示和对比表都复用该结果；只有点击“刷新文献信息（重新调用模型）”才会重新抽取。旧缓存缺少 `paper_type` 时会以“其他”兼容显示并提示刷新。

## 项目结构

```text
.
├── app.py            # Streamlit 页面、交互和结果展示
├── rag.py            # PDF 解析、切块、Embedding、FAISS、问答与表格抽取
├── requirements.txt  # Python 依赖
├── .env.example      # 环境变量示例，不包含真实 Key
├── .gitignore        # 防止提交 .env、虚拟环境等本地文件
└── README.md         # 项目说明
```

## 示例问题

- 总结这些文献的研究问题和研究对象
- 总结这些文献的核心解释变量和被解释变量，并比较变量定义上的差异
- 总结这些文献分别使用了哪些研究方法，并比较识别策略差异
- 这些文献分别如何处理内生性问题？
- 这些文献分别进行了哪些机制分析？
- 总结这些文献的核心结论，并比较研究结论的异同

## Windows 下 FAISS 安装失败怎么办

默认依赖是 `faiss-cpu`。如果当前 Python/Windows 组合没有可用的安装包，优先尝试：

1. 使用 64 位 Python 3.10 或 3.11 新建虚拟环境；
2. 升级 pip 后重新执行 `python -m pip install faiss-cpu`；
3. 仍失败时，用 Chroma 替换 FAISS。

Chroma 替换思路：

```powershell
python -m pip uninstall faiss-cpu
python -m pip install chromadb
```

然后把 `rag.py` 中的 `faiss.IndexFlatIP` 存储与 `.search()` 检索改成 Chroma collection 的 `add()` 与 `query()`。PDF 解析、切块、Embedding、问答提示词和 Streamlit 页面都可以保留。为了让最小版本更容易理解，本项目默认不同时维护两套向量库实现。

## 已知限制

- PyPDF 只提取 PDF 内已有的文字层；扫描版论文需要先 OCR。
- 双栏论文、复杂公式和表格的文本顺序可能不完美。
- 参考文献识别采用独立标题匹配，是简单启发式规则，极少数排版特殊的论文可能识别不到。
- Embedding 和回答会调用硅基流动 API；免费模型有速率限制，平台规则也可能变化。
- 当前索引仅保存在内存中，没有数据库和持久化设计。

## 官方接口参考

- [硅基流动快速上手](https://docs.siliconflow.cn/cn/userguide/quickstart)
- [硅基流动 Chat Completions](https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions)
- [硅基流动 Embeddings](https://docs.siliconflow.cn/cn/api-reference/embeddings/create-embeddings)
- [硅基流动模型价格](https://siliconflow.cn/pricing)

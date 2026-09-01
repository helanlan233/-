"""Streamlit 页面：上传 PDF、科研问答、文献概览和结构化比较。"""

import os
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv

from rag import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    answer_question,
    build_comparison_table,
    build_rag_store,
    chunk_pages,
    clean_display_title,
    create_siliconflow_client,
    deduplicate_pdf_files,
    ensure_profile_defaults,
    extract_paper_profiles,
    extract_pdf_pages,
    find_duplicate_paper_titles,
    get_answer_evidence,
    merge_rag_stores,
    remove_documents_from_rag_store,
    retrieve_chunks,
    strip_source_markers,
)


load_dotenv(override=True)
st.set_page_config(page_title="科研文献智能分析助手", page_icon="📚", layout="wide")

INDEX_SCHEMA_VERSION = 3
if st.session_state.get("index_schema_version") != INDEX_SCHEMA_VERSION:
    for state_key in (
        "rag_store", "file_names", "processed_hashes", "paper_profiles",
        "comparison_rows", "last_answer", "last_sources", "last_evidence",
        "last_question", "profiles_need_refresh", "duplicate_title_notices",
    ):
        st.session_state.pop(state_key, None)
    st.session_state["index_schema_version"] = INDEX_SCHEMA_VERSION

# 旧缓存不报错：先补默认字段并提示刷新，索引本身可以继续使用。
cached_profiles = st.session_state.get("paper_profiles", {})
if cached_profiles:
    if any("paper_type" not in profile for profile in cached_profiles.values()):
        st.session_state["profiles_need_refresh"] = True
    st.session_state["paper_profiles"] = {
        document_id: ensure_profile_defaults(profile)
        for document_id, profile in cached_profiles.items()
    }

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 4rem; max-width: 1280px;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    .app-subtitle {color: #5b6472; font-size: 1.05rem; margin-top: -0.7rem;}
    [data-testid="stMetric"] {
        background: #f7f8fa; border: 1px solid #e5e7eb; border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e3e7ec; border-radius: 12px;
    }
    div.stButton > button {border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

QUICK_QUESTIONS = (
    "总结这些文献的研究问题和研究对象",
    "总结这些文献的核心解释变量和被解释变量",
    "总结这些文献分别使用了哪些研究方法",
    "这些文献如何处理内生性问题？",
    "总结这些文献的机制分析",
    "总结这些文献的核心结论并比较差异",
)


def _document_key(chunk):
    return chunk.get("file_hash") or chunk["file_name"]


def _queue_question(question):
    """快捷按钮的回调：填入问题，并在本次刷新后自动执行。"""
    st.session_state["question_input"] = question
    st.session_state["run_question"] = True


def _profile_title(document_id, file_name):
    profile = st.session_state.get("paper_profiles", {}).get(document_id, {})
    return profile.get("title") or clean_display_title(file_name)


def render_paper_card(profile):
    """根据论文类型自适应展示档案，不把理论概念强行显示成 X/Y。"""
    profile = ensure_profile_defaults(profile)
    paper_type = profile["paper_type"]
    with st.container(border=True):
        st.markdown(f'### 《{profile["title"]}》')
        st.caption(f"论文类型：{paper_type}")

        if paper_type == "实证研究":
            left, right = st.columns([1, 1])
            with left:
                st.markdown(f'**研究对象**  \n{profile["research_object"]}')
                st.markdown(f'**核心解释变量**  \n{profile["x"]}')
                st.markdown(f'**被解释变量**  \n{profile["y"]}')
            with right:
                st.markdown(f'**数据来源**  \n{profile["data_source"]}')
                st.markdown(f'**主要研究方法**  \n{profile["baseline_method"]}')
        elif paper_type == "理论研究":
            st.markdown(f'**研究问题**  \n{profile["research_question"]}')
            st.markdown(f'**理论视角**  \n{profile["theoretical_perspective"]}')
            st.markdown(f'**核心概念**  \n{profile["core_concepts"]}')
            st.markdown(f'**主要理论机制**  \n{profile["theoretical_mechanism"]}')
        elif paper_type == "综述研究":
            st.markdown(f'**综述主题**  \n{profile["review_topic"]}')
            st.markdown(f'**研究范围**  \n{profile["research_scope"]}')
            st.markdown(f'**主要研究脉络**  \n{profile["main_research_streams"]}')
            st.markdown(f'**研究空白**  \n{profile["research_gaps"]}')
        else:
            st.markdown(f'**研究问题**  \n{profile["research_question"]}')
            st.markdown(f'**研究对象**  \n{profile["research_object"]}')
            st.markdown(f'**主要研究方法**  \n{profile["baseline_method"]}')

        with st.expander("查看详细信息", expanded=False):
            st.markdown(f'**类型判断依据**  \n{profile["paper_type_reason"]}')
            if paper_type == "实证研究":
                st.markdown(f'**研究问题**  \n{profile["research_question"]}')
                st.markdown(
                    f'**内生性 / 识别策略**  \n'
                    f'{profile["identification_strategy"]}；{profile["endogeneity"]}'
                )
                st.markdown(f'**机制分析**  \n{profile["mechanism"]}')
            elif paper_type == "理论研究":
                st.markdown(f'**主要观点**  \n{profile["main_arguments"]}')
            elif paper_type == "综述研究":
                st.markdown(f'**文献范围**  \n{profile["literature_scope"]}')
                st.markdown(f'**主要争议**  \n{profile["main_debates"]}')
                st.markdown(f'**综述方法**  \n{profile["baseline_method"]}')
            st.markdown(f'**核心结论**  \n{profile["conclusion"]}')


def render_answer_evidence(evidence):
    """只展示答案真正引用的少量证据，方便科研核验。"""
    with st.expander("查看回答依据", expanded=False):
        if not evidence:
            st.caption("本次回答没有可展示的引用片段。")
            return
        for number, source in enumerate(evidence, start=1):
            document_id = _document_key(source)
            title = _profile_title(document_id, source["file_name"])
            st.markdown(f'**{number}. 《{title}》 · 第 {source["page_number"]} 页**')
            st.text_area(
                "对应原文", value=source["text"], height=130, disabled=True,
                key=f'evidence_{document_id}_{source.get("chunk_id", number)}',
                label_visibility="collapsed",
            )


def render_retrieval_debug(sources):
    """按论文折叠显示检索细节；所有内部技术信息只出现在这里。"""
    grouped = defaultdict(list)
    for source in sources:
        grouped[_document_key(source)].append(source)

    with st.expander("查看检索详情", expanded=False):
        st.caption("用于检查每篇论文实际进入回答上下文的片段。")
        for document_id, paper_sources in grouped.items():
            title = _profile_title(document_id, paper_sources[0]["file_name"])
            show_paper = st.toggle(
                f'《{title}》 · 检索到 {len(paper_sources)} 个片段',
                key=f"debug_toggle_{document_id}",
            )
            if not show_paper:
                continue
            for number, source in enumerate(paper_sources, start=1):
                st.markdown(f"**Chunk {number}**")
                st.caption(
                    f'页码：{source["page_number"]} ｜ '
                    f'section：{source.get("section", "main")} ｜ '
                    f'相似度：{source["score"]:.3f} ｜ '
                    f'文档ID：{document_id[:8]} ｜ '
                    f'chunk index：{source.get("chunk_id", number)}'
                )
                st.text_area(
                    "原文", value=source["text"], height=140, disabled=True,
                    key=f'debug_text_{document_id}_{source.get("chunk_id", number)}',
                    label_visibility="collapsed",
                )


chat_model = os.getenv("SILICONFLOW_CHAT_MODEL", DEFAULT_CHAT_MODEL)
embedding_model = os.getenv("SILICONFLOW_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
api_configured = bool(os.getenv("SILICONFLOW_API_KEY", "").strip())

with st.sidebar:
    st.header("项目状态")
    st.markdown("🟢 API 状态：已连接" if api_configured else "🔴 API 状态：未配置")
    st.caption("API：硅基流动")
    st.caption(f"回答模型：{chat_model.split('/')[-1]}")
    st.caption(f"向量模型：{embedding_model.split('/')[-1]}")
    with st.expander("高级设置", expanded=False):
        per_file_k = st.slider("每篇检索片段数", min_value=1, max_value=5, value=3)
        chunk_size = st.slider("Chunk 长度（字符）", 600, 1800, 1200, 100)
        chunk_overlap = st.slider("Chunk 重叠（字符）", 50, 400, 200, 50)
        st.caption("这些参数只影响文本切分和检索范围。")
    with st.expander("关于项目", expanded=False):
        st.caption("PDF 解析 → 文本向量化 → 按论文检索 → 大模型依据原文回答。")

st.title("📚 科研文献智能分析助手")
st.markdown(
    '<p class="app-subtitle">面向多篇学术论文的检索、问答与结构化比较</p>',
    unsafe_allow_html=True,
)

if not api_configured:
    st.warning("尚未检测到 SILICONFLOW_API_KEY。请在 .env 中填写后刷新页面。")

for duplicate_title in st.session_state.pop("duplicate_title_notices", []):
    st.warning(f"检测到疑似重复文献，已跳过：\n《{duplicate_title}》")

st.divider()
st.header("① 上传与解析")
uploaded_files = st.file_uploader(
    "上传一篇或多篇 PDF 论文", type=["pdf"], accept_multiple_files=True
)
parse_button = st.button("解析文献", type="primary", disabled=not uploaded_files)
st.caption("将提取正文并建立检索索引。每篇新论文只进行一次结构化信息提取。")

if parse_button:
    try:
        known_hashes = st.session_state.get("processed_hashes", set())
        new_files, skipped_names = deduplicate_pdf_files(uploaded_files, known_hashes)
        for skipped_name in skipped_names:
            st.warning(f"已跳过重复文件：{skipped_name}")

        if not new_files:
            st.info("没有新增文献，继续使用当前分析结果。")
        else:
            client = create_siliconflow_client()
            with st.status("正在解析文献……", expanded=True) as status:
                st.write("1/4 逐页读取论文文字")
                pages = extract_pdf_pages(new_files)
                if not pages:
                    raise ValueError("没有提取到文字。若 PDF 是扫描件，请先使用 OCR。")

                st.write("2/4 清洗文字并识别正文与参考文献")
                chunks = chunk_pages(pages, chunk_size, chunk_overlap)

                st.write("3/4 识别论文类型并检查重复标题")
                existing_store = st.session_state.get("rag_store")
                temporary_store = build_rag_store(client, chunks, embedding_model)
                new_document_ids = {file.file_hash for file in new_files}
                existing_profiles = dict(
                    st.session_state.get("paper_profiles", {})
                )
                title_duplicates = []
                profile_extraction_succeeded = False
                try:
                    extracted_profiles = extract_paper_profiles(
                        client, temporary_store, chat_model
                    )
                    new_profiles, title_duplicates = find_duplicate_paper_titles(
                        extracted_profiles, existing_profiles
                    )
                    duplicate_ids = {
                        item["document_id"] for item in title_duplicates
                    }
                    temporary_store = remove_documents_from_rag_store(
                        temporary_store, duplicate_ids
                    )
                    profile_extraction_succeeded = True
                    st.session_state["profiles_need_refresh"] = False
                except Exception as profile_exc:
                    # 类型抽取失败时仍允许正文进入索引，但不冒险按不可信标题删除论文。
                    new_profiles = {}
                    st.session_state["profiles_need_refresh"] = True
                    st.warning(f"论文可检索，但文献信息暂未生成：{profile_exc}")

                st.write("4/4 将保留的文献加入检索索引")
                rag_store = merge_rag_stores(existing_store, temporary_store)
                st.session_state["rag_store"] = rag_store
                st.session_state["processed_hashes"] = known_hashes | new_document_ids
                accepted_ids = (
                    set(new_profiles)
                    if profile_extraction_succeeded
                    else new_document_ids
                )
                existing_names = st.session_state.get("file_names", [])
                st.session_state["file_names"] = existing_names + [
                    file.name for file in new_files if file.file_hash in accepted_ids
                ]
                existing_profiles.update(new_profiles)
                st.session_state["paper_profiles"] = existing_profiles
                for state_key in (
                    "comparison_rows", "last_answer", "last_sources",
                    "last_evidence", "last_question",
                ):
                    st.session_state.pop(state_key, None)
                status.update(label="文献解析完成", state="complete", expanded=False)

            for duplicate in title_duplicates:
                st.warning(
                    "检测到疑似重复文献，已跳过：\n"
                    f'《{duplicate["title"]}》'
                )

            total_documents = len({_document_key(c) for c in rag_store["chunks"]})
            main_count = len(rag_store["chunks"])
            reference_count = sum(
                c.get("section") == "references"
                for c in rag_store.get("all_chunks", [])
            )
            metric_1, metric_2, metric_3 = st.columns(3)
            metric_1.metric("已解析文献", f"{total_documents} 篇")
            metric_2.metric("正文片段", f"{main_count} 个")
            metric_3.metric("已排除参考文献片段", f"{reference_count} 个")
    except Exception as exc:
        st.error(f"处理失败：{exc}")

rag_store = st.session_state.get("rag_store")
if not rag_store:
    st.info("请先上传 PDF，并点击“解析文献”。")
    st.stop()

paper_profiles = st.session_state.get("paper_profiles", {})
st.subheader("文献概览")
if st.session_state.get("profiles_need_refresh"):
    st.warning("部分旧缓存缺少论文类型，请点击“刷新文献信息”重新识别。")
if paper_profiles:
    for profile in paper_profiles.values():
        render_paper_card(profile)
else:
    st.info("论文正文已可检索，点击下方按钮可生成文献概览。")

if st.button("刷新文献信息（重新调用模型）", help="会为每篇论文重新提取一次结构化信息"):
    try:
        client = create_siliconflow_client()
        with st.spinner("正在逐篇刷新文献信息……"):
            refreshed_profiles = extract_paper_profiles(
                client, rag_store, chat_model
            )
            refreshed_profiles, refresh_duplicates = find_duplicate_paper_titles(
                refreshed_profiles
            )
            if refresh_duplicates:
                duplicate_ids = {
                    item["document_id"] for item in refresh_duplicates
                }
                rag_store = remove_documents_from_rag_store(
                    rag_store, duplicate_ids
                )
                st.session_state["rag_store"] = rag_store
                st.session_state["file_names"] = list(
                    dict.fromkeys(chunk["file_name"] for chunk in rag_store["chunks"])
                )
                st.session_state["duplicate_title_notices"] = [
                    item["title"] for item in refresh_duplicates
                ]
            st.session_state["paper_profiles"] = refreshed_profiles
            st.session_state["profiles_need_refresh"] = False
            st.session_state.pop("comparison_rows", None)
        st.success("文献信息已刷新。")
        st.rerun()
    except Exception as exc:
        st.error(f"刷新文献信息失败：{exc}")

st.divider()
st.header("② 文献问答")
st.markdown("**常用科研问题**")
quick_columns = st.columns(2)
for index, quick_question in enumerate(QUICK_QUESTIONS):
    quick_columns[index % 2].button(
        quick_question,
        key=f"quick_question_{index}",
        on_click=_queue_question,
        args=(quick_question,),
        width="stretch",
    )

question = st.text_input(
    "输入问题", key="question_input", placeholder="也可以输入你自己的科研问题……"
)
run_question = st.button("开始分析", type="primary", disabled=not question.strip())
run_question = run_question or st.session_state.pop("run_question", False)

if run_question and question.strip():
    try:
        client = create_siliconflow_client()
        with st.spinner("正在查找文献依据并组织回答……"):
            sources = retrieve_chunks(client, question, rag_store, per_file_k)
            raw_answer = answer_question(
                client, question, sources, chat_model,
                paper_profiles=st.session_state.get("paper_profiles", {}),
            )
            evidence = get_answer_evidence(raw_answer, sources)
            st.session_state["last_answer"] = strip_source_markers(raw_answer)
            st.session_state["last_sources"] = sources
            st.session_state["last_evidence"] = evidence
            st.session_state["last_question"] = question
    except Exception as exc:
        st.error(f"问答失败：{exc}")

if st.session_state.get("last_answer"):
    st.markdown("### 分析结果")
    st.markdown(st.session_state["last_answer"])
    render_answer_evidence(st.session_state.get("last_evidence", []))
    render_retrieval_debug(st.session_state.get("last_sources", []))

st.divider()
st.header("③ 文献对比")
st.caption("概览卡片和对比表复用同一份结构化结果，不会重复调用模型。")
if st.button("生成 / 刷新文献对比表"):
    try:
        profiles = st.session_state.get("paper_profiles", {})
        document_ids = {_document_key(chunk) for chunk in rag_store["chunks"]}
        missing_ids = document_ids - set(profiles)
        if missing_ids:
            client = create_siliconflow_client()
            with st.spinner("正在补充缺少的文献信息……"):
                missing_profiles = extract_paper_profiles(
                    client, rag_store, chat_model, document_ids=missing_ids
                )
                missing_profiles, title_duplicates = find_duplicate_paper_titles(
                    missing_profiles, profiles
                )
                if title_duplicates:
                    duplicate_ids = {
                        item["document_id"] for item in title_duplicates
                    }
                    rag_store = remove_documents_from_rag_store(
                        rag_store, duplicate_ids
                    )
                    st.session_state["rag_store"] = rag_store
                    for item in title_duplicates:
                        st.warning(
                            "检测到疑似重复文献，已跳过：\n"
                            f'《{item["title"]}》'
                        )
                profiles.update(missing_profiles)
                st.session_state["paper_profiles"] = profiles
        st.session_state["comparison_rows"] = build_comparison_table(profiles)
    except Exception as exc:
        st.error(f"生成对比表失败：{exc}")

if st.session_state.get("comparison_rows"):
    st.dataframe(
        st.session_state["comparison_rows"], width="stretch", hide_index=True
    )

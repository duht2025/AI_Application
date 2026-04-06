#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 问答系统 - 基于 PDF 文档的检索增强生成

功能：
1. 加载 PDF 文档，切分文本，构建向量数据库（使用 HuggingFace Embeddings + Chroma）
2. 支持通过预定义配置选择 LLM 后端：DeepSeek 或 Ollama
3. 对用户问题进行检索增强问答

使用前请确保已安装所需依赖：
    pip install langchain-community langchain-text-splitters langchain-chroma langchain-huggingface langchain-openai pypdf sentence-transformers chromadb
"""

import os
import sys
import logging
import traceback
from typing import Optional, List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # 注意：langchain_chroma 是新的包
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==================== 配置区（按需修改） ====================
# 设置 HuggingFace 镜像源（可选，用于加速模型下载）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 离线模式：若已缓存 Embedding 模型可开启，避免网络请求；若未缓存则报错
# 如需联网下载模型，请注释下面两行
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ---------- LLM 提供商选择 ----------
# 可选值: "ollama", "deepseek"
LLM_PROVIDER = "ollama"  # 修改此项切换后端

# Ollama 配置（当 LLM_PROVIDER = "ollama" 时生效）
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL_NAME = "qwen2.5:7b"   # 您本地 Ollama 已拉取的模型名
OLLAMA_API_KEY = "ollama"           # Ollama 的 fake api key

# DeepSeek 配置（当 LLM_PROVIDER = "deepseek" 时生效）
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL_NAME = "deepseek-chat"  # 或 "deepseek-reasoner"
# 请在环境变量中设置 DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
# 或直接写在这里（不推荐）
DEEPSEEK_API_KEY = "sk-d2b2650965144e48849cd7d1a6147ebf"

# ---------- Embedding 模型配置 ----------
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh"   # 可替换为其他本地或在线模型
PERSIST_DIR = "./chroma_db"                   # 向量数据库持久化目录

# ---------- 文档切分配置 ----------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_K = 3                               # 检索时返回的文档片段数

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==================== 辅助函数 ====================
def clean_path(raw_path: str) -> str:
    """
    清理用户输入的路径，去除引号、多余空格，并转换为绝对路径。

    Args:
        raw_path (str): 原始路径字符串

    Returns:
        str: 清理后的绝对路径
    """
    path = raw_path.strip()
    # 去除可能的 "& " 前缀（某些终端复制时产生）
    if path.startswith("& "):
        path = path[2:].strip()
    # 去除首尾的引号
    if (path.startswith("'") and path.endswith("'")) or (path.startswith('"') and path.endswith('"')):
        path = path[1:-1].strip()
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path


def get_llm() -> ChatOpenAI:
    """
    根据预定义的 LLM_PROVIDER 配置返回对应的 ChatOpenAI 实例。

    Returns:
        ChatOpenAI: 配置好的 LLM 客户端

    Raises:
        ValueError: 当 LLM_PROVIDER 配置错误或缺少必要 API Key 时抛出
    """
    if LLM_PROVIDER == "ollama":
        logger.info(f"初始化 Ollama LLM: model={OLLAMA_MODEL_NAME}, base_url={OLLAMA_BASE_URL}")
        return ChatOpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key=OLLAMA_API_KEY,
            model=OLLAMA_MODEL_NAME,
            temperature=0.1,          # 适当降低随机性
            verbose=False
        )
    elif LLM_PROVIDER == "deepseek":
        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your-api-key-here":
            raise ValueError("DeepSeek API Key 未正确设置，请检查 DEEPSEEK_API_KEY 环境变量或代码中的配置。")
        logger.info(f"初始化 DeepSeek LLM: model={DEEPSEEK_MODEL_NAME}, base_url={DEEPSEEK_BASE_URL}")
        return ChatOpenAI(
            base_url=DEEPSEEK_BASE_URL,
            api_key=DEEPSEEK_API_KEY,
            model=DEEPSEEK_MODEL_NAME,
            temperature=0.1
        )
    else:
        raise ValueError(f"不支持的 LLM_PROVIDER: {LLM_PROVIDER}，请选择 'ollama' 或 'deepseek'。")


# ==================== 核心类 ====================
class VectorStoreManager:
    """向量数据库管理器：负责构建、加载和查询向量库"""

    def __init__(self, embedding_model_name: str, persist_dir: str):
        """
        Args:
            embedding_model_name (str): HuggingFace Embedding 模型名称或路径
            persist_dir (str): 向量数据库持久化目录
        """
        self.embedding_model_name = embedding_model_name
        self.persist_dir = persist_dir
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vectorstore: Optional[Chroma] = None
        self._init_embeddings()

    def _init_embeddings(self):
        """初始化 Embedding 模型（支持本地离线模式）"""
        logger.info(f"加载 Embedding 模型: {self.embedding_model_name}")
        try:
            # 尝试加载本地模型（local_files_only=True 强制离线）
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"local_files_only": True}
            )
            logger.info("Embedding 模型加载成功（离线模式）")
        except Exception as e:
            logger.error(f"离线加载模型失败: {e}")
            # 如果离线失败且用户允许联网，可尝试在线加载，但此处按原要求仅提示
            logger.error("请确保模型已下载到本地缓存目录，或取消环境变量中的离线模式后重试。")
            raise RuntimeError(f"无法加载 Embedding 模型: {e}") from e

    def build_from_pdf(self, pdf_path: str) -> Chroma:
        """
        从 PDF 文件构建向量数据库（如已存在则覆盖重建）

        Args:
            pdf_path (str): PDF 文件路径

        Returns:
            Chroma: 构建好的向量数据库实例
        """
        logger.info(f"开始从 PDF 构建向量库: {pdf_path}")
        # 1. 加载 PDF
        logger.debug("加载 PDF 文档...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"PDF 加载完成，共 {len(documents)} 页")

        # 2. 切分文档
        logger.debug("切分文档...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        docs = text_splitter.split_documents(documents)
        logger.info(f"文档切分完成，共 {len(docs)} 个片段")

        # 3. 创建向量库（若目录已存在会清空重建，符合预期）
        logger.debug(f"创建/覆盖向量库，持久化目录: {self.persist_dir}")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        logger.info("向量库构建并持久化完成")
        self.vectorstore = vectorstore
        return vectorstore

    def load_existing(self) -> Chroma:
        """
        加载已有的向量数据库（若存在）

        Returns:
            Chroma: 已加载的向量库实例

        Raises:
            FileNotFoundError: 当持久化目录不存在或为空时抛出
        """
        if not os.path.exists(self.persist_dir) or not os.listdir(self.persist_dir):
            raise FileNotFoundError(f"向量库目录不存在或为空: {self.persist_dir}")
        logger.info(f"加载已有向量库: {self.persist_dir}")
        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        self.vectorstore = vectorstore
        return vectorstore

    def retrieve(self, query: str, k: int = RETRIEVAL_K) -> List[Tuple[str, float]]:
        """
        检索与查询最相似的文档片段

        Args:
            query (str): 查询文本
            k (int): 返回的片段数量

        Returns:
            List[Tuple[str, float]]: 列表，每个元素为 (文档内容, 相似度分数)
        """
        if self.vectorstore is None:
            raise RuntimeError("向量库未初始化，请先调用 build_from_pdf() 或 load_existing()")
        
        logger.debug(f"检索查询: {query[:100]}... (k={k})")
        
        # 先多取一些候选（例如 k*2），去重后可能少于 k
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k*2)
        
        # 去重（基于内容前100字符）
        seen = set()
        unique_results = []
        for doc, score in docs_with_scores:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique_results.append((doc.page_content, score))
        
        # 取前 k 个（去重后可能不足 k，但没关系）
        results = unique_results[:k]
        logger.info(f"检索到 {len(results)} 个相关片段（去重后）")
        return results

class RAGEngine:
    """RAG 问答引擎：结合检索结果和 LLM 生成答案"""

    def __init__(self, vectorstore_manager: VectorStoreManager, llm: ChatOpenAI):
        """
        Args:
            vectorstore_manager (VectorStoreManager): 向量库管理器
            llm (ChatOpenAI): 已配置的 LLM 实例
        """
        self.vector_manager = vectorstore_manager
        self.llm = llm
        self._setup_chain()

    def _setup_chain(self):
        """构建 LCEL 链：retrieve -> prompt -> llm -> output_parser"""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "你是一个严谨的问答助手。你必须严格根据下面提供的【上下文】来回答问题。如果上下文中包含答案，请用中文给出详细、清晰的回答；如果上下文中完全没有相关信息，才回答“不知道”。不要编造信息。"),
            ("human", "上下文：\n{context}\n\n问题：{question}\n答案：")
        ])
        self.chain = (
            {
                "context": self._retrieve_context,
                "question": RunnablePassthrough()
            }
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

    def _retrieve_context(self, question: str) -> str:
        
        results = self.vector_manager.retrieve(question, k=RETRIEVAL_K)
        for i, (content, score) in enumerate(results):
            logger.info(f"片段{i+1} (分数={score}): {content[:200]}...")
       
        """检索相关文档片段并拼接为上下文字符串"""
        try:
            results = self.vector_manager.retrieve(question, k=RETRIEVAL_K)
            context_parts = [f"[片段{i+1}] {content}" for i, (content, _) in enumerate(results)]
            full_context = "\n\n".join(context_parts)
            logger.debug(f"生成上下文长度: {len(full_context)} 字符")
            return full_context
        except Exception as e:
            logger.error(f"检索上下文失败: {e}")
            return "无法获取相关上下文信息。"

    def answer(self, question: str) -> str:
        """
        执行 RAG 问答

        Args:
            question (str): 用户问题

        Returns:
            str: LLM 生成的答案
        """
        logger.info(f"处理问题: {question[:100]}")
        try:
            answer = self.chain.invoke(question)
            logger.info("答案生成完成")
            return answer
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}\n{traceback.format_exc()}")
            return f"生成答案时出错: {e}"


# ==================== 主程序 ====================
def main():
    """主函数：交互式 RAG 问答流程"""
    logger.info("启动 RAG 问答系统")
    print("\n欢迎使用 RAG 问答系统（基于 PDF 文档）")

    # 1. 获取并验证 PDF 路径
    while True:
        raw_path = input("\n请输入 PDF 文件路径: ").strip()
        if not raw_path:
            print("错误：路径不能为空，请重新输入。")
            continue
        pdf_path = clean_path(raw_path)
        print(f"清理后的路径: {pdf_path}")
        if not os.path.isfile(pdf_path):
            print(f"错误：文件不存在 - {pdf_path}")
            continue
        break

    # 2. 初始化向量库管理器
    try:
        vec_manager = VectorStoreManager(
            embedding_model_name=EMBEDDING_MODEL_NAME,
            persist_dir=PERSIST_DIR
        )
        # 构建向量库（覆盖重建，确保与当前 PDF 同步）
        # 若希望增量更新可调整逻辑，此处按简单重建处理
        vectorstore = vec_manager.build_from_pdf(pdf_path)
        logger.info("向量库准备就绪")
    except Exception as e:
        logger.error(f"构建向量库失败: {e}")
        print(f"\n构建向量库失败，请检查 PDF 文件是否可读、Embedding 模型是否可用。\n错误详情：{e}")
        return

    # 3. 初始化 LLM
    try:
        llm = get_llm()
        logger.info(f"LLM 初始化成功: {LLM_PROVIDER}")
    except Exception as e:
        logger.error(f"LLM 初始化失败: {e}")
        print(f"\nLLM 初始化失败：{e}")
        return

    # 4. 创建 RAG 引擎
    rag = RAGEngine(vec_manager, llm)
    logger.info("RAG 引擎已就绪，进入问答循环")

    print("\n向量库构建完成，可以开始提问！（输入 exit 或 quit 退出）")
    # 5. 交互问答循环
    while True:
        query = input("\n问题: ").strip()
        if query.lower() in ("exit", "quit", "q"):
            print("退出程序。")
            break
        if not query:
            print("问题不能为空，请重新输入。")
            continue

        try:
            answer = rag.answer(query)
            print(f"\n答案：{answer}")
        except Exception as e:
            logger.error(f"问答过程发生异常: {e}", exc_info=True)
            print(f"\n处理问题时出错：{e}，请检查日志。")


if __name__ == "__main__":
    main()
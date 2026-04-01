import os
# ==================== 网络与镜像配置 ====================
# 设置 HuggingFace 镜像源（可选，如使用本地模型可忽略）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 强制离线模式（如果模型已缓存，可避免网络请求；若未缓存则会报错）
# 若需联网下载，请注释下面两行
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import traceback
import re
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def clean_path(raw_path: str) -> str:
    """清理用户输入的路径，去除引号、多余字符，并转换为绝对路径。"""
    path = raw_path.strip()
    if path.startswith("& "):
        path = path[2:].strip()
    if (path.startswith("'") and path.endswith("'")) or (path.startswith('"') and path.endswith('"')):
        path = path[1:-1].strip()
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path


def build_vectorstore(pdf_path: str, model_name: str, persist_dir: str = "./db"):
    """
    构建或加载向量数据库。

    Args:
        pdf_path (str): PDF文件路径
        model_name (str): Embedding模型名称或本地路径
        persist_dir (str): 向量库持久化目录
    """
    print("=" * 50)
    print("【步骤1】构建/加载向量库")
    print(f"PDF路径: {pdf_path}")
    print(f"Embedding模型: {model_name}")

    # 1. 加载PDF
    print("1.1 加载PDF文档...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"    加载完成，共 {len(documents)} 页")

    # 2. 切分文档
    print("1.2 切分文档...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"    切分完成，共 {len(docs)} 个片段")

    # for doc in docs:
    #    print(doc)
    
    # 3. 初始化 Embedding 模型
    print("1.3 初始化 Embedding 模型...")
    try:
        # 尝试加载模型（支持本地路径或在线模型名）
        embeddings = HuggingFaceEmbeddings(
            model_name = model_name,
            model_kwargs = {"local_files_only": True}  # 强制只使用本地文件
        )
        print("    模型加载完成")
    except Exception as e:
        print(f"    模型加载失败：{e}")
        print("    请确保模型已下载到本地，或取消离线模式并检查网络。")
        raise

    # 4. 向量库持久化
    print("1.4 创建向量库...")
    # if os.path.exists(persist_dir) and os.listdir(persist_dir):
    #     print(f"    检测到已有向量库目录 {persist_dir}，直接加载...")
    #     vectorstore = Chroma(
    #         persist_directory=persist_dir,
    #         embedding_function=embeddings
    #    )
    #    print("    向量库加载完成")
    # else:
    #    print("    未发现已有向量库，创建新库...")
    
    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=persist_dir
    )
    print("    向量库创建并持久化完成")

    print("【步骤1】结束")
    print("=" * 50)
    return vectorstore

def rag_query(query: str, vectorstore: Chroma, model_name: str = "qwen2.5:7b"):
    """检索增强问答"""
    print("=" * 50)
    print("【步骤2】执行RAG问答")
    print(f"问题: {query}")

    # 1. 检索
    print("2.1 检索相关文档片段 (k=3)...")
    retrieved_docs = vectorstore.similarity_search(query, k=3)

    context = "\n".join([doc.page_content for doc in retrieved_docs])
    print(f"    检索到 {len(retrieved_docs)} 个片段") 
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"    片段{i}: {doc.page_content[:1000]}...")

    # print("context: ", context)

    # 2. 构造提示
    prompt = f"""基于以下上下文回答问题。如果上下文里没有相关信息，请说不知道。
上下文：{context}
问题：{query}
答案："""

    # 3. 调用 LLM
    print("2.2 调用 LLM (Ollama)...")
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    
    response = client.chat.completions.create(
        model = model_name,
        messages = [{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    print("    调用完成")
    print("【步骤2】结束")
    print("=" * 50)
    return answer


def main():
    try:
        # 获取 PDF 路径
        raw_path = input("请输入 PDF 文件路径: ").strip()
        if not raw_path:
            print("错误：未输入 PDF 路径。")
            return
        pdf_path = clean_path(raw_path)
        print(f"清理后的路径: {pdf_path}")

        if not os.path.isfile(pdf_path):
            print(f"错误：文件不存在: {pdf_path}")
            return

        # 获取 Embedding 模型（支持本地路径或模型名）
        # model_input = input("请输入 Embedding 模型名称或本地路径（直接回车使用默认 BAAI/bge-small-zh）: ").strip()
        # if not model_input:
        model_input = "BAAI/bge-small-zh"  # 默认模型

        # 构建向量库
        vectorstore = build_vectorstore(pdf_path, model_input)

        # 循环问答
        while True:
            query = input("\n请输入问题（输入 'exit' 退出）: ").strip()
            if query.lower() in ["exit", "quit"]:
                print("退出程序。")
                break
            if not query:
                print("问题不能为空，请重新输入。")
                continue

            answer = rag_query(query, vectorstore)
            print(f"答案：{answer}")

    except Exception as e:
        print(f"程序执行出错：{e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
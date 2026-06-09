"""
任务2：构建本地RAG知识库
功能：批量读取文档 -> 文本分割 -> 向量化 -> 存入Chroma -> 相似度检索
"""
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ===================== 1. 配置参数 =====================
# 文档存放目录
DOCS_FOLDER = "docs"
# 向量数据库持久化路径（会自动创建）
CHROMA_DB_PATH = "chroma_db"
# 分块参数（严格按任务要求）
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# 检索返回数量
TOP_K = 3

# ===================== 2. 批量加载文档 =====================
def load_all_documents(folder_path: str):
    """
    加载指定文件夹下所有 PDF 和 DOCX 文档
    返回：文档列表
    """
    documents = []
    
    # 遍历文件夹
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 加载 PDF
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"✅ 已加载 PDF：{filename}")
        
        # 加载 DOCX
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"✅ 已加载 DOCX：{filename}")

    print(f"\n📄 总计加载文档数量：{len(documents)} 页/段")
    return documents

# ===================== 3. 文本分块 =====================
def split_documents(documents):
    """
    递归文本分割器：按要求分块
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", " ", ""],  # 中文友好分隔符
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ 文本分块完成，总计分块数：{len(chunks)}")
    return chunks

# ===================== 4. 初始化嵌入模型 & 向量库 =====================
def init_embeddings():
    """使用Ollama本地嵌入模型（nomic-embed-text）"""
    return OllamaEmbeddings(
        model="nomic-embed-text",  # Ollama 内置轻量嵌入模型
        base_url="http://localhost:11434"
    )

def init_vectorstore(chunks, embeddings, persist_path):
    """
    创建/加载 Chroma 向量数据库
    已存在则直接加载，不存在则创建
    """
    if os.path.exists(persist_path):
        print("🔍 加载已存在的向量数据库...")
        vectorstore = Chroma(
            persist_directory=persist_path,
            embedding_function=embeddings
        )
    else:
        print("🆕 创建新的向量数据库...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_path
        )
        vectorstore.persist()  # 持久化保存到磁盘
    
    print("✅ 向量库初始化完成！")
    return vectorstore

# ===================== 5. 检索函数（核心任务要求） =====================
def retrieve_relevant_chunks(query: str, vectorstore, top_k: int = TOP_K):
    """
    检索函数：输入问题，返回最相关的3个文本块
    """
    # 相似度检索
    relevant_docs = vectorstore.similarity_search(query, k=top_k)
    
    print(f"\n🔎 针对问题：{query}")
    print(f"🎯 检索到 {len(relevant_docs)} 条相关文本块：\n")
    
    # 格式化输出结果
    results = []
    for i, doc in enumerate(relevant_docs, 1):
        chunk_info = {
            "序号": i,
            "内容": doc.page_content,
            "来源": doc.metadata.get("source", "未知文件")
        }
        results.append(chunk_info)
        print(f"--- 第{i}块 ---")
        print(f"来源：{chunk_info['来源']}")
        print(f"内容：{chunk_info['内容'][:150]}...\n")
    
    return results, relevant_docs

# ===================== 主函数：一键构建知识库 =====================
if __name__ == "__main__":
    print("=" * 50)
    print("开始执行 任务2：构建本地知识库")
    print("=" * 50)
    
    # 1. 加载文档
    docs = load_all_documents(DOCS_FOLDER)
    
    # 2. 文本分块
    chunks = split_documents(docs)
    
    # 3. 初始化嵌入
    embeddings = init_embeddings()
    
    # 4. 构建向量库
    vector_db = init_vectorstore(chunks, embeddings, CHROMA_DB_PATH)
    
    # 5. 测试检索（你可以修改问题测试）
    test_query = "什么是自然语言处理？"
    retrieve_relevant_chunks(test_query, vector_db)
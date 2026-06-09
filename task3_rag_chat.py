"""
任务3：RAG检索增强对话问答链
整合文档加载、分块、向量库、对话检索链、Ollama大模型
约束提示词：仅依托参考文档作答，无信息固定回复指定话术
"""
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ===================== 全局配置 =====================
DOCS_FOLDER = "docs"
CHROMA_DB_PATH = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
OLLAMA_MODEL = "deepseek-r1:7b"
EMBED_MODEL = "nomic-embed-text"

# 自动创建文件夹
os.makedirs(DOCS_FOLDER, exist_ok=True)

# ===================== 1. 文档加载（复用任务2） =====================
def load_all_documents(folder_path: str):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"✅ 加载PDF：{filename}")
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"✅ 加载DOCX：{filename}")
        except Exception as e:
            print(f"⚠️ 文件{filename}加载异常：{str(e)}")
    print(f"\n📄 总共加载文档段落：{len(documents)}")
    return documents

# ===================== 2. 文本分块 =====================
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", " ", ""],
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️ 文本分块完成，总块数：{len(chunks)}")
    return chunks

# ===================== 3. 向量库初始化 =====================
def init_vector_store(chunks=None):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url="http://localhost:11434")
    if os.path.exists(CHROMA_DB_PATH) and chunks is None:
        print("🔍 读取已存在向量库")
        db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    else:
        print("🆕 新建向量数据库")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        db.persist()
    return db

# ===================== 4. 构建ConversationalRetrievalChain问答链（任务核心） =====================
def build_qa_chain(vector_db):
    # 初始化对话记忆，保存多轮聊天上下文
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # 自定义强制约束提示词（满足题目要求）
    custom_prompt_template = """
你是私有知识库问答助手，必须严格遵守以下规则：
1. 你的回答只能完全依据下面提供的【参考文档内容】，禁止编造、联想、外部常识拓展；
2. 如果参考文档里完全没有对应问题的信息，**只允许输出一句话：文档中未找到相关答案**，禁止额外解释；
3. 回答语言简洁通顺，条理清晰，不要出现多余思考过程；

【参考文档内容】：
{context}

【用户历史对话】：
{chat_history}

【用户当前问题】：{question}
"""
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

    # 连接本地Ollama大模型
    llm = Ollama(model=OLLAMA_MODEL, base_url="http://localhost:11434")

    # 核心：组装对话检索链，绑定检索器、大模型、记忆、自定义提示词
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": TOP_K}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain, memory

# ===================== 命令行交互入口 =====================
if __name__ == "__main__":
    print("="*60)
    print("任务3 本地RAG对话问答系统（命令行版）")
    print("="*60)

    # 步骤A：执行文档加载、分块、构建向量库
    docs = load_all_documents(DOCS_FOLDER)
    chunks = split_docs(docs)
    vector_store = init_vector_store(chunks)

    # 步骤B：生成问答链实例
    qa_chain, memory = build_qa_chain(vector_store)
    print("\n💡 知识库就绪！输入问题提问，输入exit退出程序")
    print("-"*60)

    # 步骤C：循环多轮问答交互
    while True:
        user_q = input("\n请输入你的问题：")
        # 退出指令
        if user_q.strip().lower() == "exit":
            print("👋 程序退出")
            break
        # 空输入过滤
        if not user_q.strip():
            print("⚠️ 请输入有效问题")
            continue
        # 调用RAG链生成回答
        result = qa_chain.invoke({"question": user_q})
        answer = result["answer"]
        print(f"\n🤖 回答：{answer}")
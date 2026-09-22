import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========配置，和rag_demo保持一致=========
PERSIST_DIR = "./chroma_db"
BGE_MODEL_NAME = "BAAI/bge-small-zh"
DOC_FOLDER = "./txt_docs"

# 初始化嵌入模型
embeddings = HuggingFaceBgeEmbeddings(
    model_name=BGE_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 加载已有向量库
vector_db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

# 文本切分器，和之前RAG保持同样chunk策略
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=80,
    length_function=len
)

def load_all_txt(folder_path):
    all_text = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                all_text.append(content)
                print(f"✅读取文件：{filename}，文本长度：{len(content)}")
    return all_text

if __name__ == "__main__":
    # 读取全部txt文件
    raw_text_list = load_all_txt(DOC_FOLDER)
    all_chunks = []
    for text in raw_text_list:
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)

    # 追加入库，不会覆盖原有向量库里面选股csv数据
    vector_db.add_texts(texts=all_chunks)
    print(f"\n✅ 入库完成！一共新增 {len(all_chunks)} 个文本块追加进Chroma向量库")
    print("向量库目录：", PERSIST_DIR)
    print("现在运行 python rag_demo.py，即可同时检索选股技术指标 + txt基本面财报")

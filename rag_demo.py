# 环境变量，HF镜像，解决SSL证书问题
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama

# ====================== 1. 配置参数 ======================
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
RECALL_K = 3
PERSIST_DIR = "./chroma_db"

# ====================== 2. 加载文档 + 切分Chunk ======================
loader = TextLoader("stock_rule.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len
)
splits = text_splitter.split_documents(docs)
print(f"文档切分完成，一共 {len(splits)} 个chunk")

# ====================== 3. BGE本地Embedding向量模型 ======================
model_name = "BAAI/bge-small-zh"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
bge_embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# ======================4. 存入Chroma向量库（新版自动持久化）======================
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=bge_embedding,
    persist_directory=PERSIST_DIR
)
retriever = vectorstore.as_retriever(search_kwargs={"k": RECALL_K})

# ====================== 初始化本地大模型 qwen:1.8b ======================
llm = ChatOllama(model="qwen:1.8b")

# ====================== 交互式问答循环（整合选股csv） ======================
if __name__ == "__main__":
    print("=====股票RAG问答系统，输入 quit 退出=====")
    while True:
        question = input("\n请输入你的问题：")
        if question.strip().lower() == "quit":
            print("退出程序")
            break

        # 读取akshare筛选出来的股票清单
        try:
            stock_df = pd.read_csv("selected_stocks.csv")
            stock_markdown = stock_df.to_markdown(index=False)
        except Exception:
            stock_markdown = "【提示】暂无选股结果，请先运行 stock_filter.py 生成 selected_stocks.csv"

        # 检索知识库
        retrieved_chunks = retriever.invoke(question)
        context_text = ""
        print("\n【检索召回的策略文档片段】")
        for idx, chunk in enumerate(retrieved_chunks):
            print(f"片段{idx+1}: {chunk.page_content}")
            context_text += chunk.page_content + "\n"

        # 构造Prompt：策略知识库 + 选股清单一起送入大模型
        prompt = f"""
你是中短线量化选股助手，**必须严格依据下面【选股策略文档】和【候选股票清单】回答，禁止编造不存在的指标与基本面信息**。
只使用提供的资料，超出资料范围的内容如实说明资料不足。

【选股策略文档】
{context_text}

【候选股票清单】
{stock_markdown}

用户问题：{question}
"""
        print("\n=====AI回答=====")
        resp = llm.invoke(prompt)
        print(resp.content)
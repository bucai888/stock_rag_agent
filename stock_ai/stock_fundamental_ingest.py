import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

import time
import akshare as ak
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ========= 配置区（必须和 rag_demo.py 完全一致！）=========
PERSIST_DIR = "./chroma_db"          # 注意：和 rag_demo.py 同一个目录
BGE_MODEL_NAME = "BAAI/bge-small-zh"

# 要抓取的股票（stock_filter.py 选出的那两只）
stock_codes = [
    {"code": "000002", "name": "万科A"},
    {"code": "000037", "name": "深南电A"},
]

# 初始化BGE嵌入模型（和rag_demo保持一致）
embeddings = HuggingFaceBgeEmbeddings(
    model_name=BGE_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 加载已有Chroma库（直接连，不清空旧数据）
vector_db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)

# 文本切分器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=80,
    length_function=len,
)


def safe_call(fn, *args, label="", **kwargs):
    """单个接口失败不影响其他接口"""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"  ⚠️ {label} 抓取失败: {e}")
        return None


def get_stock_fundamental(stock_code, stock_name):
    doc_text = f"===== {stock_name}（{stock_code}）基本面与财报信息 =====\n"

    # 1. 个股概况（主营业务、行业、总股本等）——东财源，用纯6位代码
    info_df = safe_call(
        ak.stock_individual_info_em, symbol=stock_code, label="个股概况"
    )
    if info_df is not None:
        doc_text += "\n【公司基本概况】\n"
        for _, row in info_df.iterrows():
            doc_text += f"{row['item']}: {row['value']}\n"

    time.sleep(1.0)

    # 2. 主要财务指标（PE、ROE、营收、净利润等）——新浪源，需要 sz/sh 前缀
    sina_code = ("sh" if stock_code.startswith("6") else "sz") + stock_code
    fin_df = safe_call(
        ak.stock_financial_analysis_indicator,
        symbol=sina_code,
        label="财务指标",
    )
    if fin_df is not None:
        fin_df = fin_df.head(8)  # 最近8期
        doc_text += "\n【近期核心财务指标】\n"
        doc_text += fin_df.to_string() + "\n"

    time.sleep(1.0)

    # 3. 季度财务摘要
    quarter_df = safe_call(
        ak.stock_financial_report_sina,
        stock=sina_code,
        symbol="quarterly",
        label="季度财报",
    )
    if quarter_df is not None:
        quarter_df = quarter_df.head(5)
        doc_text += "\n【季度财务摘要】\n"
        doc_text += quarter_df.to_string() + "\n"

    return doc_text


if __name__ == "__main__":
    all_texts = []
    for stock in stock_codes:
        print(f"正在抓取 {stock['name']} {stock['code']} ...")
        content = get_stock_fundamental(stock["code"], stock["name"])
        chunks = text_splitter.split_text(content)
        all_texts.extend(chunks)
        print(f"  {stock['name']} 生成 {len(chunks)} 个文本块")
        time.sleep(1.5)

    # 追加写入向量库（add_texts 是追加，不清空旧数据）
    vector_db.add_texts(texts=all_texts)
    print(f"\n✅ 全部完成！本次新增 {len(all_texts)} 个文本块")
    print(f"知识库位置：{PERSIST_DIR}（和 rag_demo.py 共用）")
    print("现在可以运行 python rag_demo.py 提问了")

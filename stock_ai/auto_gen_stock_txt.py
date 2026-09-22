import akshare as ak
import os
import time
import pandas as pd

# ========= 配置区 =========
OUTPUT_FOLDER = "./txt_docs"
# 选股csv路径，stock_filter.py输出文件
CSV_PATH = "./selected_stocks.csv"
col_code = "code"   # csv股票代码列名
col_name = "name"   # csv股票名称列名
SLEEP_TIME = 3

# 创建txt_docs文件夹
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def safe_fetch(fn, *args, label="", retry=2, **kwargs):
    """带重试akshare抓取，失败返回None"""
    for i in range(retry):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"  ⚠️ {label} 第{i+1}次抓取失败：{str(e)}")
            time.sleep(SLEEP_TIME)
    print(f"  ❌ {label} 全部重试失败")
    return None


def fetch_stock_info(stock_code, stock_name):
    content = f"===== {stock_name}（{stock_code}）基本面与财报信息 =====\n"

    # 1.个股概况
    info_df = safe_fetch(ak.stock_individual_info_em, symbol=stock_code, label="个股概况")
    if info_df is not None:
        content += "\n【公司基本概况】\n"
        for _, row in info_df.iterrows():
            content += f"{row['item']}: {row['value']}\n"
    time.sleep(SLEEP_TIME)

    # 拼接sz/sh前缀
    sina_prefix = "sh" if stock_code.startswith("6") else "sz"
    sina_code = sina_prefix + stock_code

    # 2.财务指标
    fin_df = safe_fetch(ak.stock_financial_analysis_indicator, symbol=sina_code, label="财务指标")
    if fin_df is not None:
        fin_df = fin_df.head(8)
        content += "\n【近期核心财务指标】\n"
        content += fin_df.to_string() + "\n"
    time.sleep(SLEEP_TIME)

    #3.季度财报
    quarter_df = safe_fetch(ak.stock_financial_report_sina, stock=sina_code, symbol="quarterly", label="季度财报")
    if quarter_df is not None:
        quarter_df = quarter_df.head(5)
        content += "\n【季度财务摘要】\n"
        content += quarter_df.to_string() + "\n"

    # 人工编辑预留区域
    content += "\n\n=====【人工手动编辑区】=====\n"
    content += "可手动补充：行业点评、风险、技术面、基本面判断、新闻、公告摘要\n"
    return content


if __name__ == "__main__":
    print(f"读取选股csv文件：{CSV_PATH}")
    # 读取csv，股票代码强制字符串，防止开头0丢失
    df = pd.read_csv(CSV_PATH, dtype={col_code: str})
    stock_list = []
    for _, row in df.iterrows():
        code = str(row[col_code]).strip()
        name = str(row[col_name]).strip()
        stock_list.append({"code":code, "name":name})

    print(f"✅ 从csv读取到 {len(stock_list)} 只股票")
    for stock in stock_list:
        name = stock["name"]
        code = stock["code"]
        print(f"\n正在抓取：{name} {code}")
        text_content = fetch_stock_info(code, name)
        save_path = os.path.join(OUTPUT_FOLDER, f"{name}_{code}.txt")

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        print(f"✅ 文件生成：{save_path}")
        time.sleep(SLEEP_TIME)

    print("\n====全部txt草稿生成完毕！====")
    print("👉 打开txt_docs文件夹，人工校对、修改txt内容")
    print("👉 修改完成后运行 python txt_ingest.py 入库")
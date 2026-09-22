import akshare as ak
import pandas as pd
import time

def get_main_board_stocks():
    stock_info_df = ak.stock_info_a_code_name()
    # 沪主板60开头，深主板000开头
    stock_info_df = stock_info_df[
        (stock_info_df["code"].str.startswith("60")) |
        (stock_info_df["code"].str.startswith("000"))
    ]
    return stock_info_df

def get_kline_retry(code, max_retry=3):
    """
    带重试的日线接口。
    关键：ak.stock_zh_a_daily(新浪源) 必须带 sh/sz 前缀，
    否则内部会报 KeyError: 'date'。
    """
    if code.startswith("6"):
        symbol = "sh" + code      # 上海
    else:
        symbol = "sz" + code      # 深圳

    for attempt in range(max_retry):
        try:
            df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
            return df
        except Exception as e:
            print(f"  重试{attempt+1}次 {code} 失败: {e}")
            time.sleep(1.5)
    return None

def filter_stock_by_rule():
    stock_list = get_main_board_stocks()
    # 测试阶段：先取前30只，确认稳定后删掉这行即可全量
    stock_list = stock_list.head(30).reset_index(drop=True)
    result_list = []
    total = len(stock_list)
    print(f"本次测试股票数量：{total}，开始筛选...")

    for idx, row in stock_list.iterrows():
        code = row["code"]
        name = row["name"]
        print(f"\n==== [{idx+1}/{total}] 正在处理 {code} {name} ====")

        # 跳过ST
        if "ST" in name or "*ST" in name:
            print(f"  是ST，跳过")
            continue

        df = get_kline_retry(code)
        if df is None:
            print(f"  ❌ {code} 获取K线全部重试失败，跳过")
            continue

        # 兼容：date 可能是索引也可能是列
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)
        else:
            df = df.sort_index().reset_index(drop=True)

        if len(df) < 20:
            print(f"  K线数据不足20根(共{len(df)}根)，跳过")
            continue

        # 计算均线（新浪源字段：close / volume）
        df["ma5"]   = df["close"].rolling(5).mean()
        df["ma10"]  = df["close"].rolling(10).mean()
        df["ma20"]  = df["close"].rolling(20).mean()
        df["vol5"]  = df["volume"].rolling(5).mean()
        df["vol20"] = df["volume"].rolling(20).mean()

        latest = df.iloc[-1]
        price  = latest["close"]
        ma5    = latest["ma5"]
        ma10   = latest["ma10"]
        ma20   = latest["ma20"]
        vol5   = latest["vol5"]
        vol20  = latest["vol20"]

        cond_price = price < 20
        cond_ma    = ma5 > ma10 > ma20
        cond_vol   = vol5 > vol20

        print(f"  价格:{price:.2f} | 股价<20:{cond_price} | MA5>MA10>MA20:{cond_ma} | 放量:{cond_vol}")

        if cond_price and cond_ma and cond_vol:
            print(f"  ✅ 【入选】{code} {name}")
            result_list.append({
                "code": code,
                "name": name,
                "price": round(price, 2),
                "ma5":   round(ma5, 2),
                "ma10":  round(ma10, 2),
                "ma20":  round(ma20, 2),
                "vol5":  int(vol5),
                "vol20": int(vol20)
            })
        # 加长休眠，降低请求频率
        time.sleep(1.0)

    res_df = pd.DataFrame(result_list)
    return res_df

if __name__ == "__main__":
    df = filter_stock_by_rule()
    print("\n" + "=" * 40)
    print("==== 符合条件的股票清单 ====")
    print(df if len(df) > 0 else "（本批次无股票满足全部条件）")
    df.to_csv("selected_stocks.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 已保存 selected_stocks.csv，共 {len(df)} 只")
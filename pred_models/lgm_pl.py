import pandas as pd
from decimal import Decimal, getcontext

getcontext().prec = 10

# pl_all_500_1601-2212.csvの読み込み
df_pl = pd.read_csv('data/features/pl_all_500_1601-2212.csv')
# results.csvの読み込み
df_result = pd.read_csv('results.csv')
# print(df_result)
# Invert列を削除
df_pl.drop(columns=['Invest'], inplace=True)

# Date列を日付に変換
df_pl['Date'] = pd.to_datetime(df_pl['Date'], unit='D', origin='1899-12-30')

# 日付のフォーマットを統一する
df_pl['Date'] = df_pl['Date'].dt.strftime('%Y%m%d')

# df_resultのDate列を整数型に変換
df_result['Date'] = df_result['Date'].astype(str)

# データのマージ結合
df_merged = pd.merge(df_pl, df_result, on='Date', how='inner')
# print(df_merged)

# # 日別利益の保存
# df_merged.to_csv('merge.csv', header=True)

# 利益の計算
df_merged['P/L'] = df_merged['P/L'].apply(lambda x:Decimal(str(x)))
df_merged['pred'] = df_merged['pred'].apply(lambda x:Decimal(str(x)))
df_merged['Profit'] = df_merged.apply(lambda row: row['P/L'] * row['pred'],axis=1)

df_merged['Profit'] = df_merged['Profit'].apply(lambda x:0 if x == -0 else x)

# 日別の利益計算
daily_profit = df_merged.groupby('Date')['Profit'].sum()

# トータルの利益計算
total_profit = daily_profit.sum()

# 日別利益の保存
daily_profit.to_csv('profit.csv', header=True)

print("日別の利益:")
print(daily_profit)
print("トータルの利益:")
print(total_profit)



import pandas as pd
import matplotlib.pyplot as plt

# データを読み込む
df1 = pd.read_csv("data/features/pl_all_500_1601-2212.csv")
df2 = pd.read_csv("data/features/feature_historical_1212-2303.csv", usecols=['date','label'])

# 'date' 列を 'Date' に変更して新しいデータフレームを作成
df2 = df2.rename(columns={'date': 'Date'})

# Date列を日付に変換
df1['Date'] = pd.to_datetime(df1['Date'], unit='D', origin='1899-12-30')
# 'Date' 列を文字列に変換
df1['Date'] = df1['Date'].astype(str)
df2['Date'] = df2['Date'].astype(str)

# 日付のフォーマットを統一する
df1['Date'] = df1['Date'].str.replace('-', '')

# 日付を基準にしてデータを結合
merged_df = pd.merge(df1, df2, on="Date")

# ラベルが1のときのP/Lとラベルが0のときのP/Lを抽出
profit_when_1 = merged_df.loc[merged_df['label'] == 1, 'P/L']
profit_whe_0 = merged_df.loc[merged_df['label'] == 0, 'P/L']

# P/Lが0を基準に分割
profit_when_1_positive = profit_when_1[profit_when_1 > 0]
profit_when_1_negative = profit_when_1[profit_when_1 <= 0]

profit_when_0_positive = profit_when_0[profit_when_0 > 0]
profit_when_0_negative = profit_when_0[profit_when_0 <= 0]

# 基本統計量の計算
stats_when_1_positive = profit_when_1_positive.describe()
stats_when_1_negative = profit_when_1_negative.describe()

stats_when_0_positive = profit_when_0_positive.describe()
stats_when_0_negative = profit_when_0_negative.describe()

print("Label 1 Positive P/L Stats:")
print(stats_when_1_positive)
print("\nLabel 1 Negative P/L Stats:")
print(stats_when_1_negative)

print("\nLabel 0 Positive P/L Stats:")
print(stats_when_0_positive)
print("\nLabel 0 Negative P/L Stats:")
print(stats_when_0_negative)

# プロット
plt.figure(figsize=(12, 6))

# ラベル1のP/Lのヒストグラム
plt.subplot(1, 2, 1)
plt.hist(profit_when_1_positive, bins=30, alpha=0.6, color='b', label='Positive P/L')
plt.hist(profit_when_1_negative, bins=30, alpha=0.6, color='r', label='Negative P/L')
plt.title('Label 1 Profit/Loss Histogram')
plt.xlabel('P/L')
plt.ylabel('Frequency')
plt.legend()

# ラベル0のP/Lのヒストグラム
plt.subplot(1, 2, 2)
plt.hist(profit_when_0_positive, bins=30, alpha=0.6, color='b', label='Positive P/L')
plt.hist(profit_when_0_negative, bins=30, alpha=0.6, color='r', label='Negative P/L')
plt.title('Label 0 Profit/Loss Histogram')
plt.xlabel('P/L')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

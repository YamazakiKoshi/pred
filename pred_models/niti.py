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
profit_1 = merged_df.loc[merged_df['label'] == 1, 'P/L']
profit_0 = merged_df.loc[merged_df['label'] == 0, 'P/L']

# ラベル1の基本統計量
statistics_1 = profit_1.describe()
print(statistics_1)

# ラベル0の基本統計量
statistics_0 = profit_0.describe()
print(statistics_0)

merged_df.to_csv('result.scv', ndex=None, encoding='utf-8')
# プロット
plt.figure(figsize=(10, 6))

# ラベル1のヒストグラム
plt.subplot(1, 2, 1)
plt.hist(profit_1, bins=30, color='b', alpha=0.7, edgecolor='black', linewidth=1.0)
plt.title('Label 1 Profit/Loss Histogram')
plt.xlabel('P/L')
plt.ylabel('Frequency')
plt.xlim(-1200,1200)

# ラベル0のヒストグラム
plt.subplot(1, 2, 2)
plt.hist(profit_0, bins=30, color='g', alpha=0.7, edgecolor='black', linewidth=1.0)
plt.title('Label 0 Profit/Loss Histogram')
plt.xlabel('P/L')
plt.ylabel('Frequency')
plt.xlim(-800,800)

plt.tight_layout()
plt.show()



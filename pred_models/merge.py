import pandas as pd
import matplotlib.pyplot as plt

# データを読み込む
df1 = pd.read_csv("data/features_VI/feature_profit_1212-2212_regression.csv")
df2 = pd.read_csv("data/features_news/feature_news_bert_1601-2212.csv")

# 'date' 列を 'Date' に変更して新しいデータフレームを作成
df2 = df2.rename(columns={'date': 'Date'})

# 'Date' 列を文字列に変換
df1['Date'] = df1['Date'].astype(str)
df2['Date'] = df2['Date'].astype(str)

# 日付のフォーマットを統一する
df2['Date'] = df2['Date'].str.replace('-', '')

# 日付を基準にしてデータを結合し、feature_profit_1212-2212.csvのlabel列だけを追加
merged_df = pd.merge(df2, df1[['Date', 'P/L']], on="Date", how='outer')

# モデル結果の保存
merged_df.to_csv(f'data/features_VI/merge_news_bert_regression.csv', index=None, encoding='utf-8')
print(merged_df)


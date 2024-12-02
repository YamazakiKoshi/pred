import pandas as pd
import numpy as np
import datetime
import warnings
from tqdm import tqdm
warnings.filterwarnings(action="ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import optuna.integration.lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def down_sample(X, y, rate, random_sampling=True):
    s = y.value_counts().sum()
    t = y.value_counts()[1]
    num = s - (t // (rate / 100))
    if num < 0:
        num = 0  # numが負の場合は0に設定
    if random_sampling:
        index_list = y[y==0].sample(n=int(num), random_state=42).index
    else:
        index_list = y[y==0].iloc[:int(num)].index
    _y = y.drop(index_list).reset_index(drop=True)
    _X = X.drop(index_list).reset_index(drop=True)
    return _X, _y

def calculate_label(proba):
    return 1 if proba >= 0.5 else 0

def main():
    now = datetime.datetime.now().strftime("%Y%m%d")
    
    input_path = 'data/features_news/feature_lda_128_1212-2212.csv'
    vec_name = 'lda_32'
    pre = 50
    
    df = pd.read_csv(input_path)
    df = df.reset_index(drop=True)
    
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    
    df_2016 = df[(df['Date'] >= '2016-01-01') & (df['Date'] < '2017-01-01')]
    N_train = df_2016.shape[0]

    RANDOM_STATE = 43
    TEST_SIZE = 0.3
    
    drop_col = ["Date", "label"]
    pred_list = [] 
    count_pred_1 = 0

    for i in tqdm(range(df.shape[0] - N_train)):  # ループの開始位置を調整
        train_df = df.iloc[:i+N_train]
        test_df = df.iloc[N_train + i : N_train + i + 1]  # 1日のデータを取得
        
        # 予測開始日を1月4日に設定
        if test_df.iloc[0]['Date'] < pd.to_datetime('2017-01-04'):
            continue
        
        train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        x_test = test_df.drop(drop_col, axis=1)
        y_test = test_df['label']
        X = train_df.drop(drop_col, axis=1)
        y = train_df['label']
        
        x_train, y_train = down_sample(X, y, pre)
        
        if x_train.empty or y_train.empty:
            continue  # ダウンサンプリングの結果が空の場合はスキップ
        
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False)
        
        if x_valid.empty or y_valid.empty:
            continue  # バリデーションデータが空の場合はスキップ
        
        x_valid = x_valid.iloc[5:,]  # バリデーションデータの削除行数を調整
        y_valid = y_valid.iloc[5:,]  # バリデーションデータの削除行数を調整
        
        if x_valid.empty or y_valid.empty:
            continue  # バリデーションデータが空の場合はスキップ
        
        trains = lgb.Dataset(x_train, y_train)
        valids = lgb.Dataset(x_valid, y_valid, reference=trains)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
        }
        
        model = lgb.train(params, trains, num_boost_round=1000, valid_sets=valids, 
                          callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True), lgb.log_evaluation(0)])
        
        y_proba = model.predict(x_test, num_iteration=model.best_iteration)
        y_pred = calculate_label(y_proba)
        pred_list.append([test_df['Date'].to_list()[0], y_pred, y_proba[0]])
        count_pred_1 += y_pred

        gc.collect()

    df_pred = pd.DataFrame(pred_list, columns=['Date', 'pred', 'proba'])
    df_date = df_pred['Date']
    _df = df['label'].iloc[N_train:]
    _df.reset_index(drop=True, inplace=True)
    df_pred.reset_index(drop=True, inplace=True)
    df_result = pd.concat([df_pred, _df], axis=1)
    
    # NaN値の行を削除
    df_result = df_result.dropna(subset=['label', 'pred'])
    
    df_result.to_csv('results2.csv', index=None, encoding='utf-8')
    
    df_result2 = df_result.iloc[735:,]
    print(classification_report(df_result2['label'], df_result2['pred']))

    print(df_result2)
    print("count_pred_1")
    print(count_pred_1)

if __name__ == "__main__":
    main()




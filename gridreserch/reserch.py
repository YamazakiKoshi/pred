import pandas as pd
import numpy as np
import datetime
import warnings
from tqdm import tqdm
warnings.filterwarnings(action="ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

def down_sample(X, y, rate, random_sampling=True):
    s = y.value_counts().sum()
    t = y.value_counts()[1]
    num = s - (t // (rate / 100))
    num = max(0, num)
    if random_sampling:
        index_list = y[y == 0].sample(n=int(num), random_state=42).index
    else:
        index_list = y[y == 0].iloc[:int(num)].index
    _y = y.drop(index_list).reset_index(drop=True)
    _X = X.drop(index_list).reset_index(drop=True)
    return _X, _y

def calculate_label(proba):
    if proba < 0.5:
        label = 0
    else:
        label = 1
    return label

def main():
    now = datetime.datetime.now().strftime("%Y%m%d")
    
    # input_path = 'data/features_VI/merge_news_bert_50.csv'
    input_path = 'data_hosokawa/feature_news_bert_label_50.csv'
    vec_name = 'bert_128'
    pre = 50
    
    df = pd.read_csv(input_path)
    df = df.reset_index(drop=True)
    N_train = df[(df['Date'] >= 20160104) & (df['Date'] <= 20161230)].shape[0]

    RANDOM_STATE = 43
    TEST_SIZE = 0.3
    
    drop_col = ["Date", "label_50"]
    pred_list = [] 
    count_pred_1 = 0
    loss_history = []  


    for i in tqdm(range(df.shape[0] - (N_train))):
        train_df = df.iloc[:i + N_train]
        test_df = df.iloc[N_train + i: N_train + i + 1]

        train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        x_test = test_df.drop(drop_col, axis=1)
        y_test = test_df['label_50']
        X = train_df.drop(drop_col, axis=1)
        y = train_df['label_50']
        
        train_df = train_df.dropna(subset=['label_50'])
        test_df = test_df.dropna(subset=['label_50'])

        x_train, y_train = down_sample(X, y, pre)
        
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)

        param_grid = {
            'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
            'learning_rate_init': [0.001, 0.0005, 0.0001],
            'activation': ['relu'],
            'solver': ['adam'],
            'early_stopping': [True],
            'n_iter_no_change':[200],
            'max_iter':[1000],
        }

        mlp = MLPClassifier(random_state=RANDOM_STATE)
        grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=10, scoring='f1', n_jobs=-1, verbose=1)
        
        grid_search.fit(x_train, y_train)
        
        best_model = grid_search.best_estimator_

        loss_history.append(best_model.loss_curve_)

        y_proba = best_model.predict_proba(x_test)[:, 1]
        y_pred = calculate_label(y_proba[0])
        pred_list.append([test_df['Date'].to_list()[0], y_pred, y_proba[0]])
        count_pred_1 += y_pred
        
        gc.collect()

    df_pred = pd.DataFrame(pred_list, columns=['Date', 'pred', 'proba'])
    df_date = df_pred['Date']
    _df = df['label_50'].iloc[(N_train):]
    _df.reset_index(drop=True, inplace=True)
    df_pred.reset_index(drop=True, inplace=True)
    df_result = pd.concat([df_pred, _df], axis=1)
    df_result = df_result.dropna()

    # モデル結果の保存
    df_result.to_csv(f'data/result/{vec_name}_mlp_result_50.csv', index=None, encoding='utf-8')

    # パラメータごとの精度を確認
    cv_results = pd.DataFrame(grid_search.cv_results_)
    print(cv_results[['param_hidden_layer_sizes', 'param_learning_rate_init', 'mean_test_score', 'rank_test_score']])
    
    print(classification_report(df_result['label_50'], df_result['pred']))
    print(df_result)
    print("count_pred_1")
    print(count_pred_1)

    # エポック数の確認
    print(f"Best model's epoch count: {best_model.n_iter_}")

    # 混同行列の描画
    cm = confusion_matrix(df_result['label_50'], df_result['pred'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # 損失曲線の描画
    plt.figure(figsize=(10, 6))
    plt.plot(best_model.loss_curve_, label='Loss Curve (Best Model)')
    plt.title('Loss Curve during Training (Best Model)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


# import pandas as pd
# import numpy as np
# import datetime
# import warnings
# from tqdm import tqdm
# warnings.filterwarnings(action="ignore")
# import matplotlib.pyplot as plt
# import seaborn as sns
# import gc
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split

# def calculate_label(proba):
#     return 1 if proba >= 0.5 else 0

# def main():
#     now = datetime.datetime.now().strftime("%Y%m%d")
    
#     input_path = 'data/features_VI/merge_news_bert_50.csv'
#     vec_name = 'bert_128_downloss'
#     pre = 50
    
#     df = pd.read_csv(input_path)
#     df = df.reset_index(drop=True)
#     N_train = df[(df['Date'] >= 20160104) & (df['Date'] <= 20161230)].shape[0]

#     RANDOM_STATE = 43
#     TEST_SIZE = 0.3
    
#     drop_col = ["Date", "label_50"]
#     pred_list = [] 
#     count_pred_1 = 0
#     loss_history = []  

#     for i in tqdm(range(df.shape[0] - (N_train))):
#         train_df = df.iloc[:i + N_train]
#         test_df = df.iloc[N_train + i: N_train + i + 1]

#         # データの前処理
#         train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
#         test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)
#         x_test = test_df.drop(drop_col, axis=1)
#         y_test = test_df['label_50']
#         X = train_df.drop(drop_col, axis=1)
#         y = train_df['label_50']
        
#         # 欠損値の処理
#         train_df = train_df.dropna(subset=['label_50'])
#         test_df = test_df.dropna(subset=['label_50'])

#         # トレーニングデータとバリデーションデータの分割
#         x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False)

#         # データの標準化
#         scaler = StandardScaler()
#         x_train = scaler.fit_transform(x_train)
#         x_valid = scaler.transform(x_valid)
#         x_test = scaler.transform(x_test)

#         # ハイパーパラメータグリッドの設定
#         param_grid = {
#             'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
#             'learning_rate_init': [0.001, 0.0005, 0.0001],
#             'activation': ['relu'],
#             'solver': ['adam'],
#             'early_stopping': [True],
#             'n_iter_no_change': [200],
#             'max_iter': [1000],
#         }

#         # MLPモデルの作成
#         mlp = MLPClassifier(random_state=RANDOM_STATE)
#         grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=10, scoring='f1', n_jobs=-1, verbose=1)
        
#         # モデルの学習
#         grid_search.fit(x_train, y_train)
        
#         # 最良モデルの選択
#         best_model = grid_search.best_estimator_

#         # 損失曲線の記録
#         loss_history.append(best_model.loss_curve_)

#         # テストデータの予測
#         y_proba = best_model.predict_proba(x_test)[:, 1]
#         y_pred = calculate_label(y_proba[0])
#         pred_list.append([test_df['Date'].to_list()[0], y_pred, y_proba[0]])
#         count_pred_1 += y_pred
        
#         gc.collect()

#     # 予測結果をデータフレームにまとめる
#     df_pred = pd.DataFrame(pred_list, columns=['Date', 'pred', 'proba'])
#     _df = df['label_50'].iloc[(N_train):]
#     _df.reset_index(drop=True, inplace=True)
#     df_pred.reset_index(drop=True, inplace=True)
#     df_result = pd.concat([df_pred, _df], axis=1)
#     df_result = df_result.dropna()

#     # モデル結果の保存
#     df_result.to_csv(f'data/result/{vec_name}_mlp_result_50.csv', index=None, encoding='utf-8')

#     # パラメータごとの精度を確認
#     cv_results = pd.DataFrame(grid_search.cv_results_)
#     print(cv_results[['param_hidden_layer_sizes', 'param_learning_rate_init', 'mean_test_score', 'rank_test_score']])
    
#     # 分類レポートの表示
#     print(classification_report(df_result['label_50'], df_result['pred']))

#     # 予測の1の数の確認
#     print("count_pred_1:", count_pred_1)

#     # エポック数の確認
#     print(f"Best model's epoch count: {best_model.n_iter_}")

#     # 損失曲線の描画
#     plt.figure(figsize=(10, 6))
#     plt.plot(best_model.loss_curve_, label='Loss Curve (Best Model)')
#     plt.title('Loss Curve during Training (Best Model)')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()

# if __name__ == "__main__":
#     main()

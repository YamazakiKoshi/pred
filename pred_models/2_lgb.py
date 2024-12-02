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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import lightgbm as lgb  # 修正箇所
from sklearn.model_selection import TimeSeriesSplit
from optuna.integration import LightGBMPruningCallback
import optuna
from sklearn.metrics import log_loss
from sklearn.inspection import permutation_importance

# # ROC曲線を描き，AUCを算出
# def draw_ROC(true_data, proba_data, model_name):
#     fpr, tpr, thresholds = roc_curve(true_data, proba_data)
#     plt.figure(figsize=(9, 6))
#     plt.plot(fpr, tpr, label='roc curve(AUC = %0.3f)' % auc(fpr, tpr))
#     plt.plot([0, 0, 1], [0, 1, 1], linestyle='dashed', label='ideal line')
#     plt.plot([0, 1], [0, 1], linestyle='dashdot', label='random prediction')
#     plt.legend()
#     plt.xlabel('false positive rate(FPR)')
#     plt.ylabel('true positive rate(TPR)')
#     plt.savefig(f'{model_name}_auc.png')  # 適当に指定しておくこと

# # PR曲線を描き，APを算出
# def draw_PR(true_data, proba_data, model_name):
#     precision, recall, thresholds = precision_recall_curve(true_data, proba_data)
#     plt.figure(figsize=(9, 6))
#     plt.plot(recall, precision, label='precision_recall_curve (AP= %0.3f)' % auc(recall, precision))
#     plt.plot([0, 1], [1, 1], linestyle='dashed', label='ideal line')
#     plt.legend()
#     plt.xlabel('recall')
#     plt.ylabel('precision')
#     plt.savefig(f'{model_name}_pr.png')

# # CM曲線を描き，CMを算出
# def draw_CM(true_data, proba_data, model_name):
#     cm = confusion_matrix(true_data, proba_data)
#     plt.figure(figsize=(9, 6))
#     sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
#     title = 'overall accuracy:' + str(accuracy_score(true_data, proba_data))
#     plt.title(title)
#     plt.ylabel('True label')
#     plt.xlabel('False label')
#     plt.savefig(f'{model_name}_cm.png')

def down_sample(X, y, rate, random_sampling=True):
    s = y.value_counts().sum()
    t = y.value_counts()[1]
    num = s - (t // (rate / 100))

    if num<0:
        num=0
    if random_sampling:
        index_list = y[y == 0].sample(n=int(num), random_state=42).index
    else:
        index_list = y[y == 0].iloc[:int(num)].index
    _y = y.drop(index_list).reset_index(drop=True)
    _X = X.drop(index_list).reset_index(drop=True)
    return _X, _y

def calculate_label(proba):
    return 1 if proba >= 0.5 else 0

def calc_shap(model, x_test):
    shap = model.predict(x_test, pred_contrib=True)
    shap = shap.tolist()[0]
    return shap

# メイン処理
def main():
    now = datetime.datetime.now().strftime("%Y%m%d")

    input_path = 'data/features_news/feature_lda_128_1212-2212.csv'
    vec_name = 'historical'
    pre = 50
    df = pd.read_csv(input_path)  # 言語モデル
    # df = df.iloc[5:, :]  # 最初の5日と最後の5日を除去
    df = df.reset_index(drop=True)
    N_train = df[(df['Date'] >= 20160104) & (df['Date'] <= 20161230)].shape[0]
    print(N_train)
    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    drop_col = ["Date", "label"]
    pred_list = []
    count_pred_1 = 0

    for i in tqdm(range(df.shape[0] - (N_train + 5))):
        train_df = df.iloc[i:i + N_train]  # downsamplingあり
        test_df = df.iloc[N_train + i: N_train + i + 1]

        train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        x_test = test_df.drop(drop_col, axis=1)
        y_test = test_df['label']
        X = train_df.drop(drop_col, axis=1)
        y = train_df['label']

        
        x_train, y_train = down_sample(X, y, pre)  # downsamplingあり

        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False)

        x_valid = x_valid.iloc[6:, ]
        y_valid = y_valid.iloc[6:, ]
        
        # LightGBMのフォーマットに変換
        trains = lgb.Dataset(x_train, y_train)
        valids = lgb.Dataset(x_valid, y_valid, reference=trains)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'early_stopping_rounds': 100  # 早期停止のラウンド数を設定
            }
        # lgb.train() 関数の呼び出し時に params を渡す
        model = lgb.train(
            params,
            trains,
            num_boost_round=1000,
            valid_sets=valids
            )



        best_params = model.params
        y_proba = model.predict(x_test, num_iteration=model.best_iteration)
        y_pred = calculate_label(y_proba)
        pred_list.append([test_df['Date'].to_list()[0], y_pred, y_proba[0]])
        count_pred_1 += y_pred

        """# SHAP値の算出
        #pred = model.predict_proba(x_test)[:,1]
        shap = calc_shap(model, x_test)
        date_list.append([test_df['date'].to_list()[0]])
        shap_list.append(shap)"""
        gc.collect()

    df_pred = pd.DataFrame(pred_list, columns=['Date', 'pred', 'proba'])
    df_date = df_pred['Date']
    _df = df['label'].iloc[(N_train + 5):]
    _df.reset_index(drop=True, inplace=True)
    df_pred.reset_index(drop=True, inplace=True)
    df_result = pd.concat([df_pred, _df], axis=1)

    # モデル結果の保存
    df_result.to_csv('results2.csv', index=None, encoding='utf-8')

    # 結果を表示
    print(classification_report(df_result['label'], df_result['pred']))

    # draw_CM(df_result['label'], df_result['pred'], f'{vec_name}_lgb_down{pre}')
    # draw_ROC(df_result['label'], df_result['proba'], f'{vec_name}_lgb_down{pre}')
    # draw_PR(df_result['label'], df_result['proba'], f'{vec_name}_lgb_down{pre}')
    print("count_pred_1")
    print(count_pred_1)

if __name__ == "__main__":
    main()


# -*- coding: utf-8 -*-
#使用しているモジュール（Light_GBM）
from cProfile import label
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
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import optuna.integration.lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from optuna.integration import LightGBMPruningCallback
import optuna
from sklearn.metrics import log_loss
from sklearn.inspection import permutation_importance
#import shap

"""
#ROC曲線を描き，AUCを算出
def draw_ROC(true_data, proba_data, model_name):
    fpr, tpr, thresholds = roc_curve(true_data, proba_data)
    plt.figure(figsize=(9, 6))
    plt.plot(fpr, tpr, label='roc curve(AUC = %0.3f)' % auc(fpr, tpr))
    plt.plot([0, 0, 1],[0, 1, 1], linestyle='dashed', label = 'ideal line')
    plt.plot([0,1], [0,1], linestyle='dashdot', label='random presiction')
    plt.legend()
    plt.xlabel('false positive rate(FPR)')
    plt.ylabel('true positive rate(TPR)')
    #plt.show()
    plt.savefig('{model_name}_auc.png')#適当に指定しておくこと

#PR曲線を描き，APを算出
def draw_PR(true_data, proba_data, model_name):
    precision, recall, thresholds = precision_recall_curve(true_data, proba_data)
    plt.figure(figsize=(9, 6))
    plt.plot(recall, precision, label='precision_recall_curve (AP= %0.3f)' % auc(recall, precision))
    plt.plot([0,1], [1,1], linestyle='dashed', label='ideal line')
    plt.legend()
    plt.xlabel('recall')
    plt.figure(figsize=(9,6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    title = 'overall accuracy:' + str(accuracy_score(true_data, proba_data))
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('False label')
    #plot.show()
    plt.savefig('{model_name}_cm.png')

def predict_lgb(X_test, model):
    ypred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return ypred, y_proba
"""

def down_sample(X, y, rate, random_sampling=True):
    s = y.value_counts().sum()
    t = y.value_counts()[1]
    num = s - (t // (rate / 100))
    if random_sampling:
        index_list = y[y==0].sample(n=int(num), random_state=42).index
    else:
        index_list = y[y==0].iloc[:int(num)].index
    _y = y.drop(index_list).reset_index(drop=True)
    _X = X.drop(index_list).reset_index(drop=True)
    return _X, _y

def calculate_label(proba):
    if proba < 0.5:
        label = 0
    else:
        label = 1
    return label

def calc_shap(model, x_test):
    shap = model.predict(x_test, pred_contrib=True) 
    shap = shap.tolist()[0]
    return shap


def main():
    now = datetime.datetime.now().strftime("%Y%m%d")
    
    input_path = 'data/features_news/merge_news_bert.csv'
    vec_name = 'bert_128'
    pre = 50
    
    df = pd.read_csv(input_path) #言語モデル
    df = df.iloc[5:,:] #最初の5日と最後の5日を除去
    print(df.columns)
    print(len(df.columns))
    df = df.reset_index(drop=True)
    # N_train = df[(df['Date']>=20160104)&(df['Date']<=20161230)].shape[0] #245 #490 #735 #735 #学習期間を固定している．上田さん論文参照

    # RANDOM_STATE = 43
    # TEST_SIZE = 0.3
    
    # drop_col=["Date", "label"]
    # pred_list = [] 
    # count_pred_1 = 0
    # importance = []# 特徴量の重要度を格納する変数 54 = 説明変数（特徴量）の数

    # date_list = [] #日付のリスト
    # shap_list = [] #shap値を格納するリスト

    # N_train: 6割分のデータ, N_val: 2割分のデータ
    N = df.shape[0]  # 全データの長さ
    N_train = int(N * 0.6)
    N_val = int(N * 0.2)
        
    RANDOM_STATE = 43
    TEST_SIZE = 0.3
        
    drop_col = ["Date", "label"]
    pred_list = [] 
    count_pred_1 = 0
    loss_history = []  # Lossの記録リスト

    # ループで各テストデータごとにモデルの訓練と予測を行う
    for i in tqdm(range(df.shape[0] - (N_train + N_val))):
        train_df = df.iloc[:i + N_train]  # トレーニングデータ (iを使って少しずつ広げる)
        val_df = df.iloc[N_train:N_train + N_val]  # バリデーションデータ
        test_df = df.iloc[N_train + N_val + i:N_train + N_val + i + 1]  # テストデータ

        # NaNを含む行を削除
        train_df = train_df.dropna(subset=['label'])
        val_df = val_df.dropna(subset=['label'])
        test_df = test_df.dropna(subset=['label'])

        # トレーニング、バリデーション、テストデータのX, yを作成
        x_train = train_df.drop(drop_col, axis=1)
        y_train = train_df['label']
        
        x_valid = val_df.drop(drop_col, axis=1)
        y_valid = val_df['label']
        
        x_test = test_df.drop(drop_col, axis=1)
        y_test = test_df['label']

        # ダウンサンプリング
        x_train, y_train = down_sample(x_train, y_train, pre)

        # MLP用の標準化
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            #'device':'gpu',

        }
        
        best_params, history = {}, []
        # learning_lightgbm
        verbose_eval=0
        model = lgb.train(params,
                          trains,
                          num_boost_round= 1000, #1000,
                          valid_sets=valids,
                          callbacks=[lgb.early_stopping(stopping_rounds=10,
                                verbose=True),#early_stopping用コールバック関数
                            lgb.log_evaluation(verbose_eval)]#コマンドライン出力用コールバック関数、
                          #max_bin = 100
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

    df_pred = pd.DataFrame(pred_list, columns=['Date', 'pred', 'ploba'])
    df_date = df_pred['Date']
    _df = df['label'].iloc[(N_train+5):]
    _df.reset_index(drop=True, inplace=True)
    df_pred.reset_index(drop=True, inplace=True) 
    df_result = pd.concat([df_pred, _df], axis=1)
    df_result=df_result.dropna()
    
    df_result['Date'] = pd.to_datetime(df_result['Date']).dt.strftime('%Y%m%d')

    
    # モデル結果の保存
    df_result.to_csv(f'data/result/{vec_name}_lgm_result.csv', index=None, encoding='utf-8')

    """# shap値を格納
    x_list = df.drop(drop_col, axis=1).columns.tolist()
    x_list.append('expected_values')
    df_shap = pd.DataFrame(shap_list, columns=x_list)
    df_shap = pd.concat([df_date, df_shap], axis=1)
    df_shap.to_csv(f'../../data/result/shap_{vec_name}_lgm_down{pre}.csv', index=None, encoding='utf-8')"""

    # 結果を表示
    df_result2 = df_result.iloc[735:,]
    print(classification_report(df_result2['label'], df_result2['pred']))

    #draw_CM(df_result['label'], df_result['pred'], f'{vecter_name}_lgb_down{pre}')
    #draw_ROC(df_result['label'], df_result['proba'], f'{vecter_name}_lgb_down{pre}')
    #draw_PR(df_result['label'], df_result['proba'], f'{vecter_name}_lgb_down{pre}')

    print(df_result2)
    print("count_pred_1")
    print(count_pred_1)

if __name__ == "__main__":

    main()

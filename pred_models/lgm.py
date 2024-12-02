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
#ROCæ›²ç·šã‚’æãï¼ŒAUCã‚’ç®—å‡º
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
    plt.savefig('{model_name}_auc.png')#é©å½“ã«æŒ‡å®šã—ã¦ãŠãã“ã¨

#PRæ›²ç·šã‚’æãï¼ŒAPã‚’ç®—å‡º
def draw_PR(true_data, proba_data, model_name):
    precision, recall, thresholds = precision_recall_curve(true_data, proba_data)
    plt.figure(figsize=(9, 6))
    plt.plot(recall, precision, label='precision_recall_curve (AP= %0.3f)' % auc(recall, precision))
    plt.plot([0,1], [1,1], linestyle='dashed', label='ideal line')
    plt.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    #plt.show()
    plt.savefig('{model_name}_pr.png')

# CMæ›²ç·šã‚’æãï¼ŒCMã‚’ç®—å‡º
def draw_CM(true_data, proba_data, model_name):
    cm = confusion_matrix(true_data, proba_data)
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
    
    df = pd.read_csv(input_path) #è¨€èªžãƒ¢ãƒ‡ãƒ«
    df = df.iloc[5:,:] #æœ€åˆã®5æ—¥ã¨æœ€å¾Œã®5æ—¥ã‚’é™¤åŽ»
    print(df.columns)
    print(len(df.columns))
    df = df.reset_index(drop=True)
    N_train = 735 #245 #490 #735 #735 #å­¦ç¿’æœŸé–“ã‚’å›ºå®šã—ã¦ã„ã‚‹ï¼Žä¸Šç”°ã•ã‚“è«–æ–‡å‚ç…§

    RANDOM_STATE = 43
    TEST_SIZE = 0.3
    
    drop_col=["Date", "label"]
    pred_list = [] 
    count_pred_1 = 0
    importance = []# ç‰¹å¾´é‡ã®é‡è¦åº¦ã‚’æ ¼ç´ã™ã‚‹å¤‰æ•° 54 = èª¬æ˜Žå¤‰æ•°ï¼ˆç‰¹å¾´é‡ï¼‰ã®æ•°

    date_list = [] #æ—¥ä»˜ã®ãƒªã‚¹ãƒˆ
    shap_list = [] #shapå€¤ã‚’æ ¼ç´ã™ã‚‹ãƒªã‚¹ãƒˆ

    for i in tqdm(range(df.shape[0]-(N_train +5))):
        #train_df = df.iloc[i:i+N_train] #downsampling ã‚’è¡Œã†å ´åˆã¯ï¼Œ[:i+N_train]
        train_df = df.iloc[:i+N_train] #downsamplingã‚ã‚Š
        test_df = df.iloc[i+N_train+5:i+N_train+6]
        train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        x_test = test_df.drop(drop_col, axis=1)

        y_test = test_df['label']
        X = train_df.drop(drop_col, axis=1)
        y = train_df['label']
        
        #x_train, y_train = X, y #--- downsamplingãªã—ã®å ´åˆã¯ã“ã£ã¡ã‚’é¸æŠž
        x_train, y_train = down_sample(X,y,pre) #---- downsamplingã‚ã‚Šã®å ´åˆã¯ã“ã£ã¡ã‚’é¸æŠž
        
        x_train, x_valid, y_train, y_valid = train_test_split(x_train,
                                                              y_train,
                                                              test_size=TEST_SIZE,
                                                              random_state=RANDOM_STATE,
                                                              shuffle=False)
        
        x_valid = x_valid.iloc[6:,]
        y_valid = y_valid.iloc[6:,]
        
        #transform LightGBM format
        trains = lgb.Dataset(x_train, y_train)
        valids = lgb.Dataset(x_valid, y_valid, reference=trains)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            #'device':'gpu',

        }
        
        best_params, history = {}, []
        # =earning_lightgbm
        verbose_eval = 0  # ã“ã®æ•°å­—ã‚’1ã«ã™ã‚‹ã¨å­¦ç¿’æ™‚ã®ã‚¹ã‚³ã‚¢æŽ¨ç§»ãŒã‚³ãƒžãƒ³ãƒ‰ãƒ©ã‚¤ãƒ³è¡¨ç¤ºã•ã‚Œã‚‹
        model = lgb.train(params,
                          trains,
                          num_boost_round= 1000, #1000,
                          valid_sets=valids,
                          callbacks=[lgb.early_stopping(stopping_rounds=10, 
                                verbose=True), # early_stoppingç”¨ã‚³ãƒ¼ãƒ«ãƒãƒƒã‚¯é–¢æ•°
                           lgb.log_evaluation(verbose_eval)] # ã‚³ãƒžãƒ³ãƒ‰ãƒ©ã‚¤ãƒ³å‡ºåŠ›ç”¨ã‚³ãƒ¼ãƒ«ãƒãƒƒã‚¯é–¢æ•°, #100, #ãƒ†ã‚¹ãƒˆã®ãŸã‚è¨ˆç®—ã‚³ã‚¹ãƒˆã‚’ä¸‹ã’
                          #max_bin = 100
                          )

        best_params = model.params
        y_proba = model.predict(x_test, num_iteration=model.best_iteration)
        y_pred = calculate_label(y_proba)
        pred_list.append([test_df['Date'].to_list()[0], y_pred, y_proba[0]])
        count_pred_1 += y_pred

        """# SHAPå€¤ã®ç®—å‡º
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
    
    # ãƒ¢ãƒ‡ãƒ«çµæžœã®ä¿å­˜
    df_result.to_csv(f'data/result/{vec_name}_lgm_result.csv', index=None, encoding='utf-8')

    """# shapå€¤ã‚’æ ¼ç´
    x_list = df.drop(drop_col, axis=1).columns.tolist()
    x_list.append('expected_values')
    df_shap = pd.DataFrame(shap_list, columns=x_list)
    df_shap = pd.concat([df_date, df_shap], axis=1)
    df_shap.to_csv(f'../../data/result/shap_{vec_name}_lgm_down{pre}.csv', index=None, encoding='utf-8')"""

    # çµæžœã‚’è¡¨ç¤º
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
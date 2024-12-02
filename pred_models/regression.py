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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

def main():
    now = datetime.datetime.now().strftime("%Y%m%d")

    input_path = 'data/features_VI/merge_news_bert_regression.csv'
    vec_name = 'bert_128'

    df = pd.read_csv(input_path)
    df = df.reset_index(drop=True)
    N_train = df[(df['Date'] >= 20160104) & (df['Date'] <= 20161230)].shape[0]

    RANDOM_STATE = 43
    TEST_SIZE = 0.3

    drop_col = ["Date", "P/L"]  # 回帰用の目的変数 "P/L" を指定
    results = []
    loss_history = []  # 損失曲線の記録

    # 外れ値判定用の閾値設定（ここでは±3σを使用）
    df['is_outlier'] = (np.abs(df['P/L'] - df['P/L'].mean()) > 3 * df['P/L'].std()).astype(int)

    for i in tqdm(range(df.shape[0] - (N_train))):
        train_df = df.iloc[:i + N_train]
        test_df = df.iloc[N_train + i: N_train + i + 1]

        train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        x_test = test_df.drop(drop_col, axis=1)
        y_test = test_df['P/L']
        X = train_df.drop(drop_col, axis=1)
        y = train_df['P/L']

        train_df = train_df.dropna(subset=['P/L'])
        test_df = test_df.dropna(subset=['P/L'])

        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)

        # 外れ値をデータ重複で重み付け
        augmented_x_train = []
        augmented_y_train = []

        for idx in range(len(x_train)):
            augmented_x_train.append(x_train[idx])
            augmented_y_train.append(y_train.iloc[idx])

            # 外れ値の場合はデータを重複
            if train_df['is_outlier'].iloc[idx]:
                augmented_x_train.append(x_train[idx])  # 重複データ
                augmented_y_train.append(y_train.iloc[idx])

        # 拡張データを変換
        x_train_aug = np.array(augmented_x_train)
        y_train_aug = np.array(augmented_y_train)

        # モデル構築とハイパーパラメータ探索
        param_grid = {
            'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
            'learning_rate_init': [0.001, 0.0005, 0.0001],
            'activation': ['relu'],
            'solver': ['adam'],
            'early_stopping': [True],
            'max_iter': [1000],
        }

        mlp = MLPRegressor(random_state=RANDOM_STATE)
        grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
        grid_search.fit(x_train_aug, y_train_aug)  # 重み付け後のデータで学習

        best_model = grid_search.best_estimator_

        # テストデータ予測
        y_pred = best_model.predict(x_test)[0]
        results.append([test_df['Date'].to_list()[0], y_test.values[0], y_pred])

        gc.collect()

    df_results = pd.DataFrame(results, columns=['Date', 'Actual', 'Predicted'])
    df_results = df_results.dropna()

    # モデル結果の保存
    df_results.to_csv(f'data/result/{vec_name}_mlp_regression_result.csv', index=None, encoding='utf-8')

    # 統計情報の出力
    r2 = r2_score(df_results['Actual'], df_results['Predicted'])
    mae = mean_absolute_error(df_results['Actual'], df_results['Predicted'])
    mse = mean_squared_error(df_results['Actual'], df_results['Predicted'])
    mape = mean_absolute_percentage_error(df_results['Actual'], df_results['Predicted'])

    print("R2 Score:", r2)
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Percentage Error:", mape)

    # 実測値と予測値のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['Date'], df_results['Actual'], label='Actual', color='blue')
    plt.plot(df_results['Date'], df_results['Predicted'], label='Predicted', color='red')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Date')
    plt.ylabel('P/L')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 損失曲線の描画
    plt.figure(figsize=(10, 6))
    for idx, loss in enumerate(loss_history):
        plt.plot(loss, label=f'Fold {idx+1}')
    plt.title('Loss Curve during Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()






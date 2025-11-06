# Boston（代替：California）住宅価格 × CART 回帰アプリ / Streamlit
# - 先頭行の表示（既定50）
# - 2変数を選んで回帰の「決定境界」（予測ヒートマップ）を可視化（他特徴量は中央値で固定）
# - 決定木（回帰木）のハイパーパラメータ調整、評価指標（R2 / RMSE / MAE）、木の図、重要度
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing  # Bostonは非推奨のためCaliforniaで代替

# ---------------------------------------------------------------
# 画面設定
# ---------------------------------------------------------------
st.set_page_config(page_title="住宅価格 × CART（回帰木）", layout="wide")

# ---------------------------------------------------------------
# 0) データセットの説明（大学生向け）
# ---------------------------------------------------------------
st.title("🏠 住宅価格データ × CART（回帰木）")
st.markdown(
    """
    ### この教材で使うデータ
    よく知られた **Boston（ボストン）の住宅価格** データセットは、現在の scikit-learn では
    **倫理的配慮・データ品質**の観点から **提供が終了** しています。  
    本教材では内容が近い **California Housing**（米カリフォルニア州・18940地区）を使います。

    **目的変数**：地区の **中央値住宅価格（×100,000 USD）**  
    **説明変数（地区属性）**：
    - `MedInc`：世帯中央値所得（$10,000単位）
    - `HouseAge`：住宅の築年数（中央値）
    - `AveRooms`：平均部屋数（世帯あたり）
    - `AveBedrms`：平均寝室数（世帯あたり）
    - `Population`：人口
    - `AveOccup`：平均居住者数（世帯あたり）
    - `Latitude`：緯度
    - `Longitude`：経度

    これらの特徴量から **CART（回帰木）** で住宅価格（中央値）を予測します。
    まずデータの構造を確認し、次に **2つの特徴量** を選んで、他の特徴量を中央値に固定したときの
    **予測ヒートマップ** を描き、木の分割のされ方を直感的に理解します。
    """
)

# ---------------------------------------------------------------
# 1) データ読み込み
# ---------------------------------------------------------------
cal = fetch_california_housing(as_frame=True)
X_full = cal.data.copy()
y = cal.target.copy()  # 住宅価格（中央値, 単位は100,000 USD）
feature_names = list(X_full.columns)

# ---------------------------------------------------------------
# 2) サイドバー：設定
# ---------------------------------------------------------------
st.sidebar.header("⚙️ 学習設定 / Controls")
show_rows = st.sidebar.number_input("表示行数 / Rows to show", 10, len(X_full), 50, 10)

# 学習データ割合・乱数
split_ratio = st.sidebar.slider("学習データの割合 / Train size", 0.5, 0.9, 0.8, 0.05)
random_state = st.sidebar.number_input("乱数シード / Random state", 0, 9999, 42, 1)

# 特徴量選択（全特徴量から任意選択）
selected_features = st.sidebar.multiselect("特徴量の選択 / Select features", feature_names, default=feature_names)
if len(selected_features) < 2:
    st.sidebar.warning("少なくとも2つの特徴量を選択してください。")

# 2軸（ヒートマップ用）
axis_opts = selected_features if selected_features else feature_names
x_axis = st.sidebar.selectbox("X軸", axis_opts, index=0)
y_axis_opts = [c for c in axis_opts if c != x_axis] or [c for c in feature_names if c != x_axis]
y_axis = st.sidebar.selectbox("Y軸", y_axis_opts, index=0)

# 回帰木パラメータ
st.sidebar.subheader("🌲 回帰木パラメータ")
criterion = st.sidebar.selectbox("損失 / Criterion", ["squared_error", "friedman_mse", "absolute_error"], index=0)
max_depth = st.sidebar.slider("最大深さ / Max depth", 1, 20, 6, 1)
min_samples_split = st.sidebar.slider("最小分割サンプル数 / min_samples_split", 2, 50, 10, 1)
min_samples_leaf = st.sidebar.slider("最小葉ノード / min_samples_leaf", 1, 50, 5, 1)

cv_k = st.sidebar.slider("交差検証分割数 / CV folds", 2, 10, 5, 1)

# ---------------------------------------------------------------
# 3) 先頭行の確認
# ---------------------------------------------------------------
st.markdown("### 1) データの確認（先頭行）")
st.dataframe(pd.concat([X_full, y.rename("MedHouseVal")], axis=1).head(show_rows), use_container_width=True)
st.caption("スケールや分布の雰囲気をつかみます。")

# ---------------------------------------------------------------
# 4) 学習と評価
# ---------------------------------------------------------------
X = X_full[selected_features].values if selected_features else X_full.values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=split_ratio, random_state=random_state
)

reg = DecisionTreeRegressor(
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=random_state,
)
reg.fit(X_train, y_train)

# 交差検証（R2）
cv_r2 = cross_val_score(reg, X, y, cv=cv_k, scoring="r2")

# テスト評価（古いscikit-learn互換でRMSEを計算）
pred = reg.predict(X_test)
try:
    rmse = mean_squared_error(y_test, pred, squared=False)
except TypeError:
    rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

left, right = st.columns([1.1, 1])

with left:
    st.markdown("### 2) 評価 / Evaluation")
    st.write(f"**R² (test):** {r2:.3f}  |  **RMSE:** {rmse:.3f}  |  **MAE:** {mae:.3f}")
    st.write(f"**CV R² mean:** {cv_r2.mean():.3f}  (± {cv_r2.std():.3f})")

    # 2D ヒートマップ（他特徴量は中央値で固定）
    st.markdown("### 3) 2変数でみる予測ヒートマップ（他変数=中央値）")
    if x_axis and y_axis:
        # 中央値で固定した入力ベクトルを作る
        base = X_full[selected_features].median() if selected_features else X_full.median()
        # グリッド作成
        x_vals = np.linspace(X_full[x_axis].min(), X_full[x_axis].max(), 150)
        y_vals = np.linspace(X_full[y_axis].min(), X_full[y_axis].max(), 150)
        xx, yy = np.meshgrid(x_vals, y_vals)
        grid = pd.DataFrame({col: np.full(xx.size, base[col] if col in base.index else X_full[col].median())
                             for col in (selected_features if selected_features else feature_names)})
        grid[x_axis] = xx.ravel()
        grid[y_axis] = yy.ravel()

        Z = reg.predict(grid.values).reshape(xx.shape)
        fig_hm, ax_hm = plt.subplots(figsize=(7, 5.2), dpi=140)
        hm = ax_hm.contourf(xx, yy, Z, levels=18, alpha=0.9)
        cbar = fig_hm.colorbar(hm, ax=ax_hm, fraction=0.046, pad=0.04)
        cbar.set_label("Predicted MedHouseVal (×100k USD)")
        ax_hm.set_xlabel(x_axis)
        ax_hm.set_ylabel(y_axis)
        ax_hm.set_title("Decision regions (regression heatmap)")
        st.pyplot(fig_hm, use_container_width=True)

with right:
    st.markdown("### 4) 回帰木の図 / Tree plot")
    fig_tr, ax_tr = plt.subplots(figsize=(12, 10), dpi=160)
    plot_tree(
        reg,
        feature_names=(selected_features if selected_features else feature_names),
        filled=False,
        impurity=True,
        rounded=True,
        ax=ax_tr,
    )
    st.pyplot(fig_tr, use_container_width=True)

    st.markdown("### 5) 特徴量の重要度 / Feature importances")
    importances = pd.Series(reg.feature_importances_, index=(selected_features if selected_features else feature_names))
    st.dataframe(importances.sort_values(ascending=False).to_frame("importance"))

# ---------------------------------------------------------------
# 6) 解説（パラメータ）
# ---------------------------------------------------------------
with st.expander("🧮 パラメータ解説：max_depth / min_samples_split / min_samples_leaf"):
    st.markdown(
        """
        - **max_depth**：木の深さの上限。大きくすると複雑→過学習リスク。小さくすると単純→表現力不足の恐れ。
        - **min_samples_split**：ノードを分割するための最小サンプル数。大きめにすると細かい分岐を抑制。
        - **min_samples_leaf**：葉に残す最小サンプル数。極小葉を防ぎ、汎化を安定化。
        """
    )

# ---------------------------------------------------------------
# 7) requirements.txt（コピー用）
# ---------------------------------------------------------------
REQ_TXT = """
streamlit>=1.37
scikit-learn>=1.2  # 1.2未満でも動くようRMSEは後方互換対応済み
pandas>=2.1
numpy>=1.26
matplotlib>=3.8
"""
with st.expander("📦 requirements.txt (コピー用)"):
    st.code(REQ_TXT.strip())

# ---------------------------------------------------------------
# 8) CART 可視化から読み取れること（大学生向け解説）
# ---------------------------------------------------------------
with st.expander("📊 CARTからわかること（California Housing）"):
    st.markdown(
        """
        **1️⃣ 価格を左右する主要因が見えてくる**  
        CART（決定木回帰）を使うと、どの要素が住宅価格に強く影響しているかが「分岐」として目で見てわかります。  
        例：  
        - 「**世帯所得（MedInc）** が高いほど住宅価格も高い」  
        - 「**緯度（Latitude）や経度（Longitude）** から、沿岸部ほど高く、内陸部ほど安い傾向」  
        → 価格を決める **地域性と経済的要因** が明確になります。
        
        **2️⃣ モデルの“考え方”が木構造でわかる**  
        回帰木は「もし～なら～」という **条件分岐の連続** です。  
        例：「もし世帯所得が5.5万ドルより低ければ → 次に築年数を確認 → さらに緯度で分岐」  
        → CARTは **価格を説明する要因を段階的にたどる仕組み** を可視化します。
        
        **3️⃣ 複雑なデータを2次元で直感的に見る**  
        2つの特徴量（例：`MedInc` × `Latitude`）を選び、他を中央値で固定すると、  
        **ヒートマップで「高価格エリア」と「低価格エリア」** が色の濃淡として見えます。  
        → 「所得が高く、海岸に近い地域ほど価格が高い」など、**地理的・社会経済的傾向** を視覚的に確認できます。
        
        **4️⃣ 限界も見えてくる**  
        CARTは直線ではなく“階段状”に分けるため、ヒートマップでも **境界がブロック状** に見えます。  
        → これはCARTの特徴で、**シンプルなルールで複雑な現象を近似している** ことを意味します。
        
        **5️⃣ 学びのポイント**  
        - **CARTは「説明が見えるAI」**：どの要因がどんな順序で効いているかが明確。  
        - **分析の第一歩**として、「どんな変数が価格に影響していそうか」を把握するのに最適。  
        - 将来的には **ランダムフォレスト** や **勾配ブースティング** などへ発展可能。
        """
    )

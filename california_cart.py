# Boston（代替：California）住宅価格 × CART 回帰アプリ / Streamlit
# - 先頭行の表示（既定50）
# - 2変数を選んで回帰の「決定境界」（予測ヒートマップ）を可視化（他特徴量は中央値で固定）
# - 決定木（回帰木）のハイパーパラメータ調整、評価指標（R2 / RMSE / MAE）、木の図、重要度
# - 緯度経度×価格の地図（Folium / サークル or HeatMap 切替）
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# scikit-learn 1.2以降では Boston データセットが削除。
# 代替として California Housing データセットを使用。
from sklearn.datasets import fetch_california_housing

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
y_axis_opts = [c for c in axis_opts if c != x_axis]
y_axis = st.sidebar.selectbox("Y軸", y_axis_opts, index=0 if y_axis_opts else 0)

# 回帰木パラメータ
st.sidebar.subheader("🌲 回帰木パラメータ")
criterion = st.sidebar.selectbox("損失 / Criterion", ["squared_error", "friedman_mse", "absolute_error"], index=0)
max_depth = st.sidebar.slider("最大深さ / Max depth", 1, 20, 6, 1)
min_samples_split = st.sidebar.slider("最小分割サンプル数 / min_samples_split", 2, 50, 10, 1)
min_samples_leaf = st.sidebar.slider("最小葉ノード / min_samples_leaf", 1, 50, 5, 1)

cv_k = st.sidebar.slider("交差検証分割数 / CV folds", 2, 10, 5, 1)

# ---------------------------------------------------------------
# 🗺️ 地図で可視化（“データセット表示の上”に配置）
# ---------------------------------------------------------------
with st.expander("🗺️ 地図で可視化（緯度経度 × 住宅価格）"):
    st.markdown(
        """
        California Housing の各地区を **緯度・経度** に配置し、色で **住宅価格（MedHouseVal, ×100k USD）** を表します。  
        *Folium = インタラクティブ地図（サークル / HeatMap 切替） / Matplotlib = 静止画プロット*
        """
    )

    # 地図表示用データ
    df_map = X_full.copy()
    df_map["MedHouseVal"] = y

    # 表示点数を間引き（インタラクティブ時の負荷軽減）
    max_show = st.slider(
        "表示点数（サンプリング）", min_value=1000, max_value=len(df_map), value=min(5000, len(df_map)), step=1000
    )
    # random_state はサイドバー入力を流用
    df_show = df_map.sample(max_show, random_state=int(random_state) if "random_state" in locals() else 42)

    view = st.radio("表示方法", ["Folium（インタラクティブ）", "Matplotlib（静止画）"], index=0, horizontal=True)

    if view.startswith("Folium"):
        import folium
        from streamlit_folium import st_folium
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        from folium.plugins import HeatMap

        # ベースマップ（平均位置へ）
        m = folium.Map(
            location=[float(df_map["Latitude"].mean()), float(df_map["Longitude"].mean())],
            zoom_start=6,
            tiles="CartoDB positron",
        )

        # 表示レイヤー切替
        layer_mode = st.radio("レイヤー種別", ["サークルマーカー", "HeatMap（密度重み付き）"], index=0, horizontal=True)

        vmin, vmax = float(df_map["MedHouseVal"].min()), float(df_map["MedHouseVal"].max())

        if layer_mode == "サークルマーカー":
            # カラーマップ設定
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap("viridis")

            # サークルマーカー（価格で着色）
            for _, r in df_show.iterrows():
                color = colors.to_hex(cmap(norm(float(r["MedHouseVal"]))))
                folium.CircleMarker(
                    location=[float(r["Latitude"]), float(r["Longitude"])],
                    radius=3,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.85,
                    popup=f"MedHouseVal: {r['MedHouseVal']:.2f}",
                ).add_to(m)

            st.caption(f"色スケール: {vmin:.2f} 〜 {vmax:.2f} (×100k USD)")

        else:
            # HeatMap パラメータ
            radius = st.slider("HeatMap: radius（ぼかし半径）", 3, 30, 12, 1)
            blur   = st.slider("HeatMap: blur（ボケ具合）",  3, 30, 18, 1)
            max_z  = st.slider("HeatMap: max_zoom", 1, 18, 13, 1)

            # 重み（価格）を 0〜1 に正規化してヒートマップへ
            denom = (vmax - vmin) if (vmax - vmin) > 0 else 1.0
            heat_data = [
                [float(r["Latitude"]), float(r["Longitude"]), (float(r["MedHouseVal"]) - vmin) / denom]
                for _, r in df_show.iterrows()
            ]

            HeatMap(
                heat_data,
                radius=radius,
                blur=blur,
                max_zoom=max_z,
                min_opacity=0.2,
                max_val=1.0,  # 重みは0-1スケール
            ).add_to(m)

            st.caption("HeatMapの重み：MedHouseVal を 0〜1 に正規化（高価格ほど高強度）")

        # 地図表示
        st_folium(m, height=600, use_container_width=True)

    else:
        # 静止画：経度をX、緯度をY
        fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
        sc = ax.scatter(
            df_show["Longitude"], df_show["Latitude"],
            c=df_show["MedHouseVal"], s=8
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("MedHouseVal (×100k USD)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("California Housing: Price map")
        st.pyplot(fig, use_container_width=True)

# ---------------------------------------------------------------
# 3) 先頭行の確認（※地図セクションの“下”に移動）
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

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.evacuation_simulator import (
    DATA_DIR,
    OUTPUT_DIR,
    TERRAIN_SCENARIOS,
    analyze,
    get_terrain_scenario,
    save_plots,
    simulate_movements,
)


APP_ROOT = Path(__file__).resolve().parent


st.set_page_config(
    page_title="避難行動分析AI",
    page_icon="",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #475569;
    }
    .section-note {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.65;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def route_summary(features: pd.DataFrame) -> pd.DataFrame:
    summary = (
        features.groupby("route_id")
        .agg(
            evacuees=("user_id", "count"),
            avg_time_min=("evacuation_time_min", "mean"),
            avg_congestion=("route_congestion", "mean"),
            delay_risk_count=("delay_risk", "sum"),
        )
        .sort_values(["evacuees", "avg_time_min"], ascending=False)
        .reset_index()
    )
    summary.insert(0, "rank", [f"R{i + 1}" for i in range(len(summary))])
    return summary


@st.cache_data(show_spinner=False)
def run_analysis(
    n_users: int, scenario_key: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    movements, features = simulate_movements(n_users=n_users, scenario_key=scenario_key)
    features, pca_df, result = analyze(features)
    prediction_df = result["prediction_df"]
    metrics = result["metrics"]

    movements.to_csv(DATA_DIR / "evacuation_movements.csv", index=False)
    features.to_csv(DATA_DIR / "evacuee_features.csv", index=False)
    pca_df.to_csv(OUTPUT_DIR / "pca_projection.csv", index=False)
    prediction_df.to_csv(OUTPUT_DIR / "evacuation_time_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    save_plots(movements, features, pca_df, prediction_df)

    return movements, features, pca_df, prediction_df, metrics, route_summary(features)


def dataframe_download(label: str, data: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label=label,
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


with st.sidebar:
    st.header("分析設定")
    scenario_options = {value["label"]: key for key, value in TERRAIN_SCENARIOS.items()}
    scenario_label = st.selectbox("地形タイプ", list(scenario_options.keys()))
    scenario_key = scenario_options[scenario_label]
    scenario = get_terrain_scenario(scenario_key)
    st.caption(scenario["description"])
    n_users = st.slider("避難者数", min_value=80, max_value=600, value=300, step=20)
    st.caption("地形タイプや避難者数を変更すると、仮想都市の移動ログと分析結果を再生成します。")
    run_clicked = st.button("分析を実行", type="primary", use_container_width=True)

    st.divider()
    st.subheader("使用アルゴリズム")
    st.write("K-means")
    st.write("PCA")
    st.write("線形回帰")
    st.write("ニューラルネットワーク")


st.title("避難行動分析AI")
st.markdown(
    """
    <p class="section-note">
    災害時の移動データを想定し、避難遅延リスクと混雑経路を可視化する分析ダッシュボードです。
    地形タイプを切り替えることで、橋、駅前、狭い道路などの条件が避難結果に与える影響を比較できます。
    </p>
    """,
    unsafe_allow_html=True,
)

if run_clicked:
    st.cache_data.clear()

with st.spinner("避難行動データを生成し、AI分析を実行しています..."):
    movements_df, features_df, pca_df, prediction_df, metrics, routes_df = run_analysis(n_users, scenario_key)

delay_count = int(features_df["delay_risk"].sum())
mean_time = float(features_df["evacuation_time_min"].mean())
top_route = routes_df.iloc[0]

kpi_cols = st.columns(4)
kpi_cols[0].metric("地形タイプ", scenario["label"])
kpi_cols[1].metric("平均避難時間", f"{mean_time:.1f}分")
kpi_cols[2].metric("遅延リスク人数", f"{delay_count:,}人")
kpi_cols[3].metric("予測誤差", f"{metrics['linear_regression_mae_min']:.2f}分")

tabs = st.tabs(["概要", "避難経路", "クラスタ分析", "予測モデル", "混雑ランキング", "データ"])

with tabs[0]:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("分析結果サマリー")
        st.info(f"現在の地形タイプ: {scenario['label']}。{scenario['description']}")
        st.dataframe(
            pd.DataFrame(
                [
                    {"指標": "線形回帰 MAE", "値": f"{metrics['linear_regression_mae_min']:.3f}分"},
                    {"指標": "線形回帰 R2", "値": f"{metrics['linear_regression_r2']:.3f}"},
                    {"指標": "NN遅延分類精度", "値": f"{metrics['neural_network_delay_accuracy']:.3f}"},
                    {"指標": "遅延リスクしきい値", "値": f"{metrics['delay_threshold_min']:.1f}分以上"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.subheader("最も混雑した経路")
        st.metric("避難者数", f"{int(top_route['evacuees'])}人")
        st.write(top_route["route_id"])
        st.caption(f"平均避難時間: {top_route['avg_time_min']:.1f}分")

with tabs[1]:
    st.subheader("仮想都市上の避難経路")
    st.info("線が太く濃いほど、その経路に避難者が集中しています。丸は住宅エリア、四角は経由地点、三角は避難所です。図中のR1、R2は下のランキング表と対応します。")
    st.image(str(OUTPUT_DIR / "evacuation_map.png"), use_column_width=True)
    st.caption("地形タイプを切り替えると、人が集中する場所や混雑しやすい経路が変わります。")
    st.dataframe(
        routes_df[["rank", "route_id", "evacuees", "avg_time_min"]].head(5),
        use_container_width=True,
        hide_index=True,
    )

with tabs[2]:
    st.subheader("K-means + PCA による避難行動クラスタ")
    st.image(str(OUTPUT_DIR / "pca_clusters.png"), use_column_width=True)
    cluster_counts = features_df["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(name="evacuees")
    st.dataframe(cluster_counts, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("避難完了時間の予測")
    st.image(str(OUTPUT_DIR / "delay_prediction.png"), use_column_width=True)
    st.dataframe(prediction_df.head(20), use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("混雑経路ランキング")
    st.image(str(OUTPUT_DIR / "route_congestion.png"), use_column_width=True)
    st.dataframe(routes_df, use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("生成データ")
    data_tabs = st.tabs(["移動ログ", "分析用特徴量", "PCA", "予測結果"])
    with data_tabs[0]:
        st.dataframe(movements_df.head(200), use_container_width=True, hide_index=True)
        dataframe_download("移動ログCSVをダウンロード", movements_df, "evacuation_movements.csv")
    with data_tabs[1]:
        st.dataframe(features_df, use_container_width=True, hide_index=True)
        dataframe_download("特徴量CSVをダウンロード", features_df, "evacuee_features.csv")
    with data_tabs[2]:
        st.dataframe(pca_df, use_container_width=True, hide_index=True)
        dataframe_download("PCA結果CSVをダウンロード", pca_df, "pca_projection.csv")
    with data_tabs[3]:
        st.dataframe(prediction_df, use_container_width=True, hide_index=True)
        dataframe_download("予測結果CSVをダウンロード", prediction_df, "evacuation_time_predictions.csv")

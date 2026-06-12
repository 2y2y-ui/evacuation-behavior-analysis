from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
RANDOM_SEED = 42


@dataclass(frozen=True)
class Point:
    name: str
    lat: float
    lon: float


HOME_ZONES = [
    Point("north_residential", 35.7040, 139.7050),
    Point("east_residential", 35.6920, 139.7280),
    Point("south_residential", 35.6720, 139.7150),
    Point("west_residential", 35.6860, 139.6900),
    Point("station_area", 35.6895, 139.7005),
    Point("river_side", 35.6785, 139.7310),
]

HUBS = [
    Point("main_road_crossing", 35.6900, 139.7100),
    Point("station_plaza", 35.6885, 139.7030),
    Point("narrow_bridge", 35.6815, 139.7210),
    Point("school_gate", 35.6955, 139.7180),
]

SHELTERS = [
    Point("shelter_school", 35.6975, 139.7210),
    Point("shelter_park", 35.6820, 139.6970),
    Point("shelter_city_hall", 35.6880, 139.7160),
]

TERRAIN_SCENARIOS = {
    "standard": {
        "label": "標準の市街地",
        "plot_label": "Standard Urban Area",
        "description": "住宅地、駅前、川沿いがほどよく混在する基本ケースです。",
        "zone_weights": [0.18, 0.17, 0.18, 0.16, 0.19, 0.12],
        "route_bias": {},
        "speed_factor": 1.0,
        "congestion_strength": 0.32,
        "delay_shift": 0.0,
    },
    "river_bridge": {
        "label": "川と橋がある地域",
        "plot_label": "River and Bridge Area",
        "description": "川沿いの住民が橋に集中し、橋周辺で混雑しやすいケースです。",
        "zone_weights": [0.12, 0.13, 0.12, 0.12, 0.16, 0.35],
        "route_bias": {"river_side": "narrow_bridge"},
        "speed_factor": 0.92,
        "congestion_strength": 0.46,
        "delay_shift": 2.0,
    },
    "station_crowd": {
        "label": "駅前に人が集中する地域",
        "plot_label": "Crowded Station Area",
        "description": "駅前エリアからの避難者が多く、駅前広場に人が集まりやすいケースです。",
        "zone_weights": [0.10, 0.11, 0.12, 0.10, 0.47, 0.10],
        "route_bias": {"station_area": "station_plaza"},
        "speed_factor": 0.96,
        "congestion_strength": 0.42,
        "delay_shift": 1.0,
    },
    "remote_shelter": {
        "label": "避難所が遠い地域",
        "plot_label": "Remote Shelter Area",
        "description": "南側・西側の住宅地から避難所までの距離が長くなりやすいケースです。",
        "zone_weights": [0.08, 0.10, 0.34, 0.30, 0.10, 0.08],
        "route_bias": {"south_residential": "narrow_bridge", "west_residential": "main_road_crossing"},
        "speed_factor": 0.94,
        "congestion_strength": 0.34,
        "delay_shift": 3.0,
    },
    "narrow_roads": {
        "label": "道路が狭く混雑しやすい地域",
        "plot_label": "Narrow Road Area",
        "description": "複数エリアの避難者が同じ交差点に集中し、移動速度が落ちやすいケースです。",
        "zone_weights": [0.18, 0.18, 0.17, 0.17, 0.18, 0.12],
        "route_bias": {
            "north_residential": "main_road_crossing",
            "east_residential": "main_road_crossing",
            "south_residential": "main_road_crossing",
            "west_residential": "main_road_crossing",
        },
        "speed_factor": 0.86,
        "congestion_strength": 0.55,
        "delay_shift": 1.5,
    },
}


def haversine_km(a: Point, b: Point) -> float:
    radius = 6371.0
    lat1, lon1 = math.radians(a.lat), math.radians(a.lon)
    lat2, lon2 = math.radians(b.lat), math.radians(b.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


def jitter(point: Point, rng: random.Random, meters: float = 180.0) -> Point:
    lat_shift = rng.uniform(-meters, meters) / 111_000
    lon_shift = rng.uniform(-meters, meters) / (111_000 * math.cos(math.radians(point.lat)))
    return Point(point.name, point.lat + lat_shift, point.lon + lon_shift)


def nearest_shelter(origin: Point) -> Point:
    return min(SHELTERS, key=lambda shelter: haversine_km(origin, shelter))


def choose_route(origin: Point, shelter: Point, rng: random.Random, scenario: dict | None = None) -> list[Point]:
    route_bias = (scenario or {}).get("route_bias", {})
    if origin.name in route_bias:
        hub = next(point for point in HUBS if point.name == route_bias[origin.name])
        return [origin, hub, shelter]

    if origin.name == "river_side":
        hub = next(point for point in HUBS if point.name == "narrow_bridge")
    elif origin.name in {"north_residential", "east_residential"}:
        hub = rng.choice([HUBS[0], HUBS[3]])
    elif origin.name == "station_area":
        hub = next(point for point in HUBS if point.name == "station_plaza")
    else:
        hub = rng.choice([HUBS[0], HUBS[1], HUBS[2]])
    return [origin, hub, shelter]


def path_distance_km(path: list[Point]) -> float:
    return sum(haversine_km(path[i], path[i + 1]) for i in range(len(path) - 1))


def interpolate(a: Point, b: Point, ratio: float) -> Point:
    return Point(
        f"{a.name}_to_{b.name}",
        a.lat + (b.lat - a.lat) * ratio,
        a.lon + (b.lon - a.lon) * ratio,
    )


def get_terrain_scenario(scenario_key: str = "standard") -> dict:
    return TERRAIN_SCENARIOS.get(scenario_key, TERRAIN_SCENARIOS["standard"])


def simulate_movements(n_users: int = 300, scenario_key: str = "standard") -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario = get_terrain_scenario(scenario_key)
    rng = random.Random(RANDOM_SEED)
    np_rng = np.random.default_rng(RANDOM_SEED)
    start_time = pd.Timestamp("2026-04-10 08:00")
    movement_rows: list[dict] = []
    feature_rows: list[dict] = []
    route_counts: dict[str, int] = {}

    user_plans = []
    for idx in range(1, n_users + 1):
        zone = rng.choices(
            HOME_ZONES,
            weights=scenario["zone_weights"],
            k=1,
        )[0]
        origin = jitter(zone, rng)
        shelter = nearest_shelter(origin)
        path = choose_route(origin, shelter, rng, scenario)
        route_id = " -> ".join(point.name for point in path)
        route_counts[route_id] = route_counts.get(route_id, 0) + 1
        user_plans.append((idx, zone, origin, shelter, path, route_id))

    max_route_count = max(route_counts.values())
    for idx, zone, origin, shelter, path, route_id in user_plans:
        user_id = f"U{idx:03d}"
        vulnerable = int(rng.random() < 0.22)
        departure_delay = max(0, np_rng.normal(18 + scenario["delay_shift"], 12))
        if vulnerable:
            departure_delay += np_rng.uniform(6, 20)

        base_speed = np_rng.normal(4.2, 0.55)
        if vulnerable:
            base_speed -= np_rng.uniform(0.8, 1.4)

        route_congestion = route_counts[route_id] / max_route_count
        congestion_penalty = 1.0 - scenario["congestion_strength"] * route_congestion
        walking_speed = max(1.4, base_speed * congestion_penalty * scenario["speed_factor"])
        total_distance = path_distance_km(path)
        evac_time = (total_distance / walking_speed) * 60
        evac_time += route_congestion * 12 * (scenario["congestion_strength"] / 0.32) + np_rng.normal(0, 3)
        evac_time = max(5, evac_time)

        depart_at = start_time + pd.Timedelta(minutes=float(departure_delay))
        arrive_at = depart_at + pd.Timedelta(minutes=float(evac_time))

        movement_rows.append(
            {
                "user_id": user_id,
                "timestamp": start_time.strftime("%Y-%m-%d %H:%M"),
                "location": zone.name,
                "latitude": round(origin.lat, 6),
                "longitude": round(origin.lon, 6),
                "action": "stay",
                "speed": 0.0,
                "source": "simulation",
                "route_id": route_id,
                "zone": zone.name,
                "shelter_id": shelter.name,
                "terrain_type": scenario["label"],
                "terrain_plot_label": scenario["plot_label"],
            }
        )

        steps = max(3, int(evac_time // 5))
        for step in range(1, steps):
            ratio = step / steps
            if ratio < 0.5:
                p = interpolate(path[0], path[1], ratio / 0.5)
            else:
                p = interpolate(path[1], path[2], (ratio - 0.5) / 0.5)
            timestamp = depart_at + (arrive_at - depart_at) * ratio
            movement_rows.append(
                {
                    "user_id": user_id,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
                    "location": p.name,
                    "latitude": round(p.lat, 6),
                    "longitude": round(p.lon, 6),
                    "action": "move",
                    "speed": round(walking_speed, 2),
                    "source": "simulation",
                    "route_id": route_id,
                    "zone": zone.name,
                    "shelter_id": shelter.name,
                    "terrain_type": scenario["label"],
                    "terrain_plot_label": scenario["plot_label"],
                }
            )

        movement_rows.append(
            {
                "user_id": user_id,
                "timestamp": arrive_at.strftime("%Y-%m-%d %H:%M"),
                "location": shelter.name,
                "latitude": round(shelter.lat, 6),
                "longitude": round(shelter.lon, 6),
                "action": "stay",
                "speed": 0.0,
                "source": "simulation",
                "route_id": route_id,
                "zone": zone.name,
                "shelter_id": shelter.name,
                "terrain_type": scenario["label"],
                "terrain_plot_label": scenario["plot_label"],
            }
        )

        feature_rows.append(
            {
                "user_id": user_id,
                "zone": zone.name,
                "shelter_id": shelter.name,
                "route_id": route_id,
                "departure_delay_min": round(float(departure_delay), 2),
                "distance_km": round(total_distance, 3),
                "avg_speed_kmh": round(float(walking_speed), 2),
                "route_congestion": round(float(route_congestion), 3),
                "vulnerable": vulnerable,
                "evacuation_time_min": round(float(evac_time), 2),
                "terrain_type": scenario["label"],
                "terrain_plot_label": scenario["plot_label"],
            }
        )

    movements = pd.DataFrame(movement_rows)
    features = pd.DataFrame(feature_rows)
    threshold = features["evacuation_time_min"].quantile(0.70)
    features["delay_risk"] = (features["evacuation_time_min"] >= threshold).astype(int)
    return movements, features


def analyze(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    numeric_cols = [
        "departure_delay_min",
        "distance_km",
        "avg_speed_kmh",
        "route_congestion",
        "vulnerable",
        "evacuation_time_min",
    ]
    x = features[numeric_cols]
    cluster_model = make_pipeline(StandardScaler(), KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=20))
    features = features.copy()
    features["cluster"] = cluster_model.fit_predict(x)

    pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=RANDOM_SEED))
    pca_values = pca_pipe.fit_transform(x)
    pca_df = pd.DataFrame(pca_values, columns=["pc1", "pc2"])
    pca_df["cluster"] = features["cluster"]
    pca_df["delay_risk"] = features["delay_risk"]
    pca_df["user_id"] = features["user_id"]

    model_cols = ["departure_delay_min", "distance_km", "avg_speed_kmh", "route_congestion", "vulnerable"]
    x_train, x_test, y_train, y_test = train_test_split(
        features[model_cols],
        features["evacuation_time_min"],
        test_size=0.25,
        random_state=RANDOM_SEED,
    )
    reg = make_pipeline(StandardScaler(), LinearRegression())
    reg.fit(x_train, y_train)
    pred = reg.predict(x_test)

    clf_x_train, clf_x_test, clf_y_train, clf_y_test = train_test_split(
        features[model_cols],
        features["delay_risk"],
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=features["delay_risk"],
    )
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(12, 6), max_iter=1500, random_state=RANDOM_SEED),
    )
    clf.fit(clf_x_train, clf_y_train)
    clf_pred = clf.predict(clf_x_test)

    prediction_df = pd.DataFrame({"actual": y_test.to_numpy(), "predicted": pred})
    metrics = {
        "linear_regression_mae_min": round(float(mean_absolute_error(y_test, pred)), 3),
        "linear_regression_r2": round(float(r2_score(y_test, pred)), 3),
        "neural_network_delay_accuracy": round(float(accuracy_score(clf_y_test, clf_pred)), 3),
        "n_users": int(len(features)),
        "delay_threshold_min": round(float(features["evacuation_time_min"].quantile(0.70)), 2),
    }
    return features, pca_df, {"prediction_df": prediction_df, "metrics": metrics}


def save_plots(movements: pd.DataFrame, features: pd.DataFrame, pca_df: pd.DataFrame, prediction_df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    terrain_label = movements["terrain_plot_label"].iloc[0] if "terrain_plot_label" in movements else "Standard"

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(pca_df["pc1"], pca_df["pc2"], c=pca_df["cluster"], cmap="tab10", alpha=0.78)
    ax.set_title("Evacuation Pattern Clusters (K-means + PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(*scatter.legend_elements(), title="cluster", loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_clusters.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sample_users = movements["user_id"].drop_duplicates().sample(50, random_state=RANDOM_SEED)
    for _, user_path in movements[movements["user_id"].isin(sample_users)].groupby("user_id"):
        ax.plot(user_path["longitude"], user_path["latitude"], color="#4C78A8", alpha=0.18, linewidth=1)
    ax.scatter([p.lon for p in SHELTERS], [p.lat for p in SHELTERS], marker="^", s=160, color="#E45756", label="shelter")
    ax.scatter([p.lon for p in HOME_ZONES], [p.lat for p in HOME_ZONES], marker="o", s=70, color="#54A24B", label="home zone")
    ax.set_title(f"Simulated Evacuation Routes: {terrain_label}")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "evacuation_map.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(prediction_df["actual"], prediction_df["predicted"], color="#F58518", alpha=0.75)
    max_value = max(prediction_df.max())
    min_value = min(prediction_df.min())
    ax.plot([min_value, max_value], [min_value, max_value], color="#333333", linestyle="--", linewidth=1)
    ax.set_title("Evacuation Time Prediction")
    ax.set_xlabel("actual minutes")
    ax.set_ylabel("predicted minutes")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "delay_prediction.png", dpi=160)
    plt.close(fig)

    route_summary = (
        features.groupby("route_id")
        .agg(users=("user_id", "count"), avg_time_min=("evacuation_time_min", "mean"), congestion=("route_congestion", "mean"))
        .sort_values(["users", "avg_time_min"], ascending=False)
        .head(8)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"R{i + 1}" for i in range(len(route_summary))]
    ax.bar(labels, route_summary["users"], color="#72B7B2")
    ax.set_title("Top Congested Routes")
    ax.set_xlabel("route")
    ax.set_ylabel("number of evacuees")
    for idx, row in route_summary.iterrows():
        ax.text(idx, row["users"] + 0.5, f"{row['avg_time_min']:.1f}m", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "route_congestion.png", dpi=160)
    plt.close(fig)
    route_summary.to_csv(OUTPUT_DIR / "route_congestion_summary.csv", index=False)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    movements, features = simulate_movements()
    features, pca_df, result = analyze(features)
    prediction_df = result["prediction_df"]
    metrics = result["metrics"]

    movements.to_csv(DATA_DIR / "evacuation_movements.csv", index=False)
    features.to_csv(DATA_DIR / "evacuee_features.csv", index=False)
    pca_df.to_csv(OUTPUT_DIR / "pca_projection.csv", index=False)
    prediction_df.to_csv(OUTPUT_DIR / "evacuation_time_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    save_plots(movements, features, pca_df, prediction_df)

    print("Generated evacuation simulation dataset and analysis outputs.")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

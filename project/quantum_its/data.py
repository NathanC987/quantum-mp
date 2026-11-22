"""Classical data stage: load, clean, engineer features, scale to rotation angles, split.

Dataset: Metro Interstate Traffic Volume (UCI id 492, also on Kaggle). Hourly I-94 traffic volume
plus weather-station readings, 2012-2018.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "Metro_Interstate_Traffic_Volume.csv"
UCI_URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"

FEATURES = ["hour", "day_type", "temp", "weather"]          # x = [x1, x2, x3, x4]
FEATURE_LABELS = {
    "hour": "hour of day (cyclic, 0-23)",
    "day_type": "weekend / holiday flag",
    "temp": "temperature (K)",
    "weather": "weather severity index (0-4)",
}

# Weather severity: how strongly a condition is expected to slow traffic.
WEATHER_SEVERITY = {"Clear": 0, "Clouds": 1, "Haze": 2, "Mist": 2, "Fog": 2, "Smoke": 2,
                    "Drizzle": 2, "Rain": 3, "Snow": 4, "Thunderstorm": 4, "Squall": 4}

TEMP_RANGE = (243.0, 311.0)                                   # K, from the cleaned data


def load_raw() -> pd.DataFrame:
    if CSV.exists():
        return pd.read_csv(CSV)
    gz = CSV.with_suffix(".csv.gz")
    if gz.exists():
        return pd.read_csv(gz)
    raise FileNotFoundError(f"Download the dataset from {UCI_URL} and place the CSV at {CSV}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[(df["rain_1h"] < 100) & (df["temp"] > 100)]       # sensor glitches (9831 mm/h rain, 0 K)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.drop_duplicates("date_time")                      # several weather rows per hour -> keep one
    df["hour"] = df["date_time"].dt.hour
    holiday = df["holiday"].fillna("None").ne("None")
    # 'holiday' is only marked on the midnight row: spread it to the whole day
    holiday_days = set(df.loc[holiday, "date_time"].dt.date)
    is_holiday = df["date_time"].dt.date.isin(holiday_days)
    df["day_type"] = ((df["date_time"].dt.dayofweek >= 5) | is_holiday).astype(int)
    df["weather"] = df["weather_main"].map(WEATHER_SEVERITY).fillna(1).astype(int)
    return df.sort_values("date_time").reset_index(drop=True)


def add_label(df: pd.DataFrame, threshold: float | None = None):
    threshold = float(df["traffic_volume"].median()) if threshold is None else threshold
    df = df.copy()
    df["congested"] = (df["traffic_volume"] > threshold).astype(int)
    return df, threshold


def to_angles(hour, day_type, temp, weather) -> np.ndarray:
    """Classical preprocessing -> rotation angles for angle encoding.

    hour     -> 2*pi*h/24   (cyclic: RY(2*pi) = -I is a global phase, so 23:00 sits next to 00:00)
    day_type -> 0 or pi     (a binary flag becomes |0> or |1>, i.e. basis encoding on that qubit)
    temp     -> [0, pi]
    weather  -> [0, pi]
    """
    hour, day_type, temp, weather = map(np.asarray, (hour, day_type, temp, weather))
    t = np.clip((temp - TEMP_RANGE[0]) / (TEMP_RANGE[1] - TEMP_RANGE[0]), 0, 1)
    return np.stack([2 * np.pi * hour / 24, np.pi * day_type, np.pi * t, np.pi * weather / 4],
                    axis=-1).astype(np.float64)


def frame_to_angles(df: pd.DataFrame) -> np.ndarray:
    return to_angles(df["hour"], df["day_type"], df["temp"], df["weather"])


def load_dataset(n_test: int = 2000, seed: int = 0):
    """Chronological split: train on 2012-2016, test on 2017-2018 (no look-ahead leakage)."""
    df = clean(load_raw())
    train_df = df[df["date_time"] < "2017-01-01"]
    test_df = df[df["date_time"] >= "2017-01-01"]
    train_df, threshold = add_label(train_df)                 # threshold from training data only
    test_df, _ = add_label(test_df, threshold)
    test_df = test_df.sample(n=min(n_test, len(test_df)), random_state=seed)
    return dict(
        df=df, threshold=threshold,
        X_pool=frame_to_angles(train_df), y_pool=train_df["congested"].to_numpy(),
        X_test=frame_to_angles(test_df), y_test=test_df["congested"].to_numpy(),
        n_pool=len(train_df), n_test_total=len(df) - len(train_df),
    )


def sample_train(data: dict, n: int, seed: int):
    """Class-balanced random subset of the training pool."""
    rng = np.random.default_rng(seed)
    y = data["y_pool"]
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), n // 2, replace=False) for c in (0, 1)])
    rng.shuffle(idx)
    return data["X_pool"][idx], y[idx]

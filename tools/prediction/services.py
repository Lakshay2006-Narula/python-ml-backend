# tools/prediction/services.py

import os
import json
import math
import uuid
import shutil
import numpy as np
import pandas as pd
from typing import List
import joblib

# Optional geometry + ML
try:
    from shapely import wkt
    from shapely.geometry import Point
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False

from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle

# Optional boosted models
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    import lightgbm as lgb
    LGB_OK = True
except Exception:
    LGB_OK = False


# -------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------
KPI_RANGES = {
    "RSRP": (-140.0, -44.0),
    "RSRQ": (-19.5, -3.0),
    "SINR": (0.0, 30.0),
}


# -------------------------------------------------------------
# BASIC HELPERS
# -------------------------------------------------------------
def clamp_array(arr, kpi_name):
    lo, hi = KPI_RANGES[kpi_name]
    return np.clip(arr, lo, hi)


def normcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    return df


# -------------------------------------------------------------
# LAT/LON STANDARDIZATION FIX (Option B)
# -------------------------------------------------------------
def standardize_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known site_noMl columns:
    - lat_pred -> lat
    - lon_pred -> lon
    No dropping is done here.
    """
    df = df.copy()
    cols = df.columns

    # Standard mapping for site_noMl
    if "lat_pred" in cols and "lon_pred" in cols:
        df = df.rename(columns={"lat_pred": "lat", "lon_pred": "lon"})

    # Standard fallback
    if "latitude" in cols and "longitude" in cols:
        df = df.rename(columns={"latitude": "lat", "longitude": "lon"})

    # Already standardized
    if "lat" in cols and "lon" in cols:
        return df

    # Soft mapping
    mapping = {}
    for c in cols:
        if c in ("lat_deg", "y"):
            mapping[c] = "lat"
        if c in ("lon_deg", "x"):
            mapping[c] = "lon"
    df = df.rename(columns=mapping)

    return df


# -------------------------------------------------------------
# GEO MATH
# -------------------------------------------------------------
def to_rad(df: pd.DataFrame) -> np.ndarray:
    return np.radians(np.c_[df["lat"].values, df["lon"].values])


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))


# -------------------------------------------------------------
# REGR PIPELINE BUILDERS
# -------------------------------------------------------------
def make_regressor():
    if XGB_OK:
        return XGBRegressor(
            n_estimators=900, max_depth=8, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, reg_alpha=0.0,
            tree_method="hist", random_state=42, n_jobs=-1
        )
    if LGB_OK:
        return lgb.LGBMRegressor(
            n_estimators=1400, num_leaves=63,
            learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42
        )
    try:
        return GradientBoostingRegressor(random_state=42)
    except:
        return RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)


def build_preprocess(num_features: List[str], cat_features: List[str]) -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_features),
            ("cat", categorical, cat_features)
        ],
        remainder="drop"
    )


# -------------------------------------------------------------
# FAST MATCH (distance + angle) — Protected from edge cases
# -------------------------------------------------------------
def fast_match(work_site: pd.DataFrame, points_df: pd.DataFrame) -> pd.DataFrame:
    """
    Matches DT/test points to nearest sites.
    Protected against:
    - 1 site only
    - NaN azimuth
    - NaN distance
    """

    if len(work_site) == 0:
        raise RuntimeError("fast_match: no site rows available")

    # Fill missing azimuth
    if "azimuth" in work_site.columns:
        work_site["azimuth"] = pd.to_numeric(work_site["azimuth"], errors="coerce").fillna(0)
    else:
        work_site["azimuth"] = 0

    # At least 1 neighbor
    n_neighbors = min(10, len(work_site))
    if n_neighbors < 1:
        n_neighbors = 1

    # Fit NN model
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="ball_tree",
        metric="haversine"
    )
    nbrs.fit(to_rad(work_site))

    # Predict
    distances, indices = nbrs.kneighbors(to_rad(points_df))
    distances_m = distances * 6371000.0

    best_idx = []
    best_dist = []
    best_bearing = []
    best_delta = []

    for i in range(points_df.shape[0]):
        lat = points_df.iloc[i]["lat"]
        lon = points_df.iloc[i]["lon"]

        idxs = indices[i]
        candidates = work_site.iloc[idxs]

        # Compute bearings
        phi1 = np.radians(candidates["lat"].values)
        phi2 = np.radians(lat)
        dlmb = np.radians(lon - candidates["lon"].values)

        x = np.sin(dlmb) * np.cos(phi2)
        y = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlmb)

        brngs = (np.degrees(np.arctan2(x, y)) + 360) % 360
        deltas = np.abs((candidates["azimuth"].values - brngs + 180) % 360 - 180)

        # Best candidate = smallest delta, then smallest distance
        order = np.lexsort((distances_m[i], deltas))
        j = order[0]

        best_idx.append(idxs[j])
        best_dist.append(float(distances_m[i][j]))
        best_bearing.append(float(brngs[j]))
        best_delta.append(float(deltas[j]))

    sel = work_site.iloc[best_idx].reset_index(drop=True)

    return pd.DataFrame({
        "best_site": sel["SiteName"].astype(str),
        "best_sector": sel["sector"],
        "site_lat": sel["lat"],
        "site_lon": sel["lon"],
        "site_az": sel["azimuth"],
        "dist_m": best_dist,
        "bearing_tx_to_ue": best_bearing,
        "delta_az": best_delta
    })

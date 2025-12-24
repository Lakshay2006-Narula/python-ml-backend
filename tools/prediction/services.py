# tools/prediction/services.py

import os
import json
import math
import uuid
import numpy as np
import pandas as pd
from typing import List
import joblib

# Optional geometry imports
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

# -----------------------------
# KPI limits
# -----------------------------
KPI_RANGES = {
    "RSRP": (-140.0, -44.0),
    "RSRQ": (-19.5, -3.0),
    "SINR": (0.0, 30.0),
}


# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def clamp_array(arr, kpi_name):
    lo, hi = KPI_RANGES[kpi_name]
    return np.clip(arr, lo, hi)

def normcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def standardize_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize typical lat/lon column names → lat/lon
    """
    df = df.copy()

    rename_map = {
        "latitude": "lat",
        "longitude": "lon",
        "lat_deg": "lat",
        "lon_deg": "lon",
        "x": "lon",
        "y": "lat",
        "lat_pred": "lat",
        "lon_pred": "lon",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    return df

def to_rad(df: pd.DataFrame) -> np.ndarray:
    return np.radians(np.c_[df["lat"].values, df["lon"].values])

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def make_regressor():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=800, max_depth=7, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, random_state=42
        )
    except Exception:
        return RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)

def build_preprocess(num_features, cat_features):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", numeric, num_features),
        ("cat", categorical, cat_features)
    ])


# -------------------------------------------------------------------
# Nearest Site Matching
# -------------------------------------------------------------------
def fast_match(work_site: pd.DataFrame, points_df: pd.DataFrame) -> pd.DataFrame:
    site_rad = to_rad(work_site)
    n_neighbors = min(10, len(work_site))

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="ball_tree",
        metric="haversine"
    )
    nbrs.fit(site_rad)

    pts_rad = to_rad(points_df)
    distances, indices = nbrs.kneighbors(pts_rad)
    distances_m = distances * 6371000.0

    best_idx = []
    best_bearing = []
    best_delta = []
    best_dist = []

    for i in range(points_df.shape[0]):
        lat, lon = points_df.iloc[i]["lat"], points_df.iloc[i]["lon"]
        idxs = indices[i]
        cand = work_site.iloc[idxs]

        # bearing
        phi1 = np.radians(cand["lat"].values)
        phi2 = math.radians(lat)
        dlmb = np.radians(lon - cand["lon"].values)

        x = np.sin(dlmb) * np.cos(phi2)
        y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlmb)
        brngs = (np.degrees(np.arctan2(x, y)) + 360) % 360

        deltas = np.abs((cand["azimuth"].values - brngs + 180) % 360 - 180)
        order = np.lexsort((distances_m[i], deltas))
        j = order[0]

        best_idx.append(idxs[j])
        best_bearing.append(brngs[j])
        best_delta.append(deltas[j])
        best_dist.append(distances_m[i][j])

    sel = work_site.iloc[np.array(best_idx)].reset_index(drop=True)
    out = pd.DataFrame({
        "best_site": sel["SiteName"].astype(str),
        "best_sector": sel["sector"],
        "site_lat": sel["lat"],
        "site_lon": sel["lon"],
        "site_az": sel["azimuth"],
        "dist_m": np.array(best_dist),
        "bearing_tx_to_ue": np.array(best_bearing),
        "delta_az": np.array(best_delta),
    })
    return out


# -------------------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------------------
def run_prediction_pipeline(
    db_connection,
    project_id: str,
    session_ids: List[str],
    outdir: str,
    indoor_mode: str = "heuristic",
    pixel_size_meters: float = 22.0
):

    os.makedirs(outdir, exist_ok=True)

    # ----------------------------------------------------
    # 1) Validate sessions have GPS data
    # ----------------------------------------------------
    valid_sessions = []
    invalid_sessions = []

    for sid in session_ids:
        gps_sql = f"""
            SELECT COUNT(*) c FROM tbl_network_log
            WHERE session_id = '{sid}'
            AND lat IS NOT NULL AND lon IS NOT NULL
        """
        cnt = pd.read_sql(gps_sql, db_connection)["c"].iloc[0]
        if cnt > 0:
            valid_sessions.append(str(sid))
        else:
            invalid_sessions.append(str(sid))

    if not valid_sessions:
        raise RuntimeError(
            f"No valid GPS data found in ANY session.\n"
            f"Invalid sessions: {invalid_sessions}"
        )

    # ----------------------------------------------------
    # 2) Load SITE data
    # ----------------------------------------------------
    site_df = pd.read_sql(
        f"SELECT * FROM site_noMl WHERE project_id = '{project_id}'",
        db_connection
    )
    site_df = standardize_latlon(normcols(site_df))

    # ---------- OPTION 4 FIX ----------
    # Use StartLat/StartLon first
    if "startlat" in site_df.columns and "startlon" in site_df.columns:
        site_df["lat"] = pd.to_numeric(site_df["startlat"], errors="coerce")
        site_df["lon"] = pd.to_numeric(site_df["startlon"], errors="coerce")

    # fallback to EndLat/EndLon
    if ("lat" not in site_df.columns or site_df["lat"].isna().all()) and \
       ("endlat" in site_df.columns):
        site_df["lat"] = pd.to_numeric(site_df["endlat"], errors="coerce")

    if ("lon" not in site_df.columns or site_df["lon"].isna().all()) and \
       ("endlon" in site_df.columns):
        site_df["lon"] = pd.to_numeric(site_df["endlon"], errors="coerce")

    # final cleanup
    site_df = site_df.dropna(subset=["lat", "lon"])
    if site_df.empty:
        raise RuntimeError(
            "site_noMl contains no usable StartLat/StartLon or EndLat/EndLon."
        )

    # ----------------------------------------------------
    # 3) Load DRIVE TEST data
    # ----------------------------------------------------
    valid_sql = ", ".join([f"'{s}'" for s in valid_sessions])
    dt_df = pd.read_sql(
        f"""
            SELECT *
            FROM tbl_network_log
            WHERE session_id IN ({valid_sql})
            AND (rsrp IS NOT NULL OR rsrq IS NOT NULL OR sinr IS NOT NULL)
        """,
        db_connection
    )
    dt_df = standardize_latlon(normcols(dt_df))
    dt_df = dt_df.dropna(subset=["lat", "lon"])

    if dt_df.empty:
        raise RuntimeError("Drive-test has no valid lat/lon after filtering.")

    # ----------------------------------------------------
    # 4) Build work_site table
    # ----------------------------------------------------
    site_name_col = next((c for c in ["site_key_inferred","site","cellname"] if c in site_df.columns), None)
    sector_col = next((c for c in ["sector","sector_id","cell_index"] if c in site_df.columns), None)
    az_col = next((c for c in ["azimuth_deg_5","azimuth","az"] if c in site_df.columns), None)

    work_site = site_df.copy()
    work_site["azimuth"] = pd.to_numeric(work_site[az_col], errors="coerce").fillna(0.0) if az_col else 0.0
    work_site["SiteName"] = work_site[site_name_col].astype(str) if site_name_col else np.arange(len(work_site)).astype(str)
    work_site["sector"] = work_site[sector_col] if sector_col else 1

    # ----------------------------------------------------
    # 5) Match DT with sites
    # ----------------------------------------------------
    matched_dt = fast_match(work_site, dt_df[["lat", "lon"]])
    dt_df = pd.concat([dt_df.reset_index(drop=True), matched_dt], axis=1)

    dt_df["log10_dist"] = np.log10(np.maximum(dt_df["dist_m"], 1.0))
    dt_df["angle_gain"] = np.cos(np.radians(dt_df["delta_az"])).clip(lower=0)

    # ----------------------------------------------------
    # 6) Load TEST GRID or auto-build it
    # ----------------------------------------------------
    test_df = pd.read_sql(
        f"""
            SELECT t1.lat, t1.lon, t1.band, t1.network, t1.pci
            FROM tbl_network_log t1
            JOIN tbl_savepolygon t2 ON t1.polygon_id = t2.id
            WHERE t2.project_id = '{project_id}' AND t1.rsrp IS NULL
        """,
        db_connection
    )
    test_df = standardize_latlon(normcols(test_df))

    if test_df.empty:
        min_lat = dt_df["lat"].min() - 0.0005
        max_lat = dt_df["lat"].max() + 0.0005
        min_lon = dt_df["lon"].min() - 0.0005
        max_lon = dt_df["lon"].max() + 0.0005

        step_lat = pixel_size_meters / 111111.0
        avg = np.radians((min_lat + max_lat) / 2)
        step_lon = pixel_size_meters / (111111.0 * np.cos(avg))

        lat_steps = np.arange(min_lat, max_lat, step_lat)
        lon_steps = np.arange(min_lon, max_lon, step_lon)

        if len(lat_steps) == 0 or len(lon_steps) == 0:
            raise RuntimeError("Grid generation failed due to tiny GPS variation.")

        gv_lat, gv_lon = np.meshgrid(lat_steps, lon_steps)
        test_df = pd.DataFrame({
            "lat": gv_lat.ravel(),
            "lon": gv_lon.ravel(),
            "band": "unknown",
            "network": "unknown",
            "pci": "unknown"
        })

    # ----------------------------------------------------
    # 7) Match Test Grid with sites
    # ----------------------------------------------------
    match_test = fast_match(work_site, test_df[["lat", "lon"]])
    test_df = pd.concat([test_df.reset_index(drop=True), match_test], axis=1)

    test_df["log10_dist"] = np.log10(np.maximum(test_df["dist_m"], 1.0))
    test_df["angle_gain"] = np.cos(np.radians(test_df["delta_az"])).clip(lower=0)

    # ----------------------------------------------------
    # 8) Train ML models
    # ----------------------------------------------------
    TARGETS = [t for t in ["rsrp","rsrq","sinr"] if t in dt_df.columns]

    num_features = [
        "log10_dist","dist_m","angle_gain","delta_az","bearing_tx_to_ue",
        "site_lat","site_lon","lat","lon"
    ]
    cat_features = ["band","network","pci","best_site","best_sector"]

    models = {}

    for tgt in TARGETS:
        y = pd.to_numeric(dt_df[tgt], errors="coerce")
        valid = y.notna()
        X = dt_df.loc[valid, num_features + cat_features]
        y = y.loc[valid]

        preprocess = build_preprocess(num_features, cat_features)
        reg = make_regressor()
        pipe = Pipeline([("prep", preprocess), ("reg", reg)])
        pipe.fit(X, y)

        models[tgt] = pipe

    # ----------------------------------------------------
    # 9) Predict for Test Grid
    # ----------------------------------------------------
    out = test_df.copy()

    for tgt, pipe in models.items():
        expected = num_features + cat_features
        for c in expected:
            if c not in out.columns:
                out[c] = 0.0 if c in num_features else "unknown"

        pred = pipe.predict(out[expected])
        if tgt.upper() in KPI_RANGES:
            pred = clamp_array(pred, tgt.upper())

        out[f"pred_{tgt}"] = pred

    # ----------------------------------------------------
    # 10) Save Prediction to DB
    # ----------------------------------------------------
    final = pd.DataFrame({
        "tbl_project_id": int(project_id),
        "lat": out["lat"],
        "lon": out["lon"],
        "rsrp": out.get("pred_rsrp"),
        "rsrq": out.get("pred_rsrq"),
        "sinr": out.get("pred_sinr"),
        "serving_cell": out["best_site"],
        "band": out["band"],
        "earfcn": out.get("dl_freq_mhz"),
        "pci": out["pci"],
        "network": out["network"],
        "azimuth": np.nan,
        "tx_power": np.nan,
        "height": np.nan,
        "reference_signal_power": np.nan,
        "mtilt": np.nan,
        "etilt": np.nan,
    })

    final.to_sql(
        "tbl_prediction_data",
        db_connection,
        if_exists="append",
        index=False,
        method="multi"
    )

    return outdir, len(final)

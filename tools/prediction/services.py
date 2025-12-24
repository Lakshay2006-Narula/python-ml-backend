# tools/prediction/services.py
import os
import json
import math
import uuid
import numpy as np
import pandas as pd
from typing import List
import joblib

# Optional geometry
try:
    from shapely import wkt
    from shapely.geometry import Point
    HAS_SHAPELY = True
except:
    HAS_SHAPELY = False

# ML Imports
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle

# Optional XGBoost & LightGBM
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except:
    XGB_OK = False

try:
    import lightgbm as lgb
    LGB_OK = True
except:
    LGB_OK = False


# KPI CLAMP RANGES
KPI_RANGES = {
    "RSRP": (-140, -44),
    "RSRQ": (-19.5, -3),
    "SINR": (0, 30)
}

# ======================================================================
#  HELPER FUNCTIONS
# ======================================================================

def clamp_array(arr, kpi):
    lo, hi = KPI_RANGES[kpi]
    return np.clip(arr, lo, hi)

def normcols(df):
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    return df

def standardize_latlon(df):
    df = df.copy()
    mapping = {}

    for c in df.columns:
        if c.lower() in ("lat", "latitude", "lat_deg", "y"):
            mapping[c] = "lat"
        if c.lower() in ("lon", "longitude", "lon_deg", "x"):
            mapping[c] = "lon"

    df = df.rename(columns=mapping)
    return df

def to_rad(df):
    return np.radians(np.c_[df["lat"].values, df["lon"].values])

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)

    a = (np.sin(dphi/2)**2 +
         np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2)
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def make_regressor():
    if XGB_OK:
        return XGBRegressor(
            n_estimators=800, max_depth=8,
            learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.8, tree_method="hist",
            random_state=42, n_jobs=-1
        )
    if LGB_OK:
        return lgb.LGBMRegressor(
            n_estimators=1200, num_leaves=64,
            learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.8, random_state=42
        )

    return RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)

def build_preprocess(num, cat):
    numeric = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    categorical = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num),
            ("cat", categorical, cat)
        ]
    )

# ======================================================================
#  FAST NEAREST SITE MATCHING
# ======================================================================
def fast_match(work_site, points_df):

    if len(work_site) == 0:
        raise RuntimeError("work_site is empty — cannot run NearestNeighbors")

    site_rad = to_rad(work_site)
    pts_rad  = to_rad(points_df)

    nbrs = NearestNeighbors(
        n_neighbors=min(10, len(work_site)),
        algorithm="ball_tree",
        metric="haversine"
    )
    nbrs.fit(site_rad)

    dist, idxs = nbrs.kneighbors(pts_rad)
    dist_m = dist * 6371000

    best_idx = []
    best_bearing = []
    best_delta = []
    best_dist = []

    for i in range(len(points_df)):
        lat = points_df.iloc[i]["lat"]
        lon = points_df.iloc[i]["lon"]

        candidate = work_site.iloc[idxs[i]]

        phi1 = np.radians(candidate["lat"].values)
        phi2 = np.radians(lat)
        dLon = np.radians(lon - candidate["lon"].values)

        x = np.sin(dLon) * np.cos(phi2)
        y = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dLon)

        bearings = (np.degrees(np.arctan2(x, y)) + 360) % 360
        deltas = np.abs((candidate["azimuth"].values - bearings + 180) % 360 - 180)

        order = np.lexsort((dist_m[i], deltas))
        j = order[0]

        best_idx.append(idxs[i][j])
        best_bearing.append(bearings[j])
        best_delta.append(deltas[j])
        best_dist.append(dist_m[i][j])

    sel = work_site.iloc[np.array(best_idx)]

    return pd.DataFrame({
        "best_site": sel["SiteName"].astype(str).values,
        "best_sector": sel["sector"].values,
        "site_lat": sel["lat"].values,
        "site_lon": sel["lon"].values,
        "site_az": sel["azimuth"].values,
        "dist_m": np.array(best_dist),
        "bearing_tx_to_ue": np.array(best_bearing),
        "delta_az": np.array(best_delta),
    })

# ======================================================================
#  MAIN PIPELINE
# ======================================================================
def run_prediction_pipeline(
    db_connection,
    project_id: str,
    session_ids: List[str],
    outdir: str,
    indoor_mode="heuristic",
    pixel_size_meters=22.0
):
    os.makedirs(outdir, exist_ok=True)

    # ------------------------------------------------------------
    #  1) VERIFY SESSIONS HAVE GPS
    # ------------------------------------------------------------
    valid_sessions = []
    invalid_sessions = []

    for sid in session_ids:
        sql = f"""
            SELECT COUNT(*) AS c
            FROM tbl_network_log
            WHERE session_id='{sid}'
              AND lat IS NOT NULL
              AND lon IS NOT NULL
        """
        cnt = pd.read_sql(sql, db_connection)["c"].iloc[0]
        if cnt > 0:
            valid_sessions.append(str(sid))
        else:
            invalid_sessions.append(str(sid))

    if not valid_sessions:
        raise RuntimeError(
            f"All input sessions have NO GPS.\nInvalid: {invalid_sessions}"
        )

    # ------------------------------------------------------------
    #  2) LOAD SITE TABLE (site_noMl)
    # ------------------------------------------------------------
    site_sql = f"""
        SELECT *
        FROM site_noMl
        WHERE project_id='{project_id}'
    """
    site_df = pd.read_sql(site_sql, db_connection)
    site_df = standardize_latlon(normcols(site_df))

    # Remove rows where *both* lat & lon are NULL
    site_df = site_df[
        ~(site_df["lat"].isna() & site_df["lon"].isna())
    ].copy()

    if site_df.empty:
        raise RuntimeError(
            "site_noMl exists but contains no usable lat/lon entries.\n"
            "Fix needed in run_noml."
        )

    # ------------------------------------------------------------
    #  3) LOAD DRIVE TEST DATA
    # ------------------------------------------------------------
    valid_sql = ", ".join([f"'{s}'" for s in valid_sessions])

    dt_sql = f"""
        SELECT *
        FROM tbl_network_log
        WHERE session_id IN ({valid_sql})
          AND (rsrp IS NOT NULL OR rsrq IS NOT NULL OR sinr IS NOT NULL)
    """

    dt_df = pd.read_sql(dt_sql, db_connection)
    dt_df = standardize_latlon(normcols(dt_df))

    # DT must have GPS
    dt_core = dt_df.dropna(subset=["lat", "lon"]).copy()

    if dt_core.empty:
        raise RuntimeError("Drive-test cleaned dataset has NO GPS points.")

    # ------------------------------------------------------------
    #  4) BUILD SITE WORK TABLE
    # ------------------------------------------------------------
    site_name_col = next((c for c in [
        "site_key_inferred", "site", "cellname"
    ] if c in site_df.columns), None)

    sector_col = next((c for c in [
        "sector", "sector_id", "cell_index", "sector_count"
    ] if c in site_df.columns), None)

    az_col = next((c for c in [
        "azimuth", "az", "azimuth_deg_5"
    ] if c in site_df.columns), None)

    work_site = site_df.copy()
    work_site["azimuth"] = (
        pd.to_numeric(work_site[az_col], errors="coerce").fillna(0)
        if az_col else 0
    )
    work_site["SiteName"] = (
        work_site[site_name_col].astype(str)
        if site_name_col else np.arange(len(work_site)).astype(str)
    )
    work_site["sector"] = (
        pd.to_numeric(work_site[sector_col], errors="coerce").fillna(1)
        if sector_col else 1
    )

    # ------------------------------------------------------------
    #  5) MATCH DT → SITES
    # ------------------------------------------------------------
    matched_dt = fast_match(work_site, dt_core[["lat", "lon"]])
    dt_core = pd.concat([dt_core.reset_index(drop=True),
                         matched_dt.reset_index(drop=True)], axis=1)

    dt_core["log10_dist"] = np.log10(np.maximum(dt_core["dist_m"], 1))
    dt_core["angle_gain"] = np.cos(np.radians(dt_core["delta_az"])).clip(lower=0)

    # ------------------------------------------------------------
    #  6) LOAD GRID OR AUTO-GENERATE GRID
    # ------------------------------------------------------------
    test_sql = f"""
        SELECT t1.lat, t1.lon, t1.band, t1.network, t1.pci
        FROM tbl_network_log t1
        JOIN tbl_savepolygon t2 ON t1.polygon_id=t2.id
        WHERE t2.project_id='{project_id}'
          AND t1.rsrp IS NULL
    """

    test_df = pd.read_sql(test_sql, db_connection)
    test_df = standardize_latlon(normcols(test_df))

    if test_df.empty:
        # Auto-create grid around DT bounding box
        min_lat = dt_core["lat"].min() - 0.0005
        max_lat = dt_core["lat"].max() + 0.0005
        min_lon = dt_core["lon"].min() - 0.0005
        max_lon = dt_core["lon"].max() + 0.0005

        step_lat = pixel_size_meters / 111111
        avg_lat = np.radians((min_lat + max_lat) / 2)
        step_lon = pixel_size_meters / (111111 * np.cos(avg_lat))

        lats = np.arange(min_lat, max_lat, step_lat)
        lons = np.arange(min_lon, max_lon, step_lon)

        if len(lats) == 0 or len(lons) == 0:
            raise RuntimeError("Auto-grid failed — too small area.")

        gv_lat, gv_lon = np.meshgrid(lats, lons)

        test_df = pd.DataFrame({
            "lat": gv_lat.ravel(),
            "lon": gv_lon.ravel(),
            "band": "unknown",
            "network": "unknown",
            "pci": "unknown"
        })

    # ------------------------------------------------------------
    #  7) MATCH GRID POINTS → SITES
    # ------------------------------------------------------------
    matched_grid = fast_match(work_site, test_df[["lat", "lon"]])
    test_df = pd.concat([test_df.reset_index(drop=True),
                         matched_grid.reset_index(drop=True)], axis=1)

    test_df["log10_dist"] = np.log10(np.maximum(test_df["dist_m"], 1))
    test_df["angle_gain"] = np.cos(np.radians(test_df["delta_az"])).clip(lower=0)

    # ------------------------------------------------------------
    #  8) TRAIN MODELS
    # ------------------------------------------------------------
    num_features = [
        "log10_dist", "dist_m", "angle_gain",
        "delta_az", "bearing_tx_to_ue",
        "site_lat", "site_lon", "lat", "lon",
        "is_indoor", "est_indoor_loss_db"
    ]
    cat_features = ["band", "network", "pci", "best_site", "best_sector"]

    dt_core["is_indoor"] = 0
    dt_core["est_indoor_loss_db"] = 0

    for c in cat_features:
        if c not in dt_core.columns:
            dt_core[c] = "unknown"
        dt_core[c] = dt_core[c].astype(str)

    TARGETS = [t for t in ["rsrp", "rsrq", "sinr"] if t in dt_core.columns]
    models = {}

    for tgt in TARGETS:
        y = pd.to_numeric(dt_core[tgt], errors="coerce")
        good = y.notna()

        X = dt_core.loc[good, num_features + cat_features]
        y = y.loc[good]

        if X.empty:
            raise RuntimeError(f"No valid samples for {tgt}")

        pipe = Pipeline([
            ("prep", build_preprocess(num_features, cat_features)),
            ("reg", make_regressor())
        ])
        pipe.fit(X, y)

        models[tgt] = pipe

    # ------------------------------------------------------------
    #  9) PREDICT GRID
    # ------------------------------------------------------------
    out = test_df.copy()
    out["is_indoor"] = 0
    out["est_indoor_loss_db"] = 0

    for c in cat_features:
        if c not in out.columns:
            out[c] = "unknown"
        out[c] = out[c].astype(str)

    for tgt, pipe in models.items():
        req_cols = num_features + cat_features

        for col in req_cols:
            if col not in out.columns:
                out[col] = 0 if col in num_features else "unknown"

        pred = pipe.predict(out[req_cols])

        # clamp KPI values
        pred = clamp_array(pred, tgt.upper())

        out[f"pred_{tgt}"] = pred

    # ------------------------------------------------------------
    # 10) CREATE FINAL DB TABLE
    # ------------------------------------------------------------
    final_df = pd.DataFrame({
        "tbl_project_id": int(project_id),
        "lat": out["lat"],
        "lon": out["lon"],
        "rsrp": out.get("pred_rsrp", np.nan),
        "rsrq": out.get("pred_rsrq", np.nan),
        "sinr": out.get("pred_sinr", np.nan),
        "serving_cell": out["best_site"],
        "band": out.get("band", "unknown"),
        "earfcn": out.get("dl_freq_mhz", np.nan),
        "pci": out["pci"],
        "network": out["network"],
        "azimuth": np.nan,
        "tx_power": np.nan,
        "height": np.nan,
        "reference_signal_power": np.nan,
        "mtilt": np.nan,
        "etilt": np.nan
    })

    # ------------------------------------------------------------
    # 11) SAVE TO DB
    # ------------------------------------------------------------
    final_df.to_sql(
        "tbl_prediction_data",
        db_connection,
        if_exists="append",
        index=False,
        method="multi"
    )

    return outdir, len(final_df)

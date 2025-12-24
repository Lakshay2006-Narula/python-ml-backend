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

# Optional geometry and ML imports
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

# ----- KPI standard ranges -----
KPI_RANGES = {
    "RSRP": (-140.0, -44.0),
    "RSRQ": (-19.5, -3.0),
    "SINR": (0.0, 30.0),
}

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def clamp_array(arr, kpi_name):
    lo, hi = KPI_RANGES[kpi_name]
    return np.clip(arr, lo, hi)

def normcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    return df

def standardize_latlon(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    if "latitude" in cols and "longitude" in cols:
        return df.rename(columns={"latitude": "lat", "longitude": "lon"})
    if "lat_pred" in cols and "lon_pred" in cols:
        return df.rename(columns={"lat_pred": "lat", "lon_pred": "lon"})
    if "lat" in cols and "lon" in cols:
        return df
    mapping = {}
    for c in df.columns:
        if c in ("latitude", "lat_deg", "y"): mapping[c] = "lat"
        if c in ("longitude", "lon_deg", "x"): mapping[c] = "lon"
    return df.rename(columns=mapping)

def to_rad(df: pd.DataFrame) -> np.ndarray:
    return np.radians(np.c_[df["lat"].values, df["lon"].values])

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2.0)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

def make_regressor():
    if XGB_OK:
        return XGBRegressor(n_estimators=900, max_depth=8, learning_rate=0.05, subsample=0.9,
                            colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
                            tree_method="hist", random_state=42, n_jobs=-1)
    if LGB_OK:
        return lgb.LGBMRegressor(n_estimators=1400, num_leaves=63, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, random_state=42)
    try:
        return GradientBoostingRegressor(random_state=42)
    except Exception:
        return RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)

def build_preprocess(num_features: List[str], cat_features: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features)
        ],
        remainder="drop"
    )

def fast_match(work_site: pd.DataFrame, points_df: pd.DataFrame) -> pd.DataFrame:
    site_rad = to_rad(work_site)
    n_neighbors = min(10, len(work_site)) if len(work_site) > 0 else 1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree", metric="haversine")
    nbrs.fit(site_rad)
    pts_rad = to_rad(points_df)
    distances, indices = nbrs.kneighbors(pts_rad)
    distances_m = distances * 6371000.0
    best_idx, best_bearing, best_delta, best_dist = [], [], [], []
    for i in range(points_df.shape[0]):
        lat, lon = points_df.iloc[i]["lat"], points_df.iloc[i]["lon"]
        idxs = indices[i]
        candidate_sites = work_site.iloc[idxs]
        phi1 = np.radians(candidate_sites["lat"].values)
        phi2 = math.radians(lat)
        dlmb = np.radians(lon - candidate_sites["lon"].values)
        x = np.sin(dlmb) * np.cos(phi2)
        y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlmb)
        brngs = (np.degrees(np.arctan2(x, y)) + 360) % 360
        deltas = np.abs((candidate_sites["azimuth"].values - brngs + 180) % 360 - 180)
        order = np.lexsort((distances_m[i], deltas))
        j = order[0]
        best_idx.append(idxs[j])
        best_bearing.append(brngs[j])
        best_delta.append(deltas[j])
        best_dist.append(distances_m[i][j])
    sel = work_site.iloc[np.array(best_idx)].reset_index(drop=True)
    out = pd.DataFrame({
        "best_site": sel["SiteName"].astype(str).values,
        "best_sector": sel["sector"].values,
        "site_lat": sel["lat"].values,
        "site_lon": sel["lon"].values,
        "site_az": sel["azimuth"].values,
        "dist_m": np.array(best_dist),
        "bearing_tx_to_ue": np.array(best_bearing),
        "delta_az": np.array(best_delta),
    })
    return out

def load_buildings_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normcols(df) 
    has_loss_cols = False
    for c in ["attenuation_db","loss_db"]:
        if c in df.columns:
            df["est_loss_db"] = pd.to_numeric(df[c], errors="coerce")
            has_loss_cols = True
            break
    if "est_loss_db" not in df.columns:
        df["est_loss_db"] = np.nan
        
    if "height_m" in df.columns: has_loss_cols = True
    if "floors" in df.columns: has_loss_cols = True
    if "material" in df.columns: has_loss_cols = True

    if "height_m" in df.columns:
        df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")
    if "floors" in df.columns:
        df["floors"] = pd.to_numeric(df["floors"], errors="coerce")
    if "material" not in df.columns:
        df["material"] = np.nan
    return df

def estimate_loss_from_meta(row) -> float:
    base = 10.0
    mat = str(row.get("material", "")).lower()
    if "concrete" in mat: base = 15.0
    elif "glass" in mat or "curtain" in mat: base = 8.0
    elif "brick" in mat: base = 12.0
    
    floors = row.get("floors", np.nan)
    height = row.get("height_m", np.nan)
    add = 0.0
    if pd.notna(floors): add += min(12.0, max(0.0, (floors - 1) * 0.8))
    if pd.notna(height): add += min(10.0, max(0.0, (height / 3.0) * 0.3))
    return float(base + add)

def mark_indoor(points_df: pd.DataFrame, bld: pd.DataFrame) -> pd.DataFrame:
    out = points_df.copy()
    out["is_indoor"] = 0
    out["est_indoor_loss_db"] = 0.0
    
    wkt_col = None
    for c in ["wkt","geometry"]:
        if c in bld.columns:
            wkt_col = c
            break
    if wkt_col and HAS_SHAPELY:
        polys, metas = [], []
        for _, r in bld.iterrows():
            try:
                geom = wkt.loads(str(r[wkt_col]))
                if geom.is_valid:
                    polys.append(geom)
                    metas.append(r.to_dict())
            except Exception:
                continue
        for i in range(len(out)):
            pt = Point(out.loc[i,"lon"], out.loc[i,"lat"])
            for meta, poly in zip(metas, polys):
                if poly.contains(pt):
                    out.loc[i,"is_indoor"] = 1
                    loss = meta.get("est_loss_db", np.nan)
                    out.loc[i,"est_indoor_loss_db"] = float(loss) if pd.notna(loss) else estimate_loss_from_meta(meta)
                    break
        return out

    cols = set(bld.columns)
    if ("lat" in cols or "latitude" in cols) and ("lon" in cols or "longitude" in cols):
        bld2 = standardize_latlon(bld.copy())
        if "radius_m" in bld2.columns:
            bld2["radius_m"] = pd.to_numeric(bld2["radius_m"], errors="coerce").fillna(20.0)
        else:
            bld2["radius_m"] = 20.0
        for i in range(len(out)):
            lat, lon = out.loc[i,"lat"], out.loc[i,"lon"]
            dists = haversine_m(lat, lon, bld2["lat"].values, bld2["lon"].values)
            j = int(np.argmin(dists))
            if dists[j] <= float(bld2.iloc[j]["radius_m"]):
                out.loc[i,"is_indoor"] = 1
                loss = bld2.iloc[j].get("est_loss_db", np.nan)
                out.loc[i,"est_indoor_loss_db"] = float(loss) if pd.notna(loss) else estimate_loss_from_meta(bld2.iloc[j].to_dict())
        return out
    return out

# -------------------------------------------------------------------
# MAIN PIPELINE FUNCTION
# -------------------------------------------------------------------
def run_prediction_pipeline(
    db_connection,
    project_id: str,
    session_ids: List[str],
    outdir: str,
    indoor_mode: str = "heuristic",
    pixel_size_meters: float = 22.0
):
    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)

    # -------------------------------
    # 1) VALIDATE SESSIONS HAVE GPS
    # -------------------------------
    valid_sessions = []
    invalid_sessions = []

    for sid in session_ids:
        gps_check_sql = f"""
            SELECT COUNT(*) AS c 
            FROM tbl_network_log 
            WHERE session_id = '{sid}'
            AND lat IS NOT NULL 
            AND lon IS NOT NULL
        """
        cnt = pd.read_sql(gps_check_sql, db_connection)["c"].iloc[0]

        if cnt > 0:
            valid_sessions.append(str(sid))
        else:
            invalid_sessions.append(str(sid))

    # ❌ No valid sessions → stop here
    if not valid_sessions:
        raise RuntimeError(
            f"None of the provided session_ids contain GPS lat/lon.\n"
            f"Invalid sessions: {invalid_sessions}"
        )

    print(f"VALID sessions used: {valid_sessions}")
    print(f"IGNORED sessions (no GPS): {invalid_sessions}")

    # -------------------------------
    # 2) LOAD SITE DATA
    # -------------------------------
    site_query = f"SELECT * FROM site_noMl WHERE project_id = '{project_id}'"

    site_df = pd.read_sql(site_query, db_connection)
    site_df = standardize_latlon(normcols(site_df))

    if site_df.empty:
        raise RuntimeError(
            f"No site entries found in site_noMl for project_id={project_id}"
        )

    # Remove rows without lat/lon
    site_df = site_df.dropna(subset=["lat", "lon"])
    if site_df.empty:
        raise RuntimeError("Site table exists but contains no valid lat/lon entries.")

    # -------------------------------
    # 3) LOAD DRIVE TEST DATA
    # -------------------------------
    valid_sql = ", ".join([f"'{s}'" for s in valid_sessions])

    drive_sql = f"""
        SELECT *
        FROM tbl_network_log
        WHERE session_id IN ({valid_sql})
        AND (rsrp IS NOT NULL OR rsrq IS NOT NULL OR sinr IS NOT NULL)
    """

    dt_df = pd.read_sql(drive_sql, db_connection)
    dt_df = standardize_latlon(normcols(dt_df))

    # Remove missing lat/lon
    dt_core = dt_df.dropna(subset=["lat", "lon"])

    if dt_core.empty:
        raise RuntimeError(
            f"Drive test contains NO valid lat/lon points after cleaning.\n"
            f"Valid sessions tried: {valid_sessions}"
        )

    # --------------------------------------------------
    # 4) BUILD WORK_SITE TABLE FOR NEAREST NEIGHBORS
    # --------------------------------------------------
    site_name_col = next((c for c in ["site_key_inferred", "site", "cellname"] if c in site_df.columns), None)
    sector_col = next((c for c in ["sector", "sector_id", "cell_index"] if c in site_df.columns), None)
    az_col = next((c for c in ["azimuth_deg_5", "azimuth", "az"] if c in site_df.columns), None)

    work_site = site_df.copy()
    work_site["azimuth"] = pd.to_numeric(work_site[az_col], errors="coerce").fillna(0.0) if az_col else 0.0
    work_site["SiteName"] = work_site[site_name_col].astype(str) if site_name_col else np.arange(len(work_site)).astype(str)
    work_site["sector"] = work_site[sector_col] if sector_col else 1

    # 🚨 MUST HAVE AT LEAST ONE SITE
    if len(work_site) == 0:
        raise RuntimeError("work_site dataframe is empty — cannot run prediction.")

    # --------------------------------------------------
    # 5) MATCH DRIVE TEST WITH SITES
    # --------------------------------------------------
    dt_matched = fast_match(work_site, dt_core[["lat", "lon"]])
    dt_core = pd.concat([dt_core.reset_index(drop=True), dt_matched.reset_index(drop=True)], axis=1)

    # Add distance-based features
    dt_core["log10_dist"] = np.log10(np.maximum(dt_core["dist_m"], 1.0))
    dt_core["angle_gain"] = np.cos(np.radians(dt_core["delta_az"])).clip(lower=0)

    # --------------------------------------------------
    # 6) HANDLE MISSING TEST GRID
    # If test_df empty → auto-generate grid around DT data
    # --------------------------------------------------
    test_query = f"""
        SELECT t1.lat, t1.lon, t1.band, t1.network, t1.pci
        FROM tbl_network_log t1
        JOIN tbl_savepolygon t2 ON t1.polygon_id = t2.id
        WHERE t2.project_id = '{project_id}' AND t1.rsrp IS NULL
    """
    test_df = pd.read_sql(test_query, db_connection)
    test_df = standardize_latlon(normcols(test_df))

    if test_df.empty:
        # Auto-generate prediction grid
        min_lat = dt_core["lat"].min() - 0.0005
        max_lat = dt_core["lat"].max() + 0.0005
        min_lon = dt_core["lon"].min() - 0.0005
        max_lon = dt_core["lon"].max() + 0.0005

        # Convert meters → degrees
        step_lat = pixel_size_meters / 111111.0
        avg_lat_rad = np.radians((min_lat + max_lat) / 2)
        step_lon = pixel_size_meters / (111111.0 * np.cos(avg_lat_rad))

        lat_steps = np.arange(min_lat, max_lat, step_lat)
        lon_steps = np.arange(min_lon, max_lon, step_lon)

        # 🚨 If grid empty → FAIL HERE
        if len(lat_steps) == 0 or len(lon_steps) == 0:
            raise RuntimeError(
                "Failed to auto-build prediction grid. Check if drive test GPS variance is too small."
            )

        gv_lat, gv_lon = np.meshgrid(lat_steps, lon_steps)

        test_df = pd.DataFrame({
            "lat": gv_lat.ravel(),
            "lon": gv_lon.ravel(),
            "band": "unknown",
            "network": "unknown",
            "pci": "unknown",
        })

    # --------------------------------------------------
    # 7) MATCH TEST POINTS WITH SITES
    # --------------------------------------------------
    test_match = fast_match(work_site, test_df[["lat", "lon"]])
    test_df = pd.concat([test_df.reset_index(drop=True), test_match.reset_index(drop=True)], axis=1)

    # Additional features
    test_df["log10_dist"] = np.log10(np.maximum(test_df["dist_m"], 1.0))
    test_df["angle_gain"] = np.cos(np.radians(test_df["delta_az"])).clip(lower=0)

    # --------------------------------------------------
    # 8) TRAIN MODELS (RSRP/RSRQ/SINR)
    # --------------------------------------------------
    num_features = [
        "dl_freq_mhz", "log10_dist", "dist_m", "angle_gain",
        "delta_az", "bearing_tx_to_ue", "site_lat", "site_lon",
        "lat", "lon", "is_indoor", "est_indoor_loss_db"
    ]

    cat_features = ["band", "network", "pci", "best_site", "best_sector"]

    # Build indoor labels
    dt_core["is_indoor"] = 0
    dt_core["est_indoor_loss_db"] = 0.0

    # Select target KPIs
    TARGETS = [t for t in ["rsrp", "rsrq", "sinr"] if t in dt_core.columns]

    models = {}

    for tgt in TARGETS:
        y = pd.to_numeric(dt_core[tgt], errors="coerce")
        valid = y.notna()

        X = dt_core.loc[valid, num_features + cat_features]
        y = y.loc[valid]

        if X.empty:
            raise RuntimeError(f"No valid training samples for {tgt}")

        # Build full ML pipeline
        preprocess = build_preprocess(num_features, cat_features)
        reg = make_regressor()
        pipe = Pipeline(steps=[("prep", preprocess), ("reg", reg)])

        pipe.fit(X, y)
        models[tgt] = pipe

    # --------------------------------------------------
    # 9) PREDICT FOR TEST GRID
    # --------------------------------------------------
    out = test_df.copy()

    for tgt, pipe in models.items():
        expected_cols = num_features + cat_features

        # Fill missing columns
        for col in expected_cols:
            if col not in out.columns:
                out[col] = 0.0 if col in num_features else "unknown"

        pred = pipe.predict(out[expected_cols])

        # Clamp values
        if tgt.upper() in KPI_RANGES:
            pred = clamp_array(pred, tgt.upper())

        # Indoor correction
        if indoor_mode == "heuristic":
            loss = out["est_indoor_loss_db"].fillna(0.0).values
            is_in = out["is_indoor"].values.astype(int)

            if tgt == "rsrp":
                pred = pred - (loss * is_in)
            elif tgt == "rsrq":
                pred = pred - np.minimum(3.0, loss * 0.15) * is_in
            elif tgt == "sinr":
                pred = pred - np.minimum(10.0, loss * 0.6) * is_in

        out[f"pred_{tgt}"] = pred

    # Prepare DB output
    final_out = pd.DataFrame({
        "tbl_project_id": int(project_id),
        "lat": out["lat"],
        "lon": out["lon"],
        "rsrp": out.get("pred_rsrp", np.nan),
        "rsrq": out.get("pred_rsrq", np.nan),
        "sinr": out.get("pred_sinr", np.nan),
        "serving_cell": out["best_site"],
        "band": out["band"],
        "earfcn": out["dl_freq_mhz"],
        "pci": out["pci"],
        "network": out["network"],
        "azimuth": np.nan,
        "tx_power": np.nan,
        "height": np.nan,
        "reference_signal_power": np.nan,
        "mtilt": np.nan,
        "etilt": np.nan,
    })

    # Save to DB
    final_out.to_sql(
        name="tbl_prediction_data",
        con=db_connection,
        if_exists="append",
        index=False,
        method="multi"
    )

    return outdir, len(final_out)

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
    pixel_size_meters: float = 22.0  #  (Default 22m)
):
    os.makedirs(outdir, exist_ok=True)
    
    if not session_ids:
        raise RuntimeError("No Session_ids provided.")
        
    # 1. Load Site Data
    # 🟢 FIX: Use lowercase 'site_noml' for Linux/AWS RDS compatibility
    site_query = f"SELECT * FROM `site_noml` WHERE project_id = '{project_id}'"
    
    try:
        site_df = pd.read_sql(site_query, db_connection)
    except Exception as e:
        # Fallback for systems where table might still be CamelCase
        try:
            site_query_alt = f"SELECT * FROM `site_noMl` WHERE project_id = '{project_id}'"
            site_df = pd.read_sql(site_query_alt, db_connection)
        except:
            raise RuntimeError(f"Could not load site data from DB. Error: {e}")

    site_df = standardize_latlon(normcols(site_df))
    if site_df.empty:
        raise RuntimeError(f"No site data found for project_id: {project_id}. Ensure you ran the Upload/Process Session step successfully.")

    # 2. Load Drive Test (Training) Data
    session_ids_sql_str = ", ".join([f"'{s}'" for s in session_ids])
    drive_query = f"SELECT * FROM `tbl_network_log` WHERE session_id IN ({session_ids_sql_str}) AND (rsrp IS NOT NULL OR rsrq IS NOT NULL OR sinr IS NOT NULL)"
    dt_df = pd.read_sql(drive_query, db_connection)
    dt_df = standardize_latlon(normcols(dt_df))
    if dt_df.empty:
        raise RuntimeError(f"No drive test data found for session_ids: {session_ids_sql_str}")

    # 3. Load Test (Pixel) Data
    test_query = f"""
        SELECT t1.lat, t1.lon, t1.band, t1.network, t1.pci
        FROM `tbl_network_log` t1 
        JOIN `tbl_savepolygon` t2 ON t1.polygon_id = t2.id 
        WHERE t2.project_id = '{project_id}' AND t1.rsrp IS NULL
    """
    test_df = pd.read_sql(test_query, db_connection)
    test_df = standardize_latlon(normcols(test_df))
    
    if test_df.empty:
        # Calculate bounds with a small buffer
        min_lat = dt_df['lat'].min() - 0.0005 
        max_lat = dt_df['lat'].max() + 0.0005
        min_lon = dt_df['lon'].min() - 0.0005
        max_lon = dt_df['lon'].max() + 0.0005
        
        # 1 degree lat approx 111,111 meters
        step_lat = pixel_size_meters / 111111.0
        
        # cos(lat) adjustment is needed because longitude lines get closer at poles
        avg_lat_rad = np.radians((min_lat + max_lat) / 2)
        step_lon = pixel_size_meters / (111111.0 * np.cos(avg_lat_rad))
        # -----------------------------

        lat_steps = np.arange(min_lat, max_lat, step_lat)
        lon_steps = np.arange(min_lon, max_lon, step_lon)
        gv_lat, gv_lon = np.meshgrid(lat_steps, lon_steps)
        
        test_df = pd.DataFrame({
            "lat": gv_lat.ravel(),
            "lon": gv_lon.ravel()
        })
        test_df['band'] = "unknown"
        test_df['network'] = "unknown"
        test_df['pci'] = "unknown"

    # 4. Load Building Data
    try:
        bld_query = f"SELECT * FROM `tbl_savepolygon` WHERE project_id = '{project_id}'"
        bld_df = pd.read_sql(bld_query, db_connection)
        if not bld_df.empty:
            bld_df = load_buildings_from_df(bld_df)
        else:
            bld_df = None
    except Exception:
        bld_df = None
    
    site_name_col = next((c for c in ["site_key_inferred", "site","cellname"] if c in site_df.columns), None)
    sector_col    = next((c for c in ["sector", "sector_id","cell_index"] if c in site_df.columns), None)
    az_col        = next((c for c in ["azimuth_deg_5", "azimuth","az"] if c in site_df.columns), None)

    site_df = site_df.dropna(subset=["lat","lon"]).copy()
    work_site = site_df.copy()
    work_site["azimuth"]  = pd.to_numeric(work_site[az_col], errors="coerce").fillna(0.0) if az_col else 0.0
    work_site["SiteName"] = work_site[site_name_col].astype(str) if site_name_col else np.arange(len(work_site)).astype(str)
    work_site["sector"]   = work_site[sector_col] if sector_col else 1

    dt_core = dt_df.dropna(subset=["lat","lon"]).copy()
    for c in ["earfcn","band","pci","network"]:
        if c not in dt_core.columns:
            dt_core[c] = np.nan
            
    if "speed_(km/h)" in dt_core.columns:
        dt_core["speed"] = pd.to_numeric(dt_core["speed_(km/h)"], errors="coerce")
    elif "speed" in dt_core.columns:
        dt_core["speed"] = pd.to_numeric(dt_core["speed"], errors="coerce")
    else:
        dt_core["speed"] = np.nan

    TARGETS = [t for t in ["rsrp","rsrq","sinr"] if t in dt_core.columns]
    if not TARGETS:
        raise RuntimeError("Drive-test data must include at least one of rsrp, rsrq, sinr.")

    matched = fast_match(work_site, dt_core[["lat","lon"]])
    dt_core = pd.concat([dt_core.reset_index(drop=True), matched.reset_index(drop=True)], axis=1)
    dt_core["log10_dist"]  = np.log10(np.maximum(dt_core["dist_m"], 1.0))
    dt_core["angle_gain"]  = np.cos(np.radians(dt_core["delta_az"])).clip(lower=0)
    
    def rough_dl_freq_mhz(row):
        if row.get("downlink_frequency") and not pd.isna(row["downlink_frequency"]):
            try: return float(row["downlink_frequency"])
            except Exception: pass
        if row.get("earfcn") and not pd.isna(row["earfcn"]):
            try:
                earfcn = float(row["earfcn"])
                if earfcn < 600:    return 2110 + (earfcn * 0.1)
                elif earfcn < 1200: return 1930 + (earfcn - 600) * 0.1
                elif earfcn < 1950: return 1805 + (earfcn - 1200) * 0.1
                else:               return 700  + (earfcn - 5000) * 0.1
            except Exception: return np.nan
        return np.nan
    dt_core["dl_freq_mhz"] = dt_core.apply(rough_dl_freq_mhz, axis=1)

    for c in ["band","network","pci","best_site","best_sector"]:
        if c in dt_core.columns:
            dt_core[c] = dt_core[c].astype(str)

    if bld_df is not None:
        ind_train = mark_indoor(dt_core[["lat","lon"]].copy(), bld_df)
        dt_core["is_indoor"] = ind_train["is_indoor"]
        dt_core["est_indoor_loss_db"] = ind_train["est_indoor_loss_db"]
    else:
        dt_core["is_indoor"] = 0
        dt_core["est_indoor_loss_db"] = 0.0

    num_features = [f for f in ["dl_freq_mhz","log10_dist","dist_m","angle_gain","delta_az","bearing_tx_to_ue","site_lat","site_lon","lat","lon","is_indoor","est_indoor_loss_db"] if f in dt_core.columns]
    cat_features = ["band","network","pci","best_site","best_sector"]
    
    models, metrics = {}, {}
    for tgt in TARGETS:
        y = pd.to_numeric(dt_core[tgt], errors="coerce")
        valid = y.notna()
        X = dt_core.loc[valid, num_features + cat_features].copy()
        y = y.loc[valid]
        if tgt.upper() in KPI_RANGES:
            lo, hi = KPI_RANGES[tgt.upper()]
            y = y.clip(lower=lo, upper=hi)
            
        X, y = shuffle(X, y, random_state=42)
        
        preprocess = build_preprocess(num_features, cat_features)
        reg  = make_regressor()
        pipe = Pipeline(steps=[("prep", preprocess), ("reg", reg)])
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2s, rmses = [], []
        for tr, va in kf.split(X):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]
            pipe.fit(Xtr, ytr)
            p = pipe.predict(Xva)
            r2s.append(r2_score(yva, p))
            rmses.append(mean_squared_error(yva, p))
        metrics[tgt] = {"R2_mean": float(np.mean(r2s)), "R2_std":  float(np.std(r2s)),
                        "RMSE_mean": float(np.mean(rmses)), "RMSE_std":  float(np.std(rmses)),
                        "n_samples": int(len(y))}
        pipe.fit(X, y)
        models[tgt] = pipe
        joblib.dump(pipe, os.path.join(outdir, f"trained_model_{tgt}.joblib"))

    with open(os.path.join(outdir, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    test_pts = test_df.dropna(subset=["lat","lon"]).copy()
    tmatch   = fast_match(work_site, test_pts[["lat","lon"]])
    test_pts = pd.concat([test_pts.reset_index(drop=True), tmatch.reset_index(drop=True)], axis=1)
    test_pts["log10_dist"] = np.log10(np.maximum(test_pts["dist_m"], 1.0))
    test_pts["angle_gain"] = np.cos(np.radians(test_pts["delta_az"])).clip(lower=0)
    for c in ["band","network","pci","best_site","best_sector"]:
        if c not in test_pts.columns: test_pts[c] = "unknown"
        test_pts[c] = test_pts[c].astype(str)
    
    if "dl_freq_mhz" not in test_pts.columns:
        test_pts["dl_freq_mhz"] = np.nan

    if bld_df is not None:
        ind_test = mark_indoor(test_pts[["lat","lon"]].copy(), bld_df)
        test_pts["is_indoor"] = ind_test["is_indoor"]
        test_pts["est_indoor_loss_db"] = ind_test["est_indoor_loss_db"]
    else:
        test_pts["is_indoor"] = 0
        test_pts["est_indoor_loss_db"] = 0.0

    out = test_pts[["lat","lon","best_site","best_sector","dist_m","delta_az","angle_gain","is_indoor","est_indoor_loss_db"]].copy()
    
    out['band'] = test_pts['band']
    out['network'] = test_pts['network']
    out['pci'] = test_pts['pci']
    out['earfcn'] = test_pts['dl_freq_mhz']
    
    for tgt, pipe in models.items():
        expected_cols = num_features + cat_features
        for col in expected_cols:
            if col not in test_pts.columns:
                if col in num_features:
                    test_pts[col] = 0.0 
                else:
                    test_pts[col] = "unknown" 
        
        pred = pipe.predict(test_pts[expected_cols].copy())

        tgt_upper = tgt.upper()
        if indoor_mode == "heuristic":
            loss = test_pts["est_indoor_loss_db"].fillna(0.0).values
            is_in = test_pts["is_indoor"].values.astype(int)
            if tgt_upper == "RSRP": pred = pred - (loss * is_in)
            elif tgt_upper == "RSRQ": pred = pred - np.minimum(3.0, loss * 0.15) * is_in
            elif tgt_upper == "SINR": pred = pred - np.minimum(10.0, loss * 0.6) * is_in
        if tgt_upper in KPI_RANGES:
            pred = clamp_array(pred, tgt_upper)
            
        out[f"pred_{tgt}"] = pred 

    out_to_db = pd.DataFrame({
        "tbl_project_id": int(project_id),
        "lat": out["lat"],
        "lon": out["lon"],
        "rsrp": out.get("pred_rsrp", np.nan),
        "rsrq": out.get("pred_rsrq", np.nan),
        "sinr": out.get("pred_sinr", np.nan),
        "serving_cell": out.get("best_site", "unknown"),
        "azimuth": np.nan,
        "tx_power": np.nan,
        "height": np.nan,
        "band": out.get("band", "unknown"),
        "earfcn": out.get("earfcn", np.nan),
        "reference_signal_power": np.nan,
        "pci": out.get("pci", "unknown"),
        "mtilt": np.nan,
        "etilt": np.nan,
        "network": out.get("network", "unknown")
    })

    final_db_columns = [
        "tbl_project_id", "lat", "lon", "rsrp", "rsrq", "sinr",
        "serving_cell", "azimuth", "tx_power", "height",
        "band", "earfcn", "reference_signal_power", "pci",
        "mtilt", "etilt", "network"
    ]

    final_out_to_db = out_to_db[final_db_columns]

    final_out_to_db.to_sql(
        name='tbl_prediction_data',
        con=db_connection, 
        if_exists='append',
        index=False,
        method='multi'
    )

    # Return the output path and count
    return outdir, len(final_out_to_db)

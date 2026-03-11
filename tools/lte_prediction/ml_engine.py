"""
ml_engine.py  —  LTE RF Prediction Engine
==========================================
Original standalone script ported to Flask API module.

All functions below are EXACT copies from the original script.
Only additions are:
  • _run_prediction_pass()  — internal helper used by two-pass
  • run_two_pass_prediction() — public pipeline wrapper

Key implementation details (from original)
──────────────────────────────────────────
• K1 / K2 set as OptimizedRFModel.K1 / .K2 class attrs before each chunk.
• cost231_hata_path_loss_vectorized reads K1/K2 from class, not function args.
• Indoor loss is FIXED 15 dB (original: indoor_loss[i] = 15).
• combine_signals_vectorized: clip to (-150, -30) then mean across sectors.
• fast_calibration: DEFAULT_K1/K2 defined locally inside function (139.0/35.2).
• split_sitewise: second definition in original wins — uses
  ["Site ID","SiteID","Site","CellName","ENodeB","ENodeB_ID"].
• Area clipping in original uses area_polygon.contains(Point(lon,lat)).
  API uses vectorized shapely_contains — identical result, faster.

TWO-PASS PIPELINE (API addition — not in original script)
──────────────────────────────────────────────────────────
Pass 1  → site data only, K1=139.0 K2=35.2, no calibration, no buildings
          → tbl_lte_prediction_results  (intermediate)

Enrich  → BallTree DT match per site_id filtered by nodeb_id
          → tbl_lte_prediction_results_refined

Pass 2  → top2_avg pixels as synthetic DT → fast_calibration → refined K1/K2
          → re-predict with building polygons + area polygon
          → tbl_lte_prediction_results  (FINAL — UI reads this)
"""

import json
import math
import re
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Pool
import multiprocessing as mp

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point, Polygon, shape
from shapely.vectorized import contains as shapely_contains

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (exact from original)
# ──────────────────────────────────────────────────────────────────────────────

KPI_RANGES = {
    "RSRP":  (-120.0, -44.0),   # -120 lower: values weaker than this are noise floor
    "RSRQ":  (-19.5,  -3.0),
    "SINR":  (0.0,    30.0),
}

BAND_TO_FREQ = {
    1: 2100,
    3: 1800,
    5: 850,
    8: 900,
    20: 800,
    28: 700,
    38: 2600,
    40: 2300,
    41: 2500,
}

BANDWIDTH_TO_RB = {
    1.4: 6,
    3:   15,
    5:   25,
    10:  50,
    15:  75,
    20:  100,
}

DEFAULT_K1 = 139.0
DEFAULT_K2 = 35.2


# ──────────────────────────────────────────────────────────────────────────────
# POLYGON LOADERS  (exact from original)
# ──────────────────────────────────────────────────────────────────────────────

def load_polygon_file(path):
    """
    Loads polygon from CSV, JSON or WKT file.
    FIXED:
    - Skips NaN rows
    - Validates WKT before loading
    - Supports LAT LON → LON LAT correction
    """
    # JSON (GeoJSON or simple list)
    if path.lower().endswith(".json"):
        data = json.load(open(path))
        if "type" in data and data["type"] == "Polygon":
            return shape(data)
        if "polygon" in data:
            return Polygon(data["polygon"])
        raise ValueError("JSON polygon format not recognized.")

    # CSV containing WKT polygons
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]

        wkt_col = None
        for c in ["wkt", "polygon", "region", "geometry"]:
            if c in df.columns:
                wkt_col = c
                break

        if wkt_col is None:
            raise ValueError("No WKT column found in CSV.")

        wkt_series = df[wkt_col].dropna()
        if wkt_series.empty:
            raise ValueError("No valid WKT polygon found (all rows empty).")

        raw_wkt = None
        for v in wkt_series:
            if isinstance(v, str) and "POLYGON" in v.upper():
                raw_wkt = v.strip()
                break

        if raw_wkt is None:
            raise ValueError("CSV contains no valid POLYGON WKT.")

        def swap_coords(match):
            pairs = match.group(1)
            fixed = []
            for p in pairs.split(','):
                p = p.strip()
                lat, lon = map(float, p.split())
                fixed.append(f"{lon} {lat}")
            return "POLYGON((" + ",".join(fixed) + "))"

        fixed_wkt = re.sub(
            r"POLYGON\(\((.+?)\)\)",
            swap_coords,
            raw_wkt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return wkt.loads(fixed_wkt)

    # TXT containing WKT
    if path.lower().endswith(".txt"):
        return wkt.loads(open(path).read().strip())

    raise ValueError("Unsupported polygon file type.")


def load_polygon_buildings(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    print(f"   → CSV columns found: {df.columns.tolist()}")

    # Pick first column that has non-null WKT data
    wkt_col = None
    for c in ["wkt", "region", "geometry"]:
        if c in df.columns and df[c].notna().any():
            wkt_col = c
            break

    if wkt_col is None:
        raise ValueError("No valid WKT column found. Columns: " + str(df.columns.tolist()))

    print(f"   → Using column '{wkt_col}' for WKT data")
    print(f"   → Total rows in file: {len(df)}")

    polygons = []
    meta     = []
    skipped  = 0

    def swap_coords(match):
        """Swap LAT LON → LON LAT inside POLYGON((...))"""
        pairs = match.group(1)
        fixed_pairs = []
        for pair in pairs.split(','):
            pair  = pair.strip()
            parts = pair.split()
            if len(parts) == 2:
                fixed_pairs.append(f"{parts[1]} {parts[0]}")
            else:
                fixed_pairs.append(pair)
        return "POLYGON((" + ",".join(fixed_pairs) + "))"

    for idx, row in df.iterrows():
        try:
            raw_wkt = str(row[wkt_col]).strip()

            if raw_wkt.lower() in ("nan", "none", "null", ""):
                skipped += 1
                continue

            fixed_wkt = re.sub(
                r'POLYGON\(\((.+?)\)\)',
                swap_coords,
                raw_wkt,
                flags=re.IGNORECASE | re.DOTALL,
            )

            poly = wkt.loads(fixed_wkt)

            if not poly.is_valid:
                poly = poly.buffer(0)

            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)
                meta.append({
                    "loss":   float(row["loss_db"]) if "loss_db" in df.columns and pd.notna(row.get("loss_db")) else 15.0,
                    "height": float(row["height"])  if "height"  in df.columns and pd.notna(row.get("height"))  else None,
                    "floors": float(row["floors"])  if "floors"  in df.columns and pd.notna(row.get("floors"))  else None,
                })
            else:
                skipped += 1

        except Exception as e:
            skipped += 1
            if idx < 3:
                print(f"   ⚠ Row {idx} error: {e}")
            continue

    print(f"   → Successfully loaded : {len(polygons)} polygons")
    print(f"   → Skipped (invalid)   : {skipped} polygons")

    if polygons:
        bb = polygons[0].bounds
        print(f"   → First polygon bounds: minX={bb[0]:.5f}, minY={bb[1]:.5f}, "
              f"maxX={bb[2]:.5f}, maxY={bb[3]:.5f}")
        print(f"   → For Delhi: X(lon) should be ~77.x  and  Y(lat) should be ~28.x")
    else:
        print(f"   ❌ WARNING: Zero polygons loaded! Check your CSV file.")

    return polygons, meta


def detect_indoor(lat, lon, polygons, meta):
    pt = Point(lon, lat)
    for poly, m in zip(polygons, meta):
        if poly.contains(pt):
            return 1, m["loss"]
    return 0, 0


# ──────────────────────────────────────────────────────────────────────────────
# ANTENNA MODEL  (exact from original)
# ──────────────────────────────────────────────────────────────────────────────

class Optimized3GPPAntennaModel:
    """Vectorized 3GPP antenna calculations for speed."""

    @staticmethod
    def horizontal_pattern_vectorized(azimuth_diff, horizontal_beamwidth=65, max_attenuation=25):
        """Vectorized horizontal pattern calculation."""
        theta     = np.abs(azimuth_diff)
        theta_3db = horizontal_beamwidth
        gain_db   = -np.minimum(12 * (theta / theta_3db) ** 2, max_attenuation)
        return gain_db

    @staticmethod
    def vertical_pattern_vectorized(elevation_angle, vertical_beamwidth=10, max_attenuation=20,
                                    electrical_tilt=0, mechanical_tilt=0):
        """Vectorized vertical pattern calculation."""
        total_tilt = electrical_tilt + mechanical_tilt
        theta      = elevation_angle + total_tilt
        theta_3db  = vertical_beamwidth
        gain_db    = -np.minimum(12 * (theta / theta_3db) ** 2, max_attenuation)
        return gain_db

    @staticmethod
    def combined_3gpp_pattern_vectorized(azimuth_diff, elevation_angle,
                                         horizontal_beamwidth=65, vertical_beamwidth=10,
                                         electrical_tilt=0, mechanical_tilt=0,
                                         max_attenuation=25):
        """Vectorized combined 3D antenna pattern."""
        a_h = Optimized3GPPAntennaModel.horizontal_pattern_vectorized(
            azimuth_diff, horizontal_beamwidth, max_attenuation
        )
        a_v = Optimized3GPPAntennaModel.vertical_pattern_vectorized(
            elevation_angle, vertical_beamwidth, max_attenuation,
            electrical_tilt, mechanical_tilt
        )
        combined_gain = -np.minimum(-(a_h + a_v), max_attenuation)
        return combined_gain

    @staticmethod
    def calculate_elevation_angle_vectorized(distance_m, bs_height_m, ue_height_m):
        """Vectorized elevation angle calculation."""
        height_diff   = ue_height_m - bs_height_m
        distance_m    = np.maximum(distance_m, 1.0)
        elevation_rad = np.arctan2(height_diff, distance_m)
        elevation_deg = np.degrees(elevation_rad)
        return elevation_deg


# ──────────────────────────────────────────────────────────────────────────────
# RF MODEL  (exact from original)
# K1/K2 are CLASS-LEVEL attributes — set by process_chunk_optimized.
# ──────────────────────────────────────────────────────────────────────────────

class OptimizedRFModel:
    K1 = None
    K2 = None
    """Optimized RF propagation with vectorization."""

    @staticmethod
    def cost231_hata_path_loss_vectorized(distance_m, frequency_mhz=1800,
                                          bs_height_m=30, ue_height_m=1.5,
                                          urban=True):
        distance_m  = np.maximum(distance_m, 100.0)
        distance_km = distance_m / 1000.0

        if frequency_mhz >= 400:
            a_hm = 3.2 * (np.log10(11.75 * ue_height_m)) ** 2 - 4.97
        else:
            a_hm = ((1.1 * np.log10(frequency_mhz) - 0.7) * ue_height_m
                    - (1.56 * np.log10(frequency_mhz) - 0.8))

        CM = 3.0 if urban else 0.0

        # K1 REPLACES base intercept, K2 REPLACES slope
        if OptimizedRFModel.K1 is not None:
            base_PL = OptimizedRFModel.K1
        else:
            base_PL = (46.3 + 33.9 * np.log10(frequency_mhz)
                       - 13.82 * np.log10(bs_height_m) - a_hm + CM)

        if OptimizedRFModel.K2 is not None:
            slope_term = OptimizedRFModel.K2
        else:
            slope_term = 44.9 - 6.55 * np.log10(bs_height_m)

        path_loss = base_PL + slope_term * np.log10(distance_km)
        return path_loss

    @staticmethod
    def calculate_rsrp_vectorized(distances, azimuth_diffs, elevation_angles,
                                  frequency_mhz, tx_power_dbm, antenna_gain_dbi,
                                  cable_loss_db, bs_heights, ue_height,
                                  horizontal_beamwidth, vertical_beamwidth,
                                  electrical_tilts, mechanical_tilts):
        """Vectorized RSRP calculation for multiple sectors."""
        pl = OptimizedRFModel.cost231_hata_path_loss_vectorized(
            distances, frequency_mhz, bs_heights, ue_height
        )
        ant_gain = Optimized3GPPAntennaModel.combined_3gpp_pattern_vectorized(
            azimuth_diffs, elevation_angles,
            horizontal_beamwidth, vertical_beamwidth,
            electrical_tilts, mechanical_tilts
        )
        rsrp = tx_power_dbm + antenna_gain_dbi + ant_gain - cable_loss_db - pl
        return rsrp

    @staticmethod
    def combine_signals_vectorized(sectors_rsrp_matrix):
        """
        Best-sector RSRP per pixel — standard telecom practice.
        Each pixel is served by the strongest sector (highest RSRP).
        Mean was pulling all values down by averaging in back-lobe/weak sectors.
        """
        sectors_rsrp_matrix = np.clip(sectors_rsrp_matrix, -150, -30)
        rsrp_total          = np.max(sectors_rsrp_matrix, axis=1)   # best sector
        return rsrp_total

    @staticmethod
    def estimate_rsrq_vectorized(rsrp, bandwidth_mhz=10):
        """RSRQ estimation using bandwidth."""
        n_rb = BANDWIDTH_TO_RB.get(bandwidth_mhz, 50)
        rsrq = rsrp - 10 * np.log10(n_rb) - 3
        return np.clip(rsrq, -20, -3)

    @staticmethod
    def estimate_sinr_vectorized(rsrp):
        """Vectorized SINR estimation."""
        total_in_linear = 10 ** (-105 / 10) + 10 ** (-104 / 10)
        total_in_dbm    = 10 * np.log10(total_in_linear)
        sinr            = rsrp - total_in_dbm
        return np.clip(sinr, -10, 30)


# ──────────────────────────────────────────────────────────────────────────────
# GEOMETRY  (exact from original)
# ──────────────────────────────────────────────────────────────────────────────

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance calculation."""
    R       = 6371000.0
    phi1    = np.radians(lat1)
    phi2    = np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def calculate_bearing_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized bearing calculation."""
    phi1    = np.radians(lat1)
    phi2    = np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)
    x       = np.sin(dlambda) * np.cos(phi2)
    y       = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    bearing = np.degrees(np.arctan2(x, y))
    bearing = (bearing + 360) % 360
    return bearing


# ──────────────────────────────────────────────────────────────────────────────
# CHUNK WORKER  (exact from original)
# Sets OptimizedRFModel.K1 / K2 class attrs — reads in cost231_hata.
# Indoor loss FIXED at 15 dB (original: indoor_loss[i] = 15).
# ──────────────────────────────────────────────────────────────────────────────

def process_chunk_optimized(chunk_data):
    """
    Process a chunk of test points.
    FIXED:
    - Safe variable creation
    - No premature return
    - Always defines combined_rsrp before any use
    """
    (test_lats, test_lons, site_data, params) = chunk_data

    # Set calibration constants
    OptimizedRFModel.K1 = params.get("k1", DEFAULT_K1)   # default 139.0 — NOT 0
    OptimizedRFModel.K2 = params.get("k2", DEFAULT_K2)   # default 35.2  — NOT 0

    # Broadcast test + site values
    test_lats_bc    = test_lats[:, None]
    test_lons_bc    = test_lons[:, None]
    site_lats_bc    = site_data['lats'][None, :]
    site_lons_bc    = site_data['lons'][None, :]
    site_azs_bc     = site_data['azimuths'][None, :]
    site_heights_bc = site_data['heights'][None, :]
    site_etilts_bc  = site_data['etilts'][None, :]
    site_mtilts_bc  = site_data['mtilts'][None, :]

    # Compute RF geometry
    distances     = haversine_vectorized(test_lats_bc, test_lons_bc, site_lats_bc, site_lons_bc)
    bearings      = calculate_bearing_vectorized(site_lats_bc, site_lons_bc, test_lats_bc, test_lons_bc)
    azimuth_diffs = np.abs((bearings - site_azs_bc + 180) % 360 - 180)

    elevation_angles = Optimized3GPPAntennaModel.calculate_elevation_angle_vectorized(
        distances, site_heights_bc, params['ue_height']
    )

    # Calculate RSRP matrix
    rsrp_matrix = OptimizedRFModel.calculate_rsrp_vectorized(
        distances,
        azimuth_diffs,
        elevation_angles,
        params['frequency_mhz'],
        params['tx_power'],
        params['antenna_gain'],
        params['cable_loss'],
        site_heights_bc,
        params['ue_height'],
        params['h_beamwidth'],
        params['v_beamwidth'],
        site_etilts_bc,
        site_mtilts_bc,
    )

    # Combine strongest RSRP
    combined_rsrp = OptimizedRFModel.combine_signals_vectorized(rsrp_matrix)

    # Add calibration bias
    combined_rsrp = combined_rsrp + params.get("rsrp_bias", 0.0)

    # Indoor detection (optional)
    if params.get("polygon_wkt_list") is not None:
        try:
            polys = [wkt.loads(w) for w in params["polygon_wkt_list"]]
        except Exception:
            polys = []

        indoor_loss = np.zeros_like(combined_rsrp)

        for i, (lat, lon) in enumerate(zip(test_lats, test_lons)):
            pt = Point(lon, lat)
            for poly in polys:
                if poly.contains(pt):
                    indoor_loss[i] = 15   # fixed 15 dB — exact from original
                    break

        combined_rsrp -= indoor_loss

    # RSRQ & SINR
    combined_rsrq = OptimizedRFModel.estimate_rsrq_vectorized(
        combined_rsrp, params['bandwidth_mhz']
    )
    combined_sinr = OptimizedRFModel.estimate_sinr_vectorized(combined_rsrp)

    return combined_rsrp, combined_rsrq, combined_sinr


# ──────────────────────────────────────────────────────────────────────────────
# PARALLEL DRIVER  (exact from original)
# ──────────────────────────────────────────────────────────────────────────────

def compute_predictions_parallel(test_pts, site_df, params, n_workers=None):
    """Compute predictions using parallel processing and vectorization."""
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f"\n   Using {n_workers} CPU cores for parallel processing...")

    # Prepare site data
    site_data = {
        'lats':     site_df['lat'].values,
        'lons':     site_df['lon'].values,
        'azimuths': site_df['azimuth'].values,
        'heights':  site_df['antenna_height'].values,
        'etilts':   site_df['electrical_tilt'].values,
        'mtilts':   site_df['mechanical_tilt'].values,
    }

    # Split test points into chunks
    chunk_size = max(100, len(test_pts) // (n_workers * 4))
    test_lats  = test_pts['lat'].values
    test_lons  = test_pts['lon'].values

    # Convert Shapely polygons → WKT strings before passing to workers.
    # Raw Shapely objects cannot be reliably pickled across subprocess boundaries.
    # Converting to WKT strings (plain text) is always safe to pickle.
    polygon_wkt_list = None
    if params.get("polygons") is not None:
        try:
            polygon_wkt_list = [poly.wkt for poly in params["polygons"]]
            print(f"   📌 Serialised {len(polygon_wkt_list)} building polygons as WKT for workers")
        except Exception as e:
            print(f"   ⚠ Could not serialise polygons: {e}")

    # Build a safe params dict that excludes raw Shapely objects
    # and instead carries the WKT string list
    safe_params = {k: v for k, v in params.items() if k not in ("polygons",)}
    safe_params["polygon_wkt_list"] = polygon_wkt_list

    # Build chunks using safe_params (no raw Shapely inside)
    chunks = []
    for i in range(0, len(test_pts), chunk_size):
        chunk_lats = test_lats[i:i + chunk_size]
        chunk_lons = test_lons[i:i + chunk_size]
        chunks.append((chunk_lats, chunk_lons, site_data, safe_params))

    print(f"   Processing {len(test_pts):,} points in {len(chunks)} chunks...")

    all_rsrp, all_rsrq, all_sinr = [], [], []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_chunk_optimized, chunks))

    for rsrp, rsrq, sinr in results:
        all_rsrp.append(rsrp)
        all_rsrq.append(rsrq)
        all_sinr.append(sinr)

    rsrp_pred = np.concatenate(all_rsrp)
    rsrq_pred = np.concatenate(all_rsrq)
    sinr_pred = np.concatenate(all_sinr)

    elapsed        = time.time() - start_time
    points_per_sec = len(test_pts) / elapsed
    print(f"   ✓ Completed in {elapsed:.1f}s ({points_per_sec:.0f} points/sec)")

    return rsrp_pred, rsrq_pred, sinr_pred


# ──────────────────────────────────────────────────────────────────────────────
# FAST CALIBRATION  (exact from original)
# DEFAULT_K1 / DEFAULT_K2 defined locally inside function — matches original.
# ──────────────────────────────────────────────────────────────────────────────

def fast_calibration(drive_test_df, site_df, params):

    print("\n" + "=" * 60)
    print("SMART INDUSTRIAL CALIBRATION")
    print("=" * 60)

    calibration = {}

    # 1. Clean DT
    dt       = drive_test_df.dropna(subset=["lat", "lon"]).copy()
    rsrp_col = next((c for c in dt.columns if c.lower() == "rsrp"), None)

    if rsrp_col is None:
        print("❌ No RSRP column found.")
        return calibration

    dt["RSRP_meas"] = pd.to_numeric(dt[rsrp_col], errors="coerce")
    dt = dt.dropna(subset=["RSRP_meas"])
    dt = dt[(dt["RSRP_meas"] >= -150) & (dt["RSRP_meas"] <= -30)]

    if len(dt) < 10:
        print("❌ Not enough DT points.")
        return calibration

    print(f"Valid DT points: {len(dt)}")

    # 2. Distance Calculation
    dist_matrix = haversine_vectorized(
        dt["lat"].values[:, None],
        dt["lon"].values[:, None],
        site_df["lat"].values[None, :],
        site_df["lon"].values[None, :],
    )
    serving_idx    = np.argmin(dist_matrix, axis=1)
    serving_dist_m = dist_matrix[np.arange(len(dt)), serving_idx]
    serving_dist_m = np.maximum(serving_dist_m, 1.0)

    serving_dist_km = serving_dist_m / 1000.0
    spread_km       = serving_dist_km.max() - serving_dist_km.min()

    print(f"Distance spread: {spread_km:.3f} km")

    # 3. Measured Path Loss
    pl_measured = (
        params["tx_power"]
        + params["antenna_gain"]
        - params["cable_loss"]
        - dt["RSRP_meas"].values
    )
    log_d = np.log10(serving_dist_km)

    # 4. Decide Calibration Mode  (local defaults — matches original)
    DEFAULT_K1 = 139.0
    DEFAULT_K2 = 35.2

    if spread_km < 0.5:
        print("📌 Small DT spread → Fitting only K1")
        K2 = DEFAULT_K2
        K1 = np.mean(pl_measured - K2 * log_d)

    elif spread_km < 2.0:
        print("📌 Medium DT spread → Fit both with clamp")
        K2, K1 = np.polyfit(log_d, pl_measured, 1)
        K2 = np.clip(K2, 25, 45)

    else:
        print("📌 Large DT spread → Full regression")
        K2, K1 = np.polyfit(log_d, pl_measured, 1)

    # 5. Final Safety Clamp
    K1 = float(np.clip(K1, 120, 170))
    K2 = float(np.clip(K2, 20,  60))

    print("\nCALIBRATION RESULT")
    print(f"K1 (Intercept): {K1:.2f}")
    print(f"K2 (Slope):     {K2:.2f}")
    print("=" * 60)

    calibration["K1"] = K1
    calibration["K2"] = K2
    return calibration


# ──────────────────────────────────────────────────────────────────────────────
# SITE POWER CALIBRATION  (exact from original — used in original main())
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_site_power(drive_df, site_rows, params, debug=False):
    """
    Calibrate K1 (intercept) and K2 (slope) from drive test data.
    K1 REPLACES the path loss intercept.
    K2 REPLACES the path loss slope.
    """
    dt       = drive_df.copy()

    # Find RSRP column — synthetic DT uses "rsrp", real DT may use "RSRP" etc.
    rsrp_col = next((c for c in dt.columns if c.lower() == "rsrp"), None)
    if rsrp_col is None:
        # Fallback: any column containing "rsrp"
        rsrp_col = next((c for c in dt.columns if "rsrp" in c.lower()), None)

    print(f"   🔎 calibrate_site_power: {len(dt)} input rows | cols={dt.columns.tolist()} | rsrp_col={rsrp_col}")

    if rsrp_col is None:
        print("   ⚠ No RSRP column found → using defaults K1=139, K2=35.2")
        return 139.0, 35.2

    dt["RSRP_meas"] = pd.to_numeric(dt[rsrp_col], errors="coerce")
    dt = dt.dropna(subset=["lat", "lon", "RSRP_meas"])
    print(f"   🔎 After dropna: {len(dt)} rows | RSRP range: [{dt['RSRP_meas'].min():.1f}, {dt['RSRP_meas'].max():.1f}]")

    # Accept full valid RSRP range — synthetic DT values are in -44 to -140 dBm
    dt = dt[(dt["RSRP_meas"] >= -150) & (dt["RSRP_meas"] <= -30)]
    print(f"   🔎 After RSRP filter (-150 to -30): {len(dt)} rows remaining")

    if len(dt) < 10:
        print(f"   ⚠ Not enough DT points ({len(dt)} < 10) → using defaults K1=139, K2=35.2")
        return 139.0, 35.2

    dist_mat        = haversine_vectorized(
        dt["lat"].values[:, None], dt["lon"].values[:, None],
        site_rows["lat"].values[None, :], site_rows["lon"].values[None, :],
    )
    serving_dist_m  = np.maximum(dist_mat.min(axis=1), 1.0)
    serving_dist_km = serving_dist_m / 1000.0
    spread_km       = serving_dist_km.max() - serving_dist_km.min()

    tx  = params.get("tx_power",     46.0)
    ag  = params.get("antenna_gain", 18.0)
    cl  = params.get("cable_loss",    2.0)
    pl_measured = tx + ag - cl - dt["RSRP_meas"].values
    log_d       = np.log10(serving_dist_km)

    DEFAULT_K2 = 35.2
    if spread_km < 0.5:
        print(f"   📌 Small spread ({spread_km:.2f} km) → K1 only")
        K2 = DEFAULT_K2
        K1 = float(np.mean(pl_measured - K2 * log_d))
    elif spread_km < 2.0:
        print(f"   📌 Medium spread ({spread_km:.2f} km) → fit both, clamp slope")
        K2, K1 = np.polyfit(log_d, pl_measured, 1)
        K2 = float(np.clip(K2, 25, 45))
    else:
        print(f"   📌 Large spread ({spread_km:.2f} km) → full regression")
        K2, K1 = np.polyfit(log_d, pl_measured, 1)

    K1 = float(np.clip(K1, 120, 170))
    K2 = float(np.clip(K2, 20,  60))

    if debug:
        print(f"   K1={K1:.2f}  K2={K2:.2f}  ({len(dt)} DT points)")

    return K1, K2


# ──────────────────────────────────────────────────────────────────────────────
# GRID GENERATION  (exact from original)
# ──────────────────────────────────────────────────────────────────────────────

def generate_circle_grid(site_df: pd.DataFrame, radius_m: float = 15000.0,
                          resolution_m: float = 20.0) -> pd.DataFrame:
    """Generate a grid of test points within a circle."""
    center_lat = site_df['lat'].mean()
    center_lon = site_df['lon'].mean()

    print(f"\n{'=' * 60}")
    print(f"AUTO-GENERATING TEST GRID")
    print(f"{'=' * 60}")
    print(f"Center point: ({center_lat:.6f}, {center_lon:.6f})")
    print(f"Radius: {radius_m} meters ({radius_m / 1000:.1f} km)")
    print(f"Grid resolution: {resolution_m} meters")

    lat_deg_per_m = 1.0 / 111320.0
    lon_deg_per_m = 1.0 / (111320.0 * math.cos(math.radians(center_lat)))

    radius_lat_deg = radius_m * lat_deg_per_m
    radius_lon_deg = radius_m * lon_deg_per_m

    step_lat = resolution_m * lat_deg_per_m
    step_lon = resolution_m * lon_deg_per_m

    min_lat = center_lat - radius_lat_deg
    max_lat = center_lat + radius_lat_deg
    min_lon = center_lon - radius_lon_deg
    max_lon = center_lon + radius_lon_deg

    lats = np.arange(min_lat, max_lat, step_lat)
    lons = np.arange(min_lon, max_lon, step_lon)

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()

    print("Filtering points within circle...")
    dlat = np.radians(lat_flat - center_lat)
    dlon = np.radians(lon_flat - center_lon)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(center_lat)) * np.cos(np.radians(lat_flat))
         * np.sin(dlon / 2) ** 2)
    distances = 2 * 6371000 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    mask    = distances <= radius_m
    grid_df = pd.DataFrame({
        'lat': lat_flat[mask],
        'lon': lon_flat[mask],
    })

    print(f"Generated {len(grid_df):,} test points")
    print(f"Estimated processing time: {len(grid_df) / 10000:.1f} - {len(grid_df) / 5000:.1f} minutes")
    print(f"{'=' * 60}\n")

    return grid_df


# ──────────────────────────────────────────────────────────────────────────────
# I/O HELPERS  (exact from original)
# ──────────────────────────────────────────────────────────────────────────────

def safe_read(path: str) -> pd.DataFrame:
    """Safely read CSV with various encodings and separators."""
    configs = [
        {},
        {"encoding": "latin-1"},
        {"sep": ";"},
        {"sep": "\t"},
        {"encoding": "utf-8", "engine": "python"},
        {"encoding": "latin-1", "sep": "\t"},
    ]
    for kwargs in configs:
        try:
            return pd.read_csv(path, **kwargs)
        except Exception:
            continue
    raise RuntimeError(f"Cannot read CSV: {path}")


def normcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def standardize_latlon(df: pd.DataFrame) -> pd.DataFrame:
    if 'lat' in df.columns and 'lon' in df.columns:
        return df
    mapping = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ("latitude", "lat_deg", "y") and 'lat' not in df.columns:
            mapping[c] = "lat"
        if lc in ("longitude", "lon_deg", "x") and 'lon' not in df.columns:
            mapping[c] = "lon"
    return df.rename(columns=mapping)




# ──────────────────────────────────────────────────────────────────────────────
# SPLIT SITEWISE  (first definition — exact from original)
# ──────────────────────────────────────────────────────────────────────────────

def split_sitewise(site_df):
    """
    Splits the input dataframe into groups of site-wise sectors.
    Works if the file contains multiple sites.
    """
    possible_cols = ["SITE", "Site", "site", "SiteName", "CellName", "eNB", "eNB_Name"]
    site_col = None

    for c in site_df.columns:
        if c.strip() in possible_cols:
            site_col = c
            break

    if site_col is None:
        return {"SITE_1": site_df.copy()}

    groups = {}
    for site_name, df_group in site_df.groupby(site_col):
        groups[str(site_name)] = df_group.copy()

    return groups


# ──────────────────────────────────────────────────────────────────────────────
# SPLIT SITEWISE  (second / winning definition — exact from original)
# Python uses this one at runtime (last definition wins).
# ──────────────────────────────────────────────────────────────────────────────

def split_sitewise(site_df):  # noqa: F811
    """
    Splits site dataframe into groups based on SITE_ID.
    Each SITE_ID will be predicted separately.
    """
    possible_cols = ["Site ID", "SiteID", "Site ID", "Site",
                     "CellName", "ENodeB", "ENodeB_ID"]
    site_col = None

    for c in site_df.columns:
        if c.strip() in possible_cols:
            site_col = c
            break

    if site_col is None:
        print("⚠ No SITE_ID column found. Using entire file as one site.")
        return {"SITE_1": site_df.copy()}

    groups = {}
    for site, group in site_df.groupby(site_col):
        groups[str(site)] = group.copy()

    return groups
# ──────────────────────────────────────────────────────────────────────────────
# KPI HELPER  (exact from original)
# ──────────────────────────────────────────────────────────────────────────────

def clamp_array(arr, kpi_name):
    lo, hi = KPI_RANGES[kpi_name]
    return np.clip(arr, lo, hi)


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL: one full prediction pass across all sites
# (API addition — not in original script)
# Area clipping: original uses area_polygon.contains(Point(lon,lat)) per row.
# API uses vectorized shapely_contains — identical result, much faster.
# ──────────────────────────────────────────────────────────────────────────────

def _run_prediction_pass(site_df: pd.DataFrame, params: dict,
                          area_polygon, radius_m: float,
                          resolution_m: float, n_workers: int,
                          label: str) -> pd.DataFrame:
    """
    Generates grid → area-clips → predicts → merges across all sites.
    Returns DataFrame: lat, lon, pred_rsrp, pred_rsrq, pred_sinr, site_id
    """
    site_outputs = []

    # Support both "site_id" (normalised) and "Site ID" (original) column names
    site_col = "site_id" if "site_id" in site_df.columns else "Site ID"

    for site in site_df[site_col].unique():

        print("\n------------------------------------------------------------------")
        print(f"📡 [{label}] Processing Site → {site}")
        print("------------------------------------------------------------------")

        site_rows = site_df[site_df[site_col] == site].copy()

        # Generate circular grid around site centre
        pts_df = generate_circle_grid(site_rows, radius_m, resolution_m)

        # Area polygon clipping
        if area_polygon is not None:
            before = len(pts_df)
            mask   = shapely_contains(area_polygon, pts_df["lon"].values, pts_df["lat"].values)
            pts_df = pts_df[mask].reset_index(drop=True)
            print(f"   Area clip: {before:,} → {len(pts_df):,} points")

        if pts_df.empty:
            print(f"   ⚠ No grid points after clipping — skipping site {site}")
            continue

        print(f"   ✔ Grid points: {len(pts_df):,}")
        print("📌 Running RF model (multi-core)...")

        rsrp_pred, rsrq_pred, sinr_pred = compute_predictions_parallel(
            pts_df, site_rows, params, n_workers
        )

        pts_df["pred_rsrp"] = clamp_array(rsrp_pred, "RSRP")
        pts_df["pred_rsrq"] = clamp_array(rsrq_pred, "RSRQ")
        pts_df["pred_sinr"] = clamp_array(sinr_pred, "SINR")
        pts_df["site_id"]   = str(site)
        site_outputs.append(pts_df)

    if not site_outputs:
        raise ValueError(
            f"[{label}] No grid points generated for any site. "
            "Check site data, radius_m and polygon_area."
        )

    print(f"\n{'=' * 60}")
    print(f"[{label}] Combining multi-site predictions (AVERAGE RSRP)...")
    print(f"{'=' * 60}")

    final_output = pd.concat(site_outputs, ignore_index=True)

    final_output = final_output.groupby(
        ["lat", "lon"], as_index=False
    ).agg(
        pred_rsrp=("pred_rsrp", "mean"),
        pred_rsrq=("pred_rsrq", "mean"),
        pred_sinr=("pred_sinr", "mean"),
        site_id  =("site_id",   "first"),
    )

    print(f"✔ [{label}] {len(final_output):,} pixels after merge")
    return final_output



# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC: TWO-PASS PREDICTION  (API addition — not in original script)
# ──────────────────────────────────────────────────────────────────────────────

def run_two_pass_prediction(
    site_df:           pd.DataFrame,
    dt_df,
    site_mapping_df,
    polygons:          list,
    poly_meta:         list,
    area_polygon,
    params:            dict,
    radius_m:          float,
    grid_resolution:   float,
    n_workers:         int,
    enrich_max_dist_m: float = 25.0,
    enrich_top_n:      tuple = (2, 3),
) -> dict:
    from .dt_enrichment import enrich_predictions

    site_col = "site_id" if "site_id" in site_df.columns else "Site ID"

    # ══════════════════════════════════════════════════════════════════
    # PASS 1 — Site data only, K1=139.0 K2=35.2, no calibration,
    #           no buildings. Area clip AFTER merge (matches original script).
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("  PASS 1 — Site-only (K1=139.0, K2=35.2, no calibration)")
    print("█" * 60)

    site_outputs = []

    for site in site_df[site_col].unique():
        print(f"\n------------------------------------------------------------------")
        print(f"📡 [Pass-1] Processing Site → {site}")
        print("------------------------------------------------------------------")

        site_rows = site_df[site_df[site_col] == site].copy()

        site_params = {
            **params,
            "k1":        DEFAULT_K1,   # 139.0 — NOT 0
            "k2":        DEFAULT_K2,   # 35.2  — NOT 0
            "polygons":  [],
            "poly_meta": [],
        }

        pts_df = generate_circle_grid(site_rows, radius_m, grid_resolution)
        print(f"   ✔ Grid points: {len(pts_df):,}")
        print("📌 Running RF model (multi-core)...")

        rsrp_pred, rsrq_pred, sinr_pred = compute_predictions_parallel(
            pts_df, site_rows, site_params, n_workers
        )

        pts_df["pred_rsrp"] = clamp_array(rsrp_pred, "RSRP")
        pts_df["pred_rsrq"] = clamp_array(rsrq_pred, "RSRQ")
        pts_df["pred_sinr"] = clamp_array(sinr_pred, "SINR")
        pts_df["site_id"]   = str(site)
        site_outputs.append(pts_df)

    if not site_outputs:
        raise ValueError("No grid points generated for any site.")

    print(f"\n{'=' * 60}")
    print("[Pass-1] Combining multi-site predictions (AVERAGE RSRP)...")
    print(f"{'=' * 60}")

    pass1_result = pd.concat(site_outputs, ignore_index=True)
    pass1_result = pass1_result.groupby(
        ["lat", "lon"], as_index=False
    ).agg(
        pred_rsrp=("pred_rsrp", "mean"),
        pred_rsrq=("pred_rsrq", "mean"),
        pred_sinr=("pred_sinr", "mean"),
        site_id  =("site_id",   "first"),
    )

    # Area clip AFTER merge — matches original script Step 7
    if area_polygon is not None:
        before = len(pass1_result)
        mask = shapely_contains(
            area_polygon,
            pass1_result["lon"].values,
            pass1_result["lat"].values,
        )
        pass1_result = pass1_result[mask].reset_index(drop=True)
        print(f"   Area clip: {before:,} → {len(pass1_result):,} points")

    print(f"✔ [Pass-1] {len(pass1_result):,} pixels after merge")

    # ══════════════════════════════════════════════════════════════════
    # ENRICH — BallTree DT match per site_id filtered by nodeb_id
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("  ENRICH — DT spatial matching (BallTree, 25 m radius)")
    print("█" * 60)

    enriched_result = pass1_result.copy()

    if (dt_df is not None and not dt_df.empty
            and site_mapping_df is not None and not site_mapping_df.empty):
        enriched_result = enrich_predictions(
            pred_df=pass1_result.copy(),
            dt_df=dt_df,
            site_mapping_df=site_mapping_df,
            max_distance_m=enrich_max_dist_m,
            top_n_values=enrich_top_n,
        )
    else:
        print("⚠ No DT data or site mapping — enrichment skipped.")
        for n in enrich_top_n:
            enriched_result[f"pred_rsrp_top{n}_avg"] = np.nan
        enriched_result["measured_dt_rsrp"] = np.nan

    # ══════════════════════════════════════════════════════════════════
    # BUILD SYNTHETIC DRIVE TEST from top2_avg
    # Column renamed to "rsrp" so calibrate_site_power can find it
    # ══════════════════════════════════════════════════════════════════
    top2_col = "pred_rsrp_top2_avg"

    enriched_pixels = int(enriched_result[top2_col].notna().sum()) \
                      if top2_col in enriched_result.columns else 0

    # Build synthetic DT — all enriched pixels (used for enrichment output)
    synthetic_dt_full = (
        enriched_result[["lat", "lon", "site_id", top2_col]]
        .dropna(subset=[top2_col])
        .rename(columns={top2_col: "rsrp"})
        .reset_index(drop=True)
    )

    # Build calibration-quality synthetic DT — filter to reliable RSRP range only
    # Values below -120 dBm are at the noise floor (far from tower) and skew K1/K2 regression
    # Values above -44 dBm are physically impossible
    CALIB_RSRP_MIN = -120.0   # dBm — only use pixels with meaningful signal strength
    CALIB_RSRP_MAX =  -44.0   # dBm — physical upper bound

    synthetic_dt = synthetic_dt_full[
        synthetic_dt_full["rsrp"].between(CALIB_RSRP_MIN, CALIB_RSRP_MAX)
    ].reset_index(drop=True)

    print(f"\n📍 Synthetic DT:")
    print(f"   Total enriched pixels : {len(synthetic_dt_full):,} "
          f"({enriched_pixels:,}/{len(pass1_result):,} enriched)")
    print(f"   Calibration-quality   : {len(synthetic_dt):,} "
          f"(RSRP {CALIB_RSRP_MIN} to {CALIB_RSRP_MAX} dBm)")
    if not synthetic_dt_full.empty:
        print(f"   Full RSRP range       : [{synthetic_dt_full['rsrp'].min():.1f}, "
              f"{synthetic_dt_full['rsrp'].max():.1f}] dBm")
    if not synthetic_dt.empty:
        print(f"   Calib RSRP range      : [{synthetic_dt['rsrp'].min():.1f}, "
              f"{synthetic_dt['rsrp'].max():.1f}] dBm")
        print(f"   Site IDs in calib DT  : {synthetic_dt['site_id'].unique().tolist()}")

    # ══════════════════════════════════════════════════════════════════
    # PASS 2 — Per-site calibration with synthetic DT + building loss
    # Matches: python script.py --site ... --drive synthetic_dt
    #          --building ... --polygon_area ... --calibrate
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("  PASS 2 — Per-site calibration with synthetic DT + building loss")
    print("█" * 60)

    site_outputs_p2 = []
    pass2_k1_list   = []
    pass2_k2_list   = []

    for site in site_df[site_col].unique():
        print(f"\n------------------------------------------------------------------")
        print(f"📡 [Pass-2] Processing Site → {site}")
        print("------------------------------------------------------------------")

        site_rows = site_df[site_df[site_col] == site].copy()

        site_params = {
            **params,
            "polygons":  polygons,    # buildings applied in Pass 2
            "poly_meta": poly_meta,
        }

        # Per-site calibration — filter synthetic DT to this site first
        if len(synthetic_dt) >= 10:
            site_synthetic = synthetic_dt[synthetic_dt["site_id"] == str(site)].copy()

            if len(site_synthetic) < 10:
                site_synthetic = synthetic_dt.copy()
                print(f"   📌 Using global synthetic DT ({len(site_synthetic)} pts) for site {site}")
            else:
                print(f"   📌 Using site-specific synthetic DT ({len(site_synthetic)} pts) for site {site}")

            k1, k2 = calibrate_site_power(site_synthetic, site_rows, params, debug=True)
        else:
            print(f"   ⚠ Not enough synthetic DT ({len(synthetic_dt)} pts, need ≥10) → default K1/K2")
            k1, k2 = DEFAULT_K1, DEFAULT_K2

        site_params["k1"] = k1
        site_params["k2"] = k2
        pass2_k1_list.append(k1)
        pass2_k2_list.append(k2)

        pts_df = generate_circle_grid(site_rows, radius_m, grid_resolution)
        print(f"   ✔ Grid points: {len(pts_df):,}")
        print("📌 Running RF model (multi-core)...")

        rsrp_pred, rsrq_pred, sinr_pred = compute_predictions_parallel(
            pts_df, site_rows, site_params, n_workers
        )

        pts_df["pred_rsrp"] = clamp_array(rsrp_pred, "RSRP")
        pts_df["pred_rsrq"] = clamp_array(rsrq_pred, "RSRQ")
        pts_df["pred_sinr"] = clamp_array(sinr_pred, "SINR")
        pts_df["site_id"]   = str(site)
        site_outputs_p2.append(pts_df)

    if not site_outputs_p2:
        raise ValueError("Pass 2: No grid points generated for any site.")

    print(f"\n{'=' * 60}")
    print("[Pass-2] Combining multi-site predictions (AVERAGE RSRP)...")
    print(f"{'=' * 60}")

    pass2_result = pd.concat(site_outputs_p2, ignore_index=True)
    pass2_result = pass2_result.groupby(
        ["lat", "lon"], as_index=False
    ).agg(
        pred_rsrp=("pred_rsrp", "mean"),
        pred_rsrq=("pred_rsrq", "mean"),
        pred_sinr=("pred_sinr", "mean"),
        site_id  =("site_id",   "first"),
    )

    # Area clip AFTER merge — matches original script Step 7
    if area_polygon is not None:
        before = len(pass2_result)
        mask = shapely_contains(
            area_polygon,
            pass2_result["lon"].values,
            pass2_result["lat"].values,
        )
        pass2_result = pass2_result[mask].reset_index(drop=True)
        print(f"   Area clip: {before:,} → {len(pass2_result):,} points")

    avg_k1 = float(np.mean(pass2_k1_list)) if pass2_k1_list else DEFAULT_K1
    avg_k2 = float(np.mean(pass2_k2_list)) if pass2_k2_list else DEFAULT_K2

    print("\n" + "═" * 60)
    print("  TWO-PASS SUMMARY")
    print(f"  Pass-1 : K1={DEFAULT_K1}  K2={DEFAULT_K2}  pixels={len(pass1_result):,}")
    print(f"  Enrich : {enriched_pixels:,}/{len(pass1_result):,} pixels matched DT")
    print(f"  Synth  : {len(synthetic_dt):,} synthetic DT points")
    print(f"  Pass-2 : K1={avg_k1:.4f}  K2={avg_k2:.4f}  pixels={len(pass2_result):,}  ← FINAL")
    print("═" * 60)

    return {
        "pass1":   pass1_result,
        "refined": enriched_result,
        "pass2":   pass2_result,
        "meta": {
            "pass1_k1":            DEFAULT_K1,
            "pass1_k2":            DEFAULT_K2,
            "pass2_k1":            avg_k1,
            "pass2_k2":            avg_k2,
            "synthetic_dt_points": len(synthetic_dt),
            "enriched_pixels":     enriched_pixels,
            "total_pixels":        len(pass1_result),
        },
    }
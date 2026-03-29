"""
dt_enrichment.py  —  Drive-Test Spatial Enrichment
===================================================
EXACT port of the original standalone enrichment script.

Original script logic (copied exactly):
────────────────────────────────────────
for site_id, site_preds in pred_df.groupby("site_id"):
    valid_nodes = site_df[site_df["site_id"] == site_id]["nodeb_id"].unique()
    site_dt = dt_df[dt_df["nodeb_id"].isin(valid_nodes)].copy()
    
    tree = BallTree(dt_coords, metric='haversine')
    k_neighbors = min(3, len(site_dt))
    distances, indices = tree.query(pred_coords, k=k_neighbors)
    distances_m = distances * 6371000
    
    for i, orig_index in enumerate(site_preds.index):
        valid_mask = distances_m[i] <= MAX_DISTANCE_METERS
        if np.sum(valid_mask) == 0: continue
        valid_rsrp = site_dt.iloc[idxs[valid_mask]]["rsrp"].values
        if len(valid_rsrp) >= 2: top2_avg = mean(valid_rsrp[:2])
        if len(valid_rsrp) >= 3: top3_avg = mean(valid_rsrp[:3])

Key differences from previous version:
  • Uses site_dt.iloc[indices[valid_mask]] — positional indexing, requires reset_index()
  • k_neighbors = min(3, len(site_dt)) — hardcoded 3, not max(top_n_values)
  • measured_dt_rsrp = valid_rsrp[0] (nearest single point) — kept for DB schema
  • top2_avg only written if >= 2 valid points (same as original)
  • top3_avg only written if >= 3 valid points (same as original)
  • NO top2 written for single-match pixels (matches original exactly)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_M  = 6_371_000.0
MAX_DISTANCE_M  = 25.0          # fixed — matches original MAX_DISTANCE_METERS = 25


def enrich_predictions(
    pred_df:         pd.DataFrame,
    dt_df:           pd.DataFrame,
    site_mapping_df: pd.DataFrame,
    max_distance_m:  float = 25.0,
    top_n_values:    tuple[int, ...] = (2, 3),
) -> pd.DataFrame:
    """
    Parameters
    ----------
    pred_df         : Pass-1 grid  — needs lat, lon, site_id
    dt_df           : merged DT logs — needs lat, lon, rsrp, nodeb_id
    site_mapping_df : site_id → nodeb_id mapping  (from site_prediction table)
    max_distance_m  : BallTree match radius in metres (default 25 — matches original)
    top_n_values    : which top-N averages to compute (default (2, 3))

    Returns
    -------
    pred_df with added columns:
        pred_rsrp_top2_avg   — mean RSRP of nearest 2 DT points within radius
        pred_rsrp_top3_avg   — mean RSRP of nearest 3 DT points within radius
        measured_dt_rsrp     — RSRP of single nearest DT point (added for DB schema)

    Pixels with no DT match within radius keep NaN.
    Only written if enough neighbours exist (top2 needs ≥2, top3 needs ≥3).
    """
    # ── Normalize column names ────────────────────────────────────────────────
    pred_df         = _normcols(pred_df.copy())
    dt_df           = _normcols(dt_df.copy())
    site_mapping_df = _normcols(site_mapping_df.copy())

    _require(pred_df,         ["lat", "lon", "site_id"],          "pred_df")
    _require(dt_df,           ["lat", "lon", "rsrp", "nodeb_id"], "dt_df")
    _require(site_mapping_df, ["site_id", "nodeb_id"],            "site_mapping_df")

    # ── Cast all join keys to str — prevents type-mismatch silent failures ───
    pred_df["site_id"]          = pred_df["site_id"].astype(str)
    dt_df["nodeb_id"]           = dt_df["nodeb_id"].astype(str)
    site_mapping_df["site_id"]  = site_mapping_df["site_id"].astype(str)
    site_mapping_df["nodeb_id"] = site_mapping_df["nodeb_id"].astype(str)
    has_operator_dimension = "operator" in pred_df.columns and "operator" in site_mapping_df.columns
    if has_operator_dimension:
        pred_df["operator"] = pred_df["operator"].astype(str)
        site_mapping_df["operator"] = site_mapping_df["operator"].astype(str)

    # ── Clean DT ─────────────────────────────────────────────────────────────
    dt_df = dt_df.dropna(subset=["lat", "lon", "rsrp", "nodeb_id"])
    dt_df["rsrp"] = pd.to_numeric(dt_df["rsrp"], errors="coerce")
    dt_df = dt_df.dropna(subset=["rsrp"])

    if dt_df.empty:
        print("⚠ dt_enrichment: no valid DT rows — enrichment skipped.")
        # Still initialise output columns
        for n in top_n_values:
            pred_df[f"pred_rsrp_top{n}_avg"] = np.nan
        pred_df["measured_dt_rsrp"] = np.nan
        return pred_df

    print(f"\n🔍 DT Enrichment — {len(dt_df):,} DT rows | "
          f"radius={max_distance_m} m | top_n={top_n_values}")

    # ── Initialise output columns ─────────────────────────────────────────────
    for n in top_n_values:
        pred_df[f"pred_rsrp_top{n}_avg"] = np.nan
    pred_df["measured_dt_rsrp"] = np.nan

    enriched = 0

    group_cols = ["site_id"]
    if has_operator_dimension:
        group_cols = ["operator", "site_id"]

    for group_key, site_preds in pred_df.groupby(group_cols):
        if has_operator_dimension:
            operator, site_id = group_key
        else:
            operator, site_id = None, group_key

        # ── site_id → valid nodeb_id(s) ──────────────────────────────────────
        mapping_mask = site_mapping_df["site_id"] == site_id
        if has_operator_dimension:
            mapping_mask &= site_mapping_df["operator"] == operator

        valid_nodes = site_mapping_df.loc[
            mapping_mask, "nodeb_id"
        ].unique()

        if len(valid_nodes) == 0:
            print(f"   ⚠ No nodeb_id mapping for site_id={site_id} — skipped.")
            continue

        # ── Filter DT to this site's NodeBs only ─────────────────────────────
        site_dt = dt_df[dt_df["nodeb_id"].isin(valid_nodes)].copy()

        if site_dt.empty:
            continue

        # ── CRITICAL: reset_index so iloc[...] positional indexing is safe ───
        # Original: site_dt.iloc[idxs[valid_mask]] — requires 0-based integer index
        # Without reset_index(), iloc positions and BallTree positions can diverge
        site_dt = site_dt.reset_index(drop=True)

        # ── Build BallTree — haversine needs radians ──────────────────────────
        dt_coords   = np.radians(site_dt[["lat", "lon"]].values)
        pred_coords = np.radians(site_preds[["lat", "lon"]].values)

        # k_neighbors = min(3, len(site_dt)) — exact from original
        k_neighbors = min(3, len(site_dt))
        tree = BallTree(dt_coords, metric="haversine")
        dist_rad, indices = tree.query(pred_coords, k=k_neighbors)
        dist_m = dist_rad * EARTH_RADIUS_M   # convert radians → metres

        # ── Per-pixel matching — exact logic from original ────────────────────
        for i, orig_idx in enumerate(site_preds.index):
            dists = dist_m[i]
            idxs  = indices[i]

            valid_mask = dists <= max_distance_m

            if np.sum(valid_mask) == 0:
                continue

            # site_dt.iloc[idxs[valid_mask]] — exact from original
            valid_rsrp = site_dt.iloc[idxs[valid_mask]]["rsrp"].values
            enriched  += 1

            # Nearest single DT point (for DB schema — not in original script)
            pred_df.at[orig_idx, "measured_dt_rsrp"] = float(valid_rsrp[0])

            # Top2 — only if >= 2 valid points (exact from original)
            if len(valid_rsrp) >= 2:
                pred_df.at[orig_idx, "pred_rsrp_top2_avg"] = float(
                    np.mean(valid_rsrp[:2])
                )

            # Top3 — only if >= 3 valid points (exact from original)
            if len(valid_rsrp) >= 3:
                pred_df.at[orig_idx, "pred_rsrp_top3_avg"] = float(
                    np.mean(valid_rsrp[:3])
                )

    pct = enriched / len(pred_df) * 100 if len(pred_df) else 0.0
    print(f"   ✅ {enriched:,}/{len(pred_df):,} pixels enriched ({pct:.1f}%)")
    return pred_df


# ── helpers ──────────────────────────────────────────────────────────────────

def _normcols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip, lowercase, spaces→underscore."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def _require(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"dt_enrichment '{name}' missing required columns: {missing}. "
            f"Got: {df.columns.tolist()}"
        )

"""
services.py  —  LTE Prediction Service

All RF parameters (frequency_mhz, tx_power, antenna_gain, cable_loss,
ue_height, h_beamwidth, v_beamwidth, bandwidth_mhz) are fixed constants
defined in _DEFAULT_RF at the top of this file.
tbl_project_config does not exist in this database.

The pipeline always runs as two-pass:

    Pass 1  → site data only, default K1/K2, no calibration
              → tbl_lte_prediction_results  (intermediate, replaced by Pass 2)

    Enrich  → BallTree match against tbl_network_log + tbl_network_log_neighbour
              filtered per site_id by nodeb_id (from site_prediction)
              → tbl_lte_prediction_results_refined
                 (lat, lon, site_id, top2_avg, top3_avg, measured_dt_rsrp)

    Pass 2  → top2_avg pixels as synthetic DT → recalibrate K1/K2
              → re-predict with building polygons + area polygon
              → tbl_lte_prediction_results  (FINAL — what UI reads)

DB table schemas expected
─────────────────────────
site_prediction
    tbl_project_id, latitude, longitude, azimuth, height,
    e_tilt, m_tilt, site  (mapped → Site ID)

tbl_network_log / tbl_network_log_neighbour
    session_id, lat, lon, rsrp, nodeb_id

site_prediction
    tbl_project_id, site_id, nodeb_id

tbl_savepolygon
    project_id, region (WKT)

tbl_lte_prediction_results
    id, project_id, job_id, lat, lon,
    pred_rsrp, pred_rsrq, pred_sinr, site_id, created_at

tbl_lte_prediction_results_refined
    id, project_id, job_id, lat, lon, site_id,
    pred_rsrp_top2_avg, pred_rsrp_top3_avg, measured_dt_rsrp, created_at
"""

import uuid
import time
import threading
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text, bindparam
from shapely.geometry import Polygon
from shapely import wkt
from flask import current_app

from extensions import db
from .ml_engine import run_two_pass_prediction

JOBS: dict = {}

# Default RF constants used when tbl_project_config has no row
_DEFAULT_RF = {
    "frequency_mhz": 1800.0,
    "tx_power":      46.0,
    "antenna_gain":  18.0,
    "cable_loss":    2.0,
    "ue_height":     1.5,
    "h_beamwidth":   65.0,
    "v_beamwidth":   10.0,
    "bandwidth_mhz": 10.0,
}


class LTEPredictionService:

    # ─────────────────────────────────────────────
    # PUBLIC: SUBMIT
    # ─────────────────────────────────────────────

    def submit(self, cfg: dict) -> dict:
        """
        Register a new job and launch it in a background thread.
        Returns immediately with { job_id, status: "queued" }.
        """
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {
            "job_id":       job_id,
            "status":       "queued",
            "progress":     "Queued",
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        app = current_app._get_current_object()
        threading.Thread(
            target=self._worker,
            args=(app, job_id, cfg),
            daemon=True,
        ).start()
        return {"job_id": job_id, "status": "queued"}

    # ─────────────────────────────────────────────
    # PUBLIC: GET JOB
    # ─────────────────────────────────────────────

    def get(self, job_id: str) -> dict | None:
        return JOBS.get(job_id)

    # ─────────────────────────────────────────────
    # WORKER
    # ─────────────────────────────────────────────

    def _worker(self, app, job_id: str, cfg: dict):
        with app.app_context():
            try:
                self._run(job_id, cfg)
            except Exception as exc:
                JOBS[job_id]["status"]    = "failed"
                JOBS[job_id]["error"]     = str(exc)
                JOBS[job_id]["traceback"] = traceback.format_exc()
                print(f"\n❌ Job {job_id} failed:\n{traceback.format_exc()}")

    # ─────────────────────────────────────────────
    # MAIN PIPELINE
    # ─────────────────────────────────────────────

    def _run(self, job_id: str, cfg: dict):
        pid         = cfg["project_id"]
        session_ids = cfg["session_ids"]      # already a list of ints

        # ── 1. Load RF parameters from DB ────────────────────────────
        self._update(job_id, "running", "Loading project RF parameters")
        rf_params = self._load_rf_params(pid)

        # ── 2. Load site data ─────────────────────────────────────────
        self._update(job_id, "running", "Loading site data")
        site_df = self._load_site(pid)

        # ── 3. Load drive-test logs (for enrichment BallTree only) ────
        self._update(job_id, "running", "Loading drive-test logs")
        dt_df = self._load_drive_test(session_ids)

        # ── 4. Load site_id → nodeb_id mapping ───────────────────────
        self._update(job_id, "running", "Loading site → NodeB mapping")
        site_mapping_df = self._load_site_mapping(pid)

        if site_mapping_df.empty:
            raise ValueError(
                f"No rows found in site_prediction for project_id={pid}. "
                "Two-pass prediction requires a site_id → nodeb_id mapping."
            )

        # ── 5. Load building polygons (applied in Pass 2 only) ────────
        polygons, poly_meta = [], []
        if cfg.get("building"):
            self._update(job_id, "running", "Loading building polygons")
            polygons, poly_meta = self._load_polygons(pid)

        # ── 6. Parse area polygon ─────────────────────────────────────
        area_polygon = self._parse_area_polygon(cfg.get("polygon_area"))

        # ── 7. Run the two-pass engine ────────────────────────────────
        self._update(job_id, "running", "Pass 1 — site-only prediction")

        result = run_two_pass_prediction(
            site_df=site_df,
            dt_df=dt_df,
            site_mapping_df=site_mapping_df,
            polygons=polygons,
            poly_meta=poly_meta,
            area_polygon=area_polygon,
            params=rf_params,
            radius_m=cfg["radius_m"],
            grid_resolution=cfg["grid_resolution"],
            n_workers=cfg.get("n_workers", 4),
            enrich_max_dist_m=25.0,    # fixed 25 m — matches standalone script
            enrich_top_n=(2, 3),       # always compute top2 and top3
        )

        # ── 8. Save enriched/refined results ─────────────────────────
        self._update(job_id, "running",
                     "Saving enriched results → tbl_lte_prediction_results_refined")
        self._save_refined_results(result["refined"], pid, job_id)

        # ── 9. Save final Pass-2 results ──────────────────────────────
        self._update(job_id, "running",
                     "Saving final results → tbl_lte_prediction_results")
        self._save_main_results(result["pass2"], pid, job_id)

        # ── 10. Record metadata for /result response ──────────────────
        meta = result["meta"]
        JOBS[job_id].update({
            "pass1_k1":            meta["pass1_k1"],
            "pass1_k2":            meta["pass1_k2"],
            "pass2_k1":            meta["pass2_k1"],
            "pass2_k2":            meta["pass2_k2"],
            "synthetic_dt_points": meta["synthetic_dt_points"],
            "enriched_pixels":     meta["enriched_pixels"],
            "total_pixels":        meta["total_pixels"],
        })

        self._update(job_id, "done", "Completed")

    # ─────────────────────────────────────────────
    # DB WRITERS
    # ─────────────────────────────────────────────

    @staticmethod
    def _save_main_results(df: pd.DataFrame, pid: int, job_id: str):
        """
        Insert into tbl_lte_prediction_results.
        Schema: project_id, job_id, lat, lon, pred_rsrp, pred_rsrq, pred_sinr,
                site_id, created_at
        """
        out = df[["lat", "lon", "pred_rsrp", "pred_rsrq", "pred_sinr", "site_id"]].copy()
        out["project_id"] = pid
        out["job_id"]     = job_id
        out["created_at"] = datetime.now(timezone.utc)

        out = out[[
            "project_id", "job_id", "lat", "lon",
            "pred_rsrp", "pred_rsrq", "pred_sinr",
            "site_id", "created_at",
        ]]

        out.to_sql(
            "tbl_lte_prediction_results",
            db.engine,
            if_exists="append",
            index=False,
            method="multi",
        )
        print(f"   💾 {len(out):,} rows → tbl_lte_prediction_results")

    @staticmethod
    def _save_refined_results(df: pd.DataFrame, pid: int, job_id: str):
        """
        Insert enriched rows into tbl_lte_prediction_results_refined.
        Schema: project_id, job_id, lat, lon, site_id,
                pred_rsrp_top2_avg, pred_rsrp_top3_avg,
                measured_dt_rsrp, created_at

        Only pixels with at least one non-NaN enrichment value are saved.
        """
        enrich_cols = [
            "pred_rsrp_top2_avg",
            "pred_rsrp_top3_avg",
            "measured_dt_rsrp",
        ]

        # Ensure all expected columns exist (fill any missing ones with NaN)
        for col in enrich_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Keep only pixels that were actually enriched
        has_data = df[enrich_cols].notna().any(axis=1)
        refined  = df.loc[has_data, ["lat", "lon", "site_id"] + enrich_cols].copy()

        if refined.empty:
            print("   ⚠ No enriched pixels to save — refined table not written.")
            return

        refined["project_id"] = pid
        refined["job_id"]     = job_id
        refined["created_at"] = datetime.now(timezone.utc)

        refined = refined[[
            "project_id", "job_id", "lat", "lon", "site_id",
            "pred_rsrp_top2_avg", "pred_rsrp_top3_avg",
            "measured_dt_rsrp", "created_at",
        ]]

        refined.to_sql(
            "tbl_lte_prediction_results_refined",
            db.engine,
            if_exists="append",
            index=False,
            method="multi",
        )
        print(f"   💾 {len(refined):,} rows → tbl_lte_prediction_results_refined")

    # ─────────────────────────────────────────────
    # DATA LOADERS
    # ─────────────────────────────────────────────

    @staticmethod
    def _load_rf_params(pid: int) -> dict:
        """
        Return fixed RF parameters from _DEFAULT_RF.

        tbl_project_config does not exist in this database.
        All RF constants are defined in _DEFAULT_RF at the top of this file.
        To change them, update _DEFAULT_RF — no DB migration needed.

        Current defaults:
            frequency_mhz : 1800.0  MHz
            tx_power      : 46.0    dBm
            antenna_gain  : 18.0    dBi
            cable_loss    : 2.0     dB
            ue_height     : 1.5     m
            h_beamwidth   : 65.0    degrees
            v_beamwidth   : 10.0    degrees
            bandwidth_mhz : 10.0    MHz
        """
        params = dict(_DEFAULT_RF)
        print(f"   RF params (defaults): {params}")
        return params

    def _load_site(self, pid: int) -> pd.DataFrame:
        """Load and normalise site antenna data."""
        df = pd.read_sql(
            text("SELECT * FROM site_prediction WHERE tbl_project_id = :pid"),
            db.engine,
            params={"pid": pid},
        )

        if df.empty:
            raise ValueError(f"No site data found in site_prediction for project_id={pid}")

        df = df.rename(columns={
            "latitude":  "lat",
            "longitude": "lon",
            "height":    "antenna_height",
            "e_tilt":    "electrical_tilt",
            "m_tilt":    "mechanical_tilt",
            "site":      "site_id",   # normalised lowercase — matches pred_df and site_mapping_df
        })

        # Catch any legacy "Site ID" variant
        if "Site ID" in df.columns and "site_id" not in df.columns:
            df = df.rename(columns={"Site ID": "site_id"})

        # ── Numeric coercion with sensible defaults ──────────────────────────
        df["electrical_tilt"] = pd.to_numeric(df.get("electrical_tilt"), errors="coerce").fillna(3.0)
        df["mechanical_tilt"] = pd.to_numeric(df.get("mechanical_tilt"), errors="coerce").fillna(0.0)
        df["antenna_height"]  = pd.to_numeric(df.get("antenna_height"),  errors="coerce").fillna(30.0)
        df["azimuth"]         = pd.to_numeric(df.get("azimuth"),         errors="coerce").fillna(0.0)
        df["lat"]             = pd.to_numeric(df.get("lat"),             errors="coerce")
        df["lon"]             = pd.to_numeric(df.get("lon"),             errors="coerce")

        # ── Drop rows with null lat/lon — cannot predict without coordinates ─
        before = len(df)
        df = df.dropna(subset=["lat", "lon"])
        if len(df) < before:
            print(f"   ⚠ Dropped {before - len(df)} site rows with null lat/lon")

        # ── Sanity-check coordinate range (Delhi ~28°N, 77°E) ────────────────
        valid = (
            df["lat"].between(-90, 90) &
            df["lon"].between(-180, 180)
        )
        if not valid.all():
            print(f"   ⚠ Dropped {(~valid).sum()} site rows with out-of-range lat/lon")
            df = df[valid]

        # ── Clamp tilt & height to realistic values ───────────────────────────
        df["electrical_tilt"] = df["electrical_tilt"].clip(-20, 20)
        df["mechanical_tilt"] = df["mechanical_tilt"].clip(-20, 20)
        df["antenna_height"]  = df["antenna_height"].clip(1, 200)
        df["azimuth"]         = df["azimuth"].clip(0, 360)

        if df.empty:
            raise ValueError(f"All site rows invalid after cleaning for project_id={pid}")

        # Cast to str so it matches site_prediction.site_id values
        df["site_id"] = df["site_id"].astype(str).str.strip()
        df["site_id"] = df["site_id"].replace("", "UNKNOWN")

        print(f"   Site data: {len(df)} sectors across {df['site_id'].nunique()} sites")
        return df

    @staticmethod
    def _load_drive_test(session_ids: list) -> pd.DataFrame | None:
        """
        Load and merge tbl_network_log + tbl_network_log_neighbour
        for all given session_ids.

        Only the columns needed for BallTree enrichment are fetched:
            lat, lon, rsrp, nodeb_id
        """
        stmt_serving = text(
            "SELECT lat, lon, rsrp, nodeb_id "
            "FROM tbl_network_log "
            "WHERE session_id IN :ids"
        ).bindparams(bindparam("ids", expanding=True))

        stmt_neighbour = text(
            "SELECT lat, lon, rsrp, nodeb_id "
            "FROM tbl_network_log_neighbour "
            "WHERE session_id IN :ids"
        ).bindparams(bindparam("ids", expanding=True))

        dt_serving   = pd.read_sql(stmt_serving,   db.engine, params={"ids": session_ids})
        dt_neighbour = pd.read_sql(stmt_neighbour, db.engine, params={"ids": session_ids})

        raw = pd.concat([dt_serving, dt_neighbour], ignore_index=True)
        print(f"   Drive-test raw: {len(raw):,} rows "
              f"({len(dt_serving):,} serving + {len(dt_neighbour):,} neighbour)")

        # ── Coerce all columns to correct types ───────────────────────────────
        raw["lat"]      = pd.to_numeric(raw["lat"],  errors="coerce")
        raw["lon"]      = pd.to_numeric(raw["lon"],  errors="coerce")
        raw["rsrp"]     = pd.to_numeric(raw["rsrp"], errors="coerce")
        raw["nodeb_id"] = raw["nodeb_id"].astype(str).str.strip()

        # ── Drop nulls ────────────────────────────────────────────────────────
        before = len(raw)
        raw = raw.dropna(subset=["lat", "lon", "rsrp", "nodeb_id"])
        if len(raw) < before:
            print(f"   ⚠ Dropped {before - len(raw)} rows with null lat/lon/rsrp/nodeb_id")

        # ── Drop empty-string nodeb_id ─────────────────────────────────────────
        before = len(raw)
        raw = raw[raw["nodeb_id"].str.len() > 0]
        raw = raw[raw["nodeb_id"] != "nan"]
        if len(raw) < before:
            print(f"   ⚠ Dropped {before - len(raw)} rows with empty/nan nodeb_id")

        # ── Valid coordinate range ─────────────────────────────────────────────
        before = len(raw)
        raw = raw[raw["lat"].between(-90, 90) & raw["lon"].between(-180, 180)]
        if len(raw) < before:
            print(f"   ⚠ Dropped {before - len(raw)} rows with invalid coordinates")

        # ── Valid RSRP range: -44 to -140 dBm ─────────────────────────────────
        before = len(raw)
        raw = raw[raw["rsrp"].between(-140, -44)]
        if len(raw) < before:
            print(f"   ⚠ Dropped {before - len(raw)} rows with out-of-range RSRP "
                  f"(kept -140 to -44 dBm)")

        # ── Deduplicate ────────────────────────────────────────────────────────
        before = len(raw)
        raw = raw.drop_duplicates().reset_index(drop=True)
        if len(raw) < before:
            print(f"   ⚠ Dropped {before - len(raw)} duplicate rows")

        merged = raw
        print(f"   ✔ Drive-test after cleaning: {len(merged):,} valid rows "
              f"| nodeb_ids: {merged['nodeb_id'].nunique()} unique")

        return merged if not merged.empty else None

    @staticmethod
    def _load_site_mapping(pid: int) -> pd.DataFrame:
        """
        Load site_id → nodeb_id mapping from site_prediction.
        Equivalent of Mapped_Physical_with_PCI_Final.csv in the standalone script.
        """
        df = pd.read_sql(
            text(
                "SELECT site, nodeb_id "
                "FROM site_prediction "
                "WHERE tbl_project_id = :pid"
            ),
            db.engine,
            params={"pid": pid},
        )
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # ── Cast + clean ───────────────────────────────────────────────────────
        df["site_id"]  = df["site"].astype(str).str.strip()
        df["nodeb_id"] = df["nodeb_id"].astype(str).str.strip()

        # Drop rows where site_id or nodeb_id is null/empty/nan
        before = len(df)
        df = df[
            (df["site_id"].str.len() > 0) & (df["site_id"] != "nan") &
            (df["nodeb_id"].str.len() > 0) & (df["nodeb_id"] != "nan")
        ]
        df = df.dropna(subset=["site_id", "nodeb_id"])
        if len(df) < before:
            print(f"   ⚠ Dropped {before - len(df)} site-mapping rows with null/empty values")

        df = df.drop_duplicates(subset=["site_id", "nodeb_id"]).reset_index(drop=True)

        print(f"   Site mapping: {len(df)} rows, {df['site_id'].nunique()} unique sites, "
              f"{df['nodeb_id'].nunique()} unique nodeb_ids")
        return df

    @staticmethod
    def _load_polygons(pid: int) -> tuple:
        """
        Load building footprint polygons from tbl_savepolygon.
        Returns (polygons, poly_meta) lists.
        """
        rows = db.session.execute(
            text("SELECT region FROM tbl_savepolygon WHERE project_id = :pid"),
            {"pid": pid},
        ).fetchall()

        polygons, poly_meta, skipped = [], [], 0

        for r in rows:
            raw = r[0]
            if not raw:
                skipped += 1
                continue
            try:
                poly = wkt.loads(raw)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_valid and not poly.is_empty:
                    polygons.append(poly)
                    poly_meta.append({"loss": 15.0})
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        print(f"   Building polygons: {len(polygons)} loaded, {skipped} skipped")
        return polygons, poly_meta

    @staticmethod
    def _parse_area_polygon(polygon_area) -> Polygon | None:
        """
        Convert [[lon, lat], …] list from the API request into a Shapely Polygon.
        Returns None if polygon_area is absent or invalid.
        """
        if not polygon_area:
            return None
        try:
            poly = Polygon(polygon_area)
            if not poly.is_valid:
                poly = poly.buffer(0)
            return poly if poly.is_valid and not poly.is_empty else None
        except Exception as exc:
            print(f"   ⚠ Could not parse polygon_area: {exc}")
            return None

    # ─────────────────────────────────────────────
    # HELPER
    # ─────────────────────────────────────────────

    @staticmethod
    def _update(job_id: str, status: str, progress: str):
        JOBS[job_id]["status"]   = status
        JOBS[job_id]["progress"] = progress
        print(f"   [{job_id[:8]}] {status} — {progress}")
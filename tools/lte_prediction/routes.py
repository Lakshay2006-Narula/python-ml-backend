"""
routes.py  —  LTE Prediction API Blueprint

Input contract (from frontend / Postman):
─────────────────────────────────────────
{
    "project_id":   167,
    "session_ids":  [3767],
    "grid_value":   25.0,          // grid resolution in metres (optional, default 25)
    "radius_m":     5000.0,        // prediction radius in metres (optional, default 5000)
    "polygon_area": [[lon, lat], …],  // clipping polygon (optional)
    "building":     true           // apply building penetration loss (optional, default false)
}

All RF parameters (tx_power, antenna_gain, frequency_mhz, etc.) are loaded
from the database based on project_id — they are NOT accepted from the request.

The pipeline always runs as two-pass:
    Pass 1  → site data only, no calibration
    Enrich  → DT spatial match per nodeb_id
    Pass 2  → synthetic DT calibration → final prediction
"""

from flask import Blueprint, request, jsonify
from .services import LTEPredictionService

lte_prediction_bp = Blueprint("lte_prediction", __name__)
svc = LTEPredictionService()


def to_bool(val, default: bool) -> bool:
    """Safe bool coercion — avoids bool('false') == True trap."""
    if val is None:           return default
    if isinstance(val, bool): return val
    if isinstance(val, int):  return val != 0
    if isinstance(val, str):  return val.strip().lower() not in ("false", "0", "no", "off")
    return bool(val)


# ─────────────────────────────────────────────────────────────────────────────
# POST /run
# ─────────────────────────────────────────────────────────────────────────────

@lte_prediction_bp.route("/run", methods=["POST"])
def run_prediction():
    """
    Submit a two-pass LTE prediction job.

    Required
    --------
    project_id      int             e.g. 167
    session_ids     list[int|str]   e.g. [3767]  — non-empty

    Optional
    --------
    grid_value      float   25.0    grid resolution in metres
    radius_m        float   5000.0  prediction radius in metres
    polygon_area    list    null    [[lon, lat], …]  area clipping polygon
    building        bool    false   apply building penetration loss in Pass 2
    n_workers       int     4       parallel CPU workers

    Returns
    -------
    202  { job_id, status: "queued" }
    400  { error: "..." }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # ── Required fields ──────────────────────────────────────────────
    if "project_id" not in data:
        return jsonify({"error": "'project_id' is required"}), 400

    if "session_ids" not in data:
        return jsonify({"error": "'session_ids' is required"}), 400

    if not isinstance(data["session_ids"], list) or len(data["session_ids"]) == 0:
        return jsonify({"error": "'session_ids' must be a non-empty list"}), 400

    # ── Validate polygon_area if provided ────────────────────────────
    polygon_area = data.get("polygon_area")
    if polygon_area is not None:
        if not isinstance(polygon_area, list) or len(polygon_area) < 3:
            return jsonify({
                "error": "'polygon_area' must be a list of at least 3 [lon, lat] pairs"
            }), 400

    # ── Build job config ─────────────────────────────────────────────
    cfg = {
        "project_id":  int(data["project_id"]),
        "session_ids": [int(s) for s in data["session_ids"]],

        # Grid settings
        "grid_resolution": float(data.get("grid_value", 25.0)),
        "radius_m":        float(data.get("radius_m",   5000.0)),

        # Area clipping polygon — list of [lon, lat] pairs or None
        "polygon_area":    polygon_area,

        # Building penetration loss in Pass 2
        "building":        to_bool(data.get("building"), False),

        # Performance
        "n_workers":       int(data.get("n_workers", 4)),
    }

    result = svc.submit(cfg)
    return jsonify(result), 202


# ─────────────────────────────────────────────────────────────────────────────
# GET /status/<job_id>
# ─────────────────────────────────────────────────────────────────────────────

@lte_prediction_bp.route("/status/<job_id>", methods=["GET"])
def job_status(job_id):
    """
    Poll job status.

    200  {
           job_id, status, progress, submitted_at
         }
    404  { error: "Job not found" }
    """
    job = svc.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job), 200


# ─────────────────────────────────────────────────────────────────────────────
# GET /result/<job_id>
# ─────────────────────────────────────────────────────────────────────────────

@lte_prediction_bp.route("/result/<job_id>", methods=["GET"])
def job_result(job_id):
    """
    Fetch completed job result summary.

    200 (success) {
      "message":  "Prediction complete",
      "job_id":   "...",
      "status":   "done",
      "pipeline": {
        "pass1": {
          "k1": 139.0, "k2": 35.2,
          "pixels": 48320,
          "note": "site-only, default constants, no calibration"
        },
        "enrichment": {
          "pixels_enriched": 3120,
          "total_pixels":    48320,
          "enrichment_pct":  6.5,
          "output_table":    "tbl_lte_prediction_results_refined"
        },
        "synthetic_dt": {
          "points_used": 3120
        },
        "pass2": {
          "k1": 141.23, "k2": 36.10,
          "output_table": "tbl_lte_prediction_results",
          "note": "FINAL — building loss applied, shown in UI"
        }
      }
    }

    409  { error: "Job not ready yet", status, progress }
    500  { error: "Job failed",        detail }
    404  { error: "Job not found" }
    """
    job = svc.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job["status"] == "failed":
        return jsonify({
            "error":  "Job failed",
            "detail": job.get("error", "unknown error"),
            "job_id": job_id,
            "status": "failed",
        }), 500

    if job["status"] != "done":
        return jsonify({
            "error":    "Job not ready yet",
            "status":   job["status"],
            "progress": job.get("progress", ""),
        }), 409

    total  = job.get("total_pixels", 0)
    enrich = job.get("enriched_pixels", 0)

    return jsonify({
        "message": "Prediction complete",
        "job_id":  job_id,
        "status":  "done",
        "pipeline": {
            "pass1": {
                "k1":    round(job.get("pass1_k1", 139.0), 4),
                "k2":    round(job.get("pass1_k2", 35.2),  4),
                "pixels": total,
                "note":  "site-only prediction, default constants, no calibration",
            },
            "enrichment": {
                "pixels_enriched": enrich,
                "total_pixels":    total,
                "enrichment_pct":  round(enrich / total * 100, 1) if total else 0,
                "output_table":    "tbl_lte_prediction_results_refined",
                "columns": [
                    "lat", "lon", "site_id",
                    "pred_rsrp_top2_avg", "pred_rsrp_top3_avg", "measured_dt_rsrp"
                ],
            },
            "synthetic_dt": {
                "points_used": job.get("synthetic_dt_points", 0),
                "source":      "pred_rsrp_top2_avg from enriched pixels",
            },
            "pass2": {
                "k1":           round(job.get("pass2_k1", 139.0), 4),
                "k2":           round(job.get("pass2_k2", 35.2),  4),
                "output_table": "tbl_lte_prediction_results",
                "note":         "FINAL — refined calibration + building loss, shown in UI",
            },
        },
    }), 200
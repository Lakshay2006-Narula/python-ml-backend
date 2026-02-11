"""Test case: generate handover analysis map and a one-page PDF.

This script is independent of the full pipeline. It:
- Reads required columns from `defaultdb.tbl_network_log` using `src.db.get_connection()`.
- Detects provider changes from `m_alpha_long` per `session_id`.
- Renders a folium HTML map with route, colored samples per provider, and "spark" SVG markers at provider-change points.
- Converts HTML to PNG via `src.playwright_utils.html_to_png` and saves to `data/images/kpi_maps/handover_map.png`.
- Builds a single-page PDF `data/processed/handover_map_report.pdf` embedding the PNG.

Usage:
    python test_cases/test_handover_map.py

Environment:
    Ensure DB env vars are set (DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME).
    Ensure Playwright is installed and browsers are configured when converting to PNG.
"""

import os
from pathlib import Path
import pandas as pd

from src.load_data_db import load_project_data
from src.map_generator import generate_distinct_colors, add_legend, get_df_bounds, merge_bounds, expand_bounds, get_polygon_bounds

try:
    from src.playwright_utils import html_to_png
except Exception:
    html_to_png = None

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from folium import Map, PolyLine, CircleMarker, Marker, Polygon
from folium import DivIcon


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_PNG = BASE_DIR / "data" / "images" / "kpi_maps" / "handover_map.png"
TMP_HTML = BASE_DIR / "data" / "tmp" / "handover_map.html"
OUT_PDF = BASE_DIR / "data" / "processed" / "handover_map_report.pdf"

# When True, also run project-level (time-ordered) detection which may include
# transitions at session boundaries (this reproduces the previous behavior).
# Set to False to only detect intra-session handovers.
USE_GLOBAL_DETECTION = True


def ensure_dirs():
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    TMP_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)


def fetch_network_logs(project_id=149):
    """Load project data using the same loader as the main pipeline and return filtered_df and project metadata."""
    raw_df, filtered_df, project_meta = load_project_data(project_id)
    return filtered_df, project_meta


def detect_handover_events(df: pd.DataFrame):
    events = []
    if df.empty:
        return events
    df = df.rename(columns={c: c.lower() for c in df.columns})

    # detect which column holds technology (some datasets use `network`)
    tech_col = None
    for cand in ("network", "technology", "tech"):
        if cand in df.columns:
            tech_col = cand
            break

    required = {"timestamp", "lat", "lon", "m_alpha_long"}
    if tech_col:
        required.add(tech_col)

    if not required.issubset(set(df.columns)):
        print("Required columns missing from query result")
        return events

    # drop rows missing spatial or provider/tech info
    drop_subset = ["lat", "lon", "m_alpha_long"]
    if tech_col:
        drop_subset.append(tech_col)
    df = df.dropna(subset=drop_subset)

    # 1) Try per-session detection if session_id exists
    # Use run-length compression: mark only when the new provider run is stable
    # (min_run_length samples). This avoids noisy alternations.
    MIN_RUN_LENGTH = 10
    if "session_id" in df.columns:
        df_s = df.sort_values(["session_id", "timestamp"]) 
        for sid, group in df_s.groupby("session_id"):
            # runs store ((provider, tech), first_row, count)
            runs = []
            prev = None
            count = 0
            first_row = None
            for _, row in group.iterrows():
                prov = str(row["m_alpha_long"]).strip()
                tech = str(row[tech_col]).strip() if tech_col else None
                key = (prov, tech)
                if prev is None:
                    prev = key
                    count = 1
                    first_row = row
                    continue
                if key == prev:
                    count += 1
                else:
                    runs.append((prev, first_row, count))
                    prev = key
                    count = 1
                    first_row = row
            # append last run
            if prev is not None:
                runs.append((prev, first_row, count))
            # mark boundaries where next run length >= MIN_RUN_LENGTH
            for i in range(len(runs) - 1):
                (cur_prov, cur_tech), cur_row, cur_cnt = runs[i]
                (next_prov, next_tech), next_row, next_cnt = runs[i + 1]
                # handover if provider OR technology changed
                if (next_prov != cur_prov or next_tech != cur_tech) and next_cnt >= MIN_RUN_LENGTH:
                    events.append({
                        "session_id": int(sid),
                        "timestamp": next_row["timestamp"],
                        "lat": float(next_row["lat"]),
                        "lon": float(next_row["lon"]),
                        "from_provider": cur_prov,
                        "to_provider": next_prov,
                        "from_network": cur_tech,
                        "to_network": next_tech,
                    })

            # Debug: print per-session summary when verbose
            if len(runs) > 1:
                print(f"Session {sid} runs: {[ (p,t,c) for (p,t),_,c in runs ]}")

        # When `session_id` exists we only perform per-session detection.
        # If no events found using MIN_RUN_LENGTH, relax rule to detect any change (min_run=1)
        if not events:
            print("No events found with MIN_RUN_LENGTH=3; running relaxed per-session detection (min_run=1)")
            events_relaxed = []
            for sid, group in df_s.groupby("session_id"):
                prev = None
                for _, row in group.iterrows():
                    prov = str(row["m_alpha_long"]).strip()
                    tech = str(row[tech_col]).strip() if tech_col else None
                    key = (prov, tech)
                    if prev is None:
                        prev = key
                        continue
                    if key != prev:
                        events_relaxed.append({
                            "session_id": int(sid),
                            "timestamp": row["timestamp"],
                            "lat": float(row["lat"]),
                            "lon": float(row["lon"]),
                            "from_provider": prev[0],
                            "to_provider": prov,
                            "from_network": prev[1],
                            "to_network": tech,
                        })
                        prev = key
                
                if events_relaxed:
                    print(f"Detected {len(events_relaxed)} relaxed events (min_run=1)")
                    events = events_relaxed

            # If we don't want global/project-level detection, return per-session results
            if not USE_GLOBAL_DETECTION:
                # But add inter-session handovers: sort sessions by earliest timestamp,
                # and mark handover at start of later session if provider/tech differs from previous.
                session_summaries = []
                for sid, group in df_s.groupby("session_id"):
                    if group.empty:
                        continue
                    earliest = group["timestamp"].min()
                    # Dominant provider/tech: most frequent
                    prov_counts = group["m_alpha_long"].astype(str).value_counts()
                    dominant_prov = prov_counts.idxmax()
                    tech_counts = group[tech_col].astype(str).value_counts() if tech_col else None
                    dominant_tech = tech_counts.idxmax() if tech_counts is not None else None
                    first_row = group.iloc[0]  # first row for location
                    session_summaries.append({
                        "session_id": sid,
                        "earliest_ts": earliest,
                        "dominant_prov": dominant_prov,
                        "dominant_tech": dominant_tech,
                        "first_row": first_row
                    })
                # Sort by earliest timestamp
                session_summaries.sort(key=lambda x: x["earliest_ts"])
                # Check consecutive sessions
                for i in range(1, len(session_summaries)):
                    prev = session_summaries[i-1]
                    curr = session_summaries[i]
                    if (prev["dominant_prov"] != curr["dominant_prov"] or
                        (tech_col and prev["dominant_tech"] != curr["dominant_tech"])):
                        # Handover: use first row of current session
                        events.append({
                            "session_id": int(curr["session_id"]),
                            "timestamp": curr["first_row"]["timestamp"],
                            "lat": float(curr["first_row"]["lat"]),
                            "lon": float(curr["first_row"]["lon"]),
                            "from_provider": prev["dominant_prov"],
                            "to_provider": curr["dominant_prov"],
                            "from_network": prev["dominant_tech"],
                            "to_network": curr["dominant_tech"],
                        })
                # Otherwise, continue to perform global detection and merge results below

    # If we reach here there was no session_id column — perform global detection
    # Global run-length compression across time-ordered samples
    df_g = df.sort_values(["timestamp"])  # time-ordered
    runs = []
    prev = None
    count = 0
    first_row = None
    for _, row in df_g.iterrows():
        prov = str(row["m_alpha_long"]).strip()
        tech = str(row[tech_col]).strip() if tech_col else None
        key = (prov, tech)
        if prev is None:
            prev = key
            count = 1
            first_row = row
            continue
        if key == prev:
            count += 1
        else:
            runs.append((prev, first_row, count))
            prev = key
            count = 1
            first_row = row
    if prev is not None:
        runs.append((prev, first_row, count))

    for i in range(len(runs) - 1):
        (cur_prov, cur_tech), cur_row, cur_cnt = runs[i]
        (next_prov, next_tech), next_row, next_cnt = runs[i + 1]
        if (next_prov != cur_prov or next_tech != cur_tech) and next_cnt >= MIN_RUN_LENGTH:
            events.append({
                "session_id": int(next_row.get("session_id")) if next_row.get("session_id") is not None else None,
                "timestamp": next_row["timestamp"],
                "lat": float(next_row["lat"]),
                "lon": float(next_row["lon"]),
                "from_provider": cur_prov,
                "to_provider": next_prov,
                "from_network": cur_tech,
                "to_network": next_tech,
            })

    # Deduplicate events (timestamp + lat/lon + from/to provider)
    unique = []
    seen = set()
    for ev in events:
        k = (str(ev.get('timestamp')), round(float(ev.get('lat')), 6), round(float(ev.get('lon')), 6), ev.get('from_provider'), ev.get('to_provider'))
        if k in seen:
            continue
        seen.add(k)
        unique.append(ev)

    return unique


def build_provider_colors(df: pd.DataFrame):
    vals = df["m_alpha_long"].dropna().astype(str).unique().tolist()
    vals = sorted(vals)
    colors = generate_distinct_colors(max(1, len(vals)))
    return {v: colors[i] for i, v in enumerate(vals)}


def render_map(df: pd.DataFrame, events: list, provider_colors: dict, project_meta: dict):
    # Create map centered on data
    df = df.dropna(subset=["lat", "lon"]) 
    center = (float(df["lat"].mean()), float(df["lon"].mean()))
    m = Map(location=center, tiles="CartoDB positron", zoom_start=13, prefer_canvas=True)

    # Per-session solid routes: draw full route (no thinning) and overlay filled points
    session_colors = generate_distinct_colors(max(1, df["session_id"].nunique()))
    session_color_map = {sid: session_colors[i] for i, sid in enumerate(sorted(df["session_id"].unique()))}

    for sid, group in df.groupby("session_id"):
        g = group.sort_values(["timestamp"]) if "timestamp" in group.columns else group
        coords = list(zip(g["lat"], g["lon"]))
        color = session_color_map.get(sid, "#2b8cbe")
        # Solid polyline backbone
        PolyLine(locations=coords, color=color, weight=5, opacity=0.95).add_to(m)
        # Filled points at each sample to produce the KPI-style thick track
        for _, r in g.iterrows():
            CircleMarker(
                location=(float(r["lat"]), float(r["lon"])),
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.95,
            ).add_to(m)

    # Add polygon boundary if available (project metadata may use 'region' or 'polygon_wkt')
    polygon_wkt = project_meta.get('polygon_wkt') or project_meta.get('region')
    if polygon_wkt:
        from shapely.wkt import loads
        geom = loads(polygon_wkt)
        polygon_latlon = [(coord[1], coord[0]) for coord in geom.exterior.coords]
        Polygon(
            locations=polygon_latlon,
            color="#FF0000",
            weight=5,
            fill=False,
            opacity=1.0,
            tooltip="Polygon Boundary"
        ).add_to(m)

    # Spark SVG for handover events (uniform color to avoid visual noise)
    spark_svg = (
        '<div style="transform: translate(-50%, -50%);">'
        '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24">'
        '<path fill="{color}" stroke="#222" stroke-width="0.6" d="M13 2 L3 14 H12 L11 22 L21 10 H12 L13 2 Z"/>'
        '</svg></div>'
    )

    # Keep the same spark appearance (no change requested)
    spark_color = "#ff9933"
    for ev in events:
        html = spark_svg.format(color=spark_color)
        icon = DivIcon(html=html, icon_size=(28, 28), icon_anchor=(14, 14))
        tooltip = f"{ev.get('from_provider')} → {ev.get('to_provider')} (Session {ev.get('session_id')})"
        Marker(location=(ev["lat"], ev["lon"]), icon=icon, tooltip=tooltip).add_to(m)

    # Fit to bounds (include polygon if present)
    bounds = get_df_bounds(df)
    if polygon_wkt:
        bounds = merge_bounds(bounds, get_polygon_bounds(polygon_wkt))
    bounds = expand_bounds(bounds, expand_factor=0.02)
    m.fit_bounds(bounds, max_zoom=18)
    m.save(str(TMP_HTML))


def html_to_png_safe(html_path: str, png_path: str):
    if html_to_png is None:
        raise RuntimeError("Playwright html_to_png not available. Install playwright and ensure browsers are installed.")
    html_to_png(html_path, png_path)


def build_pdf(png_path: str, out_pdf: str):
    c = canvas.Canvas(out_pdf, pagesize=A4)
    w, h = A4
    margin = 36
    max_w = w - margin * 2
    max_h = h - margin * 2

    from PIL import Image
    img = Image.open(png_path)
    iw, ih = img.size
    ratio = min(max_w / iw, max_h / ih)
    draw_w = iw * ratio
    draw_h = ih * ratio
    x = (w - draw_w) / 2
    y = (h - draw_h) / 2
    c.drawImage(png_path, x, y, draw_w, draw_h)
    c.showPage()
    c.save()


def main():
    ensure_dirs()
    print("Loading filtered project data (same as pipeline)...")
    filtered_df, project_meta = fetch_network_logs()

    if filtered_df is None or filtered_df.empty:
        print("No filtered data returned — check project_id and polygons.")
        return

    print(f"Filtered rows: {len(filtered_df)}")
    events = detect_handover_events(filtered_df)
    print(f"Detected {len(events)} provider-change events")

    provider_colors = build_provider_colors(filtered_df)
    print("Rendering map HTML...")
    render_map(filtered_df, events, provider_colors, project_meta)
    print("Map HTML saved:", TMP_HTML)

    # Convert HTML to PNG (skip PDF generation)
    if html_to_png is None:
        print("Playwright html_to_png not available — PNG not created.\nInstall Playwright and run `python -m playwright install` to enable PNG conversion.")
    else:
        print("Converting HTML to PNG...")
        try:
            html_to_png_safe(str(TMP_HTML), str(OUT_PNG))
            print("PNG saved:", OUT_PNG)
        except Exception as e:
            print("HTML->PNG conversion failed:", e)


if __name__ == "__main__":
    main()

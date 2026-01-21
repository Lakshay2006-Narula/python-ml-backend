from flask import Blueprint, request, jsonify, current_app
from shapely import wkt, ops
import geopandas as gpd
import re
from .services import (
    create_block_grid, 
    fetch_buildings, 
    create_ai_zones, 
    cluster_buildings_to_polygons, 
    save_to_database, 
    export_files,
    get_project_data 
)

area_breakup_bp = Blueprint('area_breakup', __name__)

# ================== HELPER: SMART WKT PARSER ==================
def clean_and_load_wkt(raw_string):
    """
    1. Adds 'POLYGON' if missing (e.g., input is just ((...))).
    2. Detects Swapped Coordinates (Lat/Lon) and fixes them to (Lon/Lat).
    """
    if not raw_string:
        raise ValueError("WKT string is empty")

    # 1. Fix missing 'POLYGON' prefix
    clean_str = raw_string.strip()
    if not clean_str.upper().startswith("POLYGON"):
        # If it starts with ((, assume it's a polygon
        if clean_str.startswith("(("):
            clean_str = f"POLYGON {clean_str}"
        else:
            raise ValueError("Input must be a POLYGON or ((...))")

    # Load Geometry
    poly_geom = wkt.loads(clean_str)

    # 2. Check bounds to detect Lat/Lon swap
    min_x, min_y, max_x, max_y = poly_geom.bounds
    
    # Logic: 
    # - Latitude (Y) must be between -90 and 90.
    # - If Y is > 90, user definitely sent Longitude as Y. -> SWAP.
    # - If X is < 40 and Y > 60 (Specific check for users pasting Lat/Lon near India/Taiwan) -> SWAP.
    #   (This heuristic assumes you aren't actually mapping the Arctic Ocean)
    
    needs_swap = False
    
    if min_y < -90 or max_y > 90:
        needs_swap = True
    elif (min_x > -60 and max_x < 60) and (min_y > 60): 
        # Heuristic: If X is small (Europe/Africa Lon) and Y is huge (Arctic Lat)
        # But user meant Lat ~28, Lon ~77 (India) -> The logic sees 28 as X, 77 as Y.
        # Wait, 77 is < 90. Standard check won't catch Delhi (28, 77).
        # We will strictly swap if Y > 90 (Taiwan case).
        # For Delhi (28, 77), it's ambiguous, but we can't swap safely without risk.
        pass

    # FORCE SWAP if it looks like Taiwan/China coordinates sent as Lat/Lon
    if max_y > 90:
        needs_swap = True

    if needs_swap:
        print("🔄 Auto-Correcting: Swapping Lat/Lon to Lon/Lat...")
        poly_geom = ops.transform(lambda x, y: (y, x), poly_geom)

    return poly_geom

# ================== PROCESS ENDPOINT (POST) ==================
@area_breakup_bp.route('/process', methods=['POST'])
def process_data():
    try:
        data = request.get_json()
        name = data.get("Name", "output")
        wkt_input = data.get("WKT")
        project_id = data.get("project_id")
        block_size = float(data.get("grid", 100))
        min_samples = int(data.get("min_samples", 10))

        if not wkt_input:
            return jsonify({"status": "error", "message": "WKT is required"}), 400

        # --- SMART PARSING APPLIED HERE ---
        try:
            poly_geom = clean_and_load_wkt(wkt_input)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Invalid WKT format: {str(e)}"}), 400
        # ----------------------------------

        mask_polygon = gpd.GeoDataFrame(
            {"Name": [name], "project_id": [project_id]}, 
            geometry=[poly_geom], 
            crs="EPSG:4326"
        )

        results_summary = []

        # 1. PROCESS GRID
        g_blocks = create_block_grid(mask_polygon, block_size)
        if not g_blocks.empty:
            save_to_database(g_blocks, "output_grid_blocks", project_id, name)
            files = export_files(g_blocks, f"{name}_blocks")
            results_summary.append(f"Grid: {len(g_blocks)} blocks saved.")

        # 2. FETCH BUILDINGS
        g_buildings = fetch_buildings(mask_polygon)
        if not g_buildings.empty:
            
            # 3. PROCESS AI ZONES
            g_ai_zones = create_ai_zones(mask_polygon, g_buildings)
            if not g_ai_zones.empty:
                save_to_database(g_ai_zones, "output_ai_zones", project_id, name)
                files = export_files(g_ai_zones, f"{name}_ai_zones")
                results_summary.append(f"AI Zones: {len(g_ai_zones)} zones saved.")

            # 4. PROCESS CLUSTERS
            g_clusters = cluster_buildings_to_polygons(g_buildings, mask_polygon, min_samples=min_samples)
            if not g_clusters.empty:
                save_to_database(g_clusters, "output_building_clusters", project_id, name)
                files = export_files(g_clusters, f"{name}_clusters")
                results_summary.append(f"Clusters: {len(g_clusters)} clusters saved.")

        return jsonify({
            "status": "success",
            "project_id": project_id,
            "message": "Processing complete.",
            "details": results_summary,
            "parameters": {
                "grid_size": block_size,
                "min_samples": min_samples
            }
        })

    except Exception as e:
        current_app.logger.error(f"Area Breakup Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ================== FETCH ENDPOINT (GET) ==================
@area_breakup_bp.route('/fetch/<project_id>', methods=['GET'])
def fetch_data(project_id):
    try:
        data = get_project_data(project_id)
        
        if data is None:
             return jsonify({
                "status": "error", 
                "message": "Database connection failed or error executing query."
            }), 500

        # Return 200 OK even if empty
        if not data.get("grid_blocks") and not data.get("ai_zones"):
            return jsonify({
                "status": "success",
                "message": f"No data found for project_id: {project_id}",
                "data": {
                    "grid_blocks": [],
                    "ai_zones": [],
                    "building_clusters": []
                }
            }), 200

        return jsonify({
            "status": "success",
            "project_id": project_id,
            "data": data
        }), 200

    except Exception as e:
        current_app.logger.error(f"Fetch Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

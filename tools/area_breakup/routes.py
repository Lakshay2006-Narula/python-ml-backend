from flask import Blueprint, request, jsonify, current_app
from shapely import wkt
import geopandas as gpd
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

# ================== PROCESS ENDPOINT (POST) ==================
@area_breakup_bp.route('/process', methods=['POST'])
def process_data():
    try:
        data = request.get_json()
        name = data.get("Name", "output")
        wkt_string = data.get("WKT")
        project_id = data.get("project_id")
        block_size = float(data.get("grid", 100))
        
        # --- PARAMS FOR CLUSTERING ---
        min_samples = int(data.get("min_samples", 10))

        # --- PARAMS FOR AI ZONES (Dynamic Config) ---
        # Defaults: target=200 buildings/zone, min=10 zones, max=80 zones
        ai_target = int(data.get("ai_target", 200)) 
        ai_min_zones = int(data.get("ai_min_zones", 10))
        ai_max_zones = int(data.get("ai_max_zones", 80))

        if not wkt_string:
            return jsonify({"status": "error", "message": "WKT is required"}), 400

        # Load Input Polygon
        try:
            poly_geom = wkt.loads(wkt_string)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Invalid WKT: {str(e)}"}), 400

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
            results_summary.append(f"Grid: {len(g_blocks)} blocks saved. Files: {files}")

        # 2. FETCH BUILDINGS
        g_buildings = fetch_buildings(mask_polygon)
        if not g_buildings.empty:
            
            # 3. PROCESS AI ZONES (Pass dynamic params)
            g_ai_zones = create_ai_zones(
                mask_polygon, 
                g_buildings, 
                target=ai_target, 
                min_zones=ai_min_zones, 
                max_zones=ai_max_zones
            )
            
            if not g_ai_zones.empty:
                save_to_database(g_ai_zones, "output_ai_zones", project_id, name)
                files = export_files(g_ai_zones, f"{name}_ai_zones")
                results_summary.append(f"AI Zones: {len(g_ai_zones)} zones saved. Files: {files}")

            # 4. PROCESS CLUSTERS
            g_clusters = cluster_buildings_to_polygons(g_buildings, mask_polygon, min_samples=min_samples)
            if not g_clusters.empty:
                save_to_database(g_clusters, "output_building_clusters", project_id, name)
                files = export_files(g_clusters, f"{name}_clusters")
                results_summary.append(f"Clusters: {len(g_clusters)} clusters saved. Files: {files}")

        return jsonify({
            "status": "success",
            "project_id": project_id,
            "message": "Processing complete.",
            "details": results_summary,
            "parameters": {
                "grid_size": block_size,
                "min_samples": min_samples,
                "ai_target": ai_target,
                "ai_min_zones": ai_min_zones,
                "ai_max_zones": ai_max_zones
            }
        })

    except Exception as e:
        current_app.logger.error(f"Area Breakup Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ================== FETCH ENDPOINT (GET) ==================
@area_breakup_bp.route('/fetch/<project_id>', methods=['GET'])
def fetch_data(project_id):
    """
    API to get saved data for a specific Project ID.
    """
    try:
        data = get_project_data(project_id)
        
        if data is None:
             return jsonify({
                "status": "error", 
                "message": "Database connection failed or error executing query."
            }), 500

        # Check if we actually found data
        count_grid = len(data.get("grid_blocks", []))
        count_zones = len(data.get("ai_zones", []))
        
        # FIX: Return 200 OK even if empty, to stop frontend crash
        if count_grid == 0 and count_zones == 0:
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

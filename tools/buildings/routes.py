from flask import Blueprint, request, jsonify, current_app
import traceback
from .services import BuildingService

buildings_bp = Blueprint("buildings", __name__)
service = BuildingService()


# -------------------------------------
# CORS preflight
# -------------------------------------
@buildings_bp.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response, 204


# -------------------------------------
# GENERATE + SAVE BUILDINGS
# -------------------------------------
@buildings_bp.route("/generate", methods=["POST"])
def generate_buildings():

    try:
        if not request.is_json:
            return jsonify({"Status": 0, "Message": "JSON body required"}), 400

        data = request.get_json()

        result = service.process_buildings(data)

        if not result:
            return jsonify({
                "Status": 0,
                "Message": "No buildings found in this area"
            })

        geojson, extracted_count, saved_count = result


        return jsonify({
            "Status": 1,
            "Message": f"Extracted {extracted_count}, Saved {saved_count} buildings",
            "Data": geojson,
            "Stats": {
                "extracted": extracted_count,
                "saved_to_db": saved_count
            }
        })

    except Exception as e:
        current_app.logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "Status": 0,
            "Message": str(e)
        }), 500


# -------------------------------------
# TEST ENDPOINT
# -------------------------------------
@buildings_bp.route("/test", methods=["GET"])
def test():

    sample = {
        "WKT": "POLYGON((77.2090 28.6139, 77.2100 28.6139, 77.2100 28.6149, 77.2090 28.6149, 77.2090 28.6139))",
        "Name": "Test Area",
        "project_id": 999
    }

    geojson, count, saved = service.process_buildings(sample)

    return jsonify({
        "Status": 1,
        "Message": "Test successful",
        "Extracted": count,
        "Saved": saved,
        "Data": geojson
    })

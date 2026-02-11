import json
from pathlib import Path


def build_structured_area_summary(area_summary: dict) -> dict:
    """Convert raw area_summary into a section/subsection structure suitable for report.

    Output format:
    {
      "Area Summary": {
         "Covered Areas": [ {name, roles, context}, ... ],
         "Hotspots": [...],
         "Crowded Areas": [...]
      }
    }
    """
    structured = {
        "Area Summary": {
            "Covered Areas": area_summary.get("covered_areas", []),
            "Hotspots": area_summary.get("hotspots", []),
            "Crowded Areas": area_summary.get("crowded_areas", []),
        }
    }
    return structured


def main():
    md_path = Path("data/processed/report_metadata.json")
    assert md_path.exists(), f"Metadata file not found: {md_path}"

    md = json.loads(md_path.read_text(encoding="utf-8"))

    area = md.get("area_summary", {})
    structured = build_structured_area_summary(area)

    print(json.dumps(structured, indent=2, ensure_ascii=False))

    # Simple checks
    covered = structured["Area Summary"]["Covered Areas"]
    hotspots = structured["Area Summary"]["Hotspots"]
    crowded = structured["Area Summary"]["Crowded Areas"]

    print("Covered areas:", len(covered))
    print("Hotspots:", len(hotspots))
    print("Crowded areas:", len(crowded))


if __name__ == "__main__":
    main()

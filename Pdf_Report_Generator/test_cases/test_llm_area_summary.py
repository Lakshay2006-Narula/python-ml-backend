import json
from pathlib import Path
from src.llm_integration import parse_and_validate_llm_output


def load_metadata(path="metadata.json"):
    p = Path(path)
    assert p.exists(), f"metadata.json not found at {path}"
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    metadata = load_metadata()

    # A sample LLM raw output that contains Area Summary as a nested object
    sample = {
        "Introduction": "This is a drive-test report for the provided area.",
        "Area Summary": {
            "covered_areas": [
                {"name": "Outer Circle", "roles": ["covered", "hotspot", "crowded"], "context": {"class": "place", "type": "circle", "labels": ["Outer Circle"]}},
                {"name": "Middle Circle", "roles": ["covered", "hotspot"], "context": {"class": "place", "type": "circle", "labels": ["Middle Circle"]}}
            ],
            "hotspots": [
                {"name": "Outer Circle", "roles": ["hotspot"], "context": {"labels": ["Outer Circle"]}}
            ],
            "crowded_areas": [
                {"name": "Outer Circle", "roles": ["crowded"], "context": {"labels": ["Outer Circle"]}}
            ]
        },
        "Drive Summary": "The drive covered multiple sessions and collected samples.",
        "KPI Summary": "KPI summary provided.",
        "Map View - Band": "Band summary.",
        "Map View - RSRP": "RSRP summary.",
        "Map View - RSRQ": "RSRQ summary.",
        "Map View - SINR": "SINR summary.",
        "Map View - DL Throughput": "DL summary.",
        "Map View - UL Throughput": "UL summary.",
        "Map View - MOS": "MOS summary.",
        "PCI Summary": "Observed 94 unique PCI cells; top 30 account for 77.14%."
    }

    raw = json.dumps(sample, indent=2)

    out = parse_and_validate_llm_output(raw, metadata=metadata, output_path="data/processed/report_text_llm_test.json")
    print("Parsed LLM output saved to data/processed/report_text_llm_test.json")
    # basic assertions
    assert isinstance(out, dict)
    assert "Area Summary" in out
    assert isinstance(out["Area Summary"], dict)
    print("Test passed: Area Summary accepted as nested JSON object.")


if __name__ == "__main__":
    main()

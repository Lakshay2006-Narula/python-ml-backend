import json
from src.llm_integration import parse_and_validate_llm_output


def test_kpi_summary_is_numeric_snapshot(tmp_path):
    # Raw LLM output missing KPI Summary -> should be synthesized from metadata
    raw = json.dumps({
        "Introduction": "Intro text for the drive.",
        "Area Summary": "Area overview.",
        "Drive Summary": "Drive summary.",
        "Map View - Band": "Band map.",
        "Map View - RSRP": "RSRP map.",
        "Map View - RSRQ": "RSRQ map.",
        "Map View - SINR": "SINR map.",
        "Map View - DL Throughput": "DL map.",
        "Map View - UL Throughput": "UL map.",
        "Map View - MOS": "MOS map.",
        "PCI Summary": "PCI overview."
    })

    metadata = {
        "location": {"city": "TestCity"},
        "drive_summary": {"distance_covered": 12, "total_samples": 1000, "total_sessions": 3},
        "kpi_summary": {
            "RSRP": {"average": -85, "poor_percentage": 1.5},
            "RSRQ": {"average": -11, "poor_percentage": 10.0},
            "SINR": {"average": 2.5, "poor_percentage": 5.0}
        }
    }

    out = parse_and_validate_llm_output(raw, metadata=metadata, output_path=str(tmp_path / "report_text.json"))

    ks = out.get("KPI Summary")
    assert isinstance(ks, str)
    # Should contain numeric snapshot entries for KPIs present in metadata
    assert "RSRP" in ks and "avg" in ks

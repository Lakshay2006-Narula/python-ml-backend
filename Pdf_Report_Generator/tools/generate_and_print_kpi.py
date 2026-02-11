import json
from src.llm_integration import parse_and_validate_llm_output

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
    "drive_summary": {"distance_covered": 12, "total_samples": 1000, "total_sessions": 3, "start_date": "2026-01-03", "end_date": "2026-01-03"},
    "kpi_summary": ["RSRP", "RSRQ", "SINR"],
    "kpi_details": {"RSRP": {"poor_percentage": 6.13}, "RSRQ": {"poor_percentage": 17.73}, "SINR": {"poor_percentage": 22.57}}
}

out = parse_and_validate_llm_output(raw, metadata, output_path='data/processed/report_text.json')
print(out.get('KPI Summary'))

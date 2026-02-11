import json
import os
import pytest
from src.llm_integration import generate_report_text
from src.pdf_generator import PDFReportGenerator


REQUIRED_MAP_KPIS = [
    "Map View - RSRP",
    "Map View - RSRQ",
    "Map View - SINR",
    "Map View - DL Throughput",
    "Map View - UL Throughput",
    "Map View - MOS",
]


def test_llm_text_only_pdf(tmp_path, monkeypatch, capsys):
    # Load existing metadata
    metadata_path = os.path.join("data", "processed", "report_metadata.json")
    assert os.path.exists(metadata_path), "Missing metadata file for test"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Generate report text (this may call the LLM; fallback is allowed)
    out_text_path = os.path.join("data", "processed", "report_text_test.json")
    report_text = generate_report_text(metadata, output_path=out_text_path, verbose=True)

    # Sanity checks on top-level keys
    for k in [
        "Introduction",
        "Area Summary",
        "Drive Summary",
        "KPI Summary",
        "PCI Summary",
    ] + REQUIRED_MAP_KPIS:
        assert k in report_text, f"Missing section {k} in generated report_text"

    # Ensure Area Summary is structured
    assert isinstance(report_text["Area Summary"], dict)

    # Replace add_image to avoid any image insertions
    def _no_image(self, *args, **kwargs):
        return

    monkeypatch.setattr(PDFReportGenerator, "add_image", _no_image)

    # Generate PDF (text-only)
    out_pdf = tmp_path / "llm_text_only_report.pdf"
    out_pdf_persist = os.path.join("data", "processed", "llm_text_only_report.pdf")
    gen = PDFReportGenerator(output_path=str(out_pdf), images_dir="/nonexistent/images")
    gen.generate_report(report_text, metadata, verbose=True)

    # Also write a persistent copy for easy inspection
    gen_persist = PDFReportGenerator(output_path=out_pdf_persist, images_dir="/nonexistent/images")
    gen_persist.generate_report(report_text, metadata, verbose=True)

    assert out_pdf.exists(), "PDF was not generated"
    assert os.path.exists(out_pdf_persist), "Persistent text-only PDF was not generated"

    # Verify each Map View KPI paragraph references numeric fields when available in metadata
    kpi_summary = metadata.get("kpi_summary", {})
    for key in REQUIRED_MAP_KPIS:
        text = report_text.get(key, "")
        assert isinstance(text, str) and text.strip(), f"Empty Map View content for {key}"

        # Try to find corresponding KPI in metadata by key name
        name = key.replace("Map View - ", "")
        found = None
        for kk, vv in kpi_summary.items():
            if name.lower() in str(kk).lower() or str(kk).lower() in name.lower():
                found = vv
                break

        if found:
            # Expect at least one numeric value mentioned (average/min/max/poor_count/poor_percentage)
            numeric_fields = ["average", "min", "max", "poor_count", "poor_percentage", "poor_samples"]
            has_numeric = any(str(found.get(f)) in text for f in numeric_fields if found.get(f) is not None)
            assert has_numeric, f"Map View paragraph for {key} does not reference numeric metadata"

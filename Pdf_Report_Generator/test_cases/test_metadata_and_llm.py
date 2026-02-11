import json
import os
import pandas as pd
from src.kpi_analysis import generate_kpi_summary, generate_kpi_range_summary
from src.metadata_generator import build_metadata, write_metadata_file
from src.llm_integration import generate_report_text
from src.pdf_generator import PDFReportGenerator


def test_regenerate_metadata_and_llm():
    """
    Fast test: load filtered data, regenerate metadata with new schema,
    then generate LLM output and text-only PDF.
    """
    # Load existing filtered data
    filtered_path = os.path.join("data", "processed", "filtered_data.csv")
    assert os.path.exists(filtered_path), "Missing filtered_data.csv - run main pipeline first"
    
    filtered_df = pd.read_csv(filtered_path)
    print(f"Loaded {len(filtered_df)} filtered samples")
    
    # Generate KPI metadata (no images, just data)
    kpi_metadata = generate_kpi_summary(filtered_df)
    kpi_ranges = generate_kpi_range_summary(filtered_df, user_id=13)
    
    if kpi_metadata:
        for kpi, ranges in kpi_ranges.items():
            if kpi in kpi_metadata:
                kpi_metadata[kpi]["distribution"] = ranges
    
    # Build drive summary metadata from existing sessions
    drive_summary_metadata = {
        "distance_covered": 93.01,
        "total_samples": len(filtered_df),
        "total_sessions": filtered_df["session_id"].nunique() if "session_id" in filtered_df.columns else 1,
        "number_of_days": 10,
        "start_date": "2023-02-01",
        "end_date": "2023-02-10",
    }
    
    # Build metadata with new schema (regenerate area_summary with new format)
    metadata = build_metadata(
        filtered_df,
        kpi_details=kpi_metadata,
        drive_summary=drive_summary_metadata,
    )
    
    # Write updated metadata
    metadata_path = os.path.join("data", "processed", "report_metadata.json")
    write_metadata_file(metadata, metadata_path)
    print(f"\n✓ Updated metadata written to {metadata_path}")
    
    # Verify new schema
    assert "kpi_summary" in metadata, "Missing kpi_summary in metadata"
    assert "band_summary" in metadata, "Missing band_summary in metadata"
    assert isinstance(metadata["kpi_summary"], dict), "kpi_summary should be dict"
    
    print(f"✓ kpi_summary has {len(metadata['kpi_summary'])} KPIs")
    print(f"✓ band_summary has {len(metadata.get('band_summary') or [])} bands")
    
    # Generate LLM report text
    report_text_path = os.path.join("data", "processed", "report_text.json")
    report_text = generate_report_text(
        metadata=metadata,
        output_path=report_text_path,
        verbose=True
    )
    
    print(f"\n✓ LLM output written to {report_text_path}")
    
    # Check KPI Summary
    kpi_summary_text = report_text.get("KPI Summary", "")
    print(f"\nKPI Summary: {kpi_summary_text[:200]}...")
    
    # Check Map View sections
    for kpi in ["RSRP", "RSRQ", "SINR", "DL Throughput", "UL Throughput", "MOS"]:
        key = f"Map View - {kpi}"
        text = report_text.get(key, "")
        print(f"\n{key}: {text[:150]}...")
    
    # Generate text-only PDF
    PDFReportGenerator.add_image = lambda self, *args, **kwargs: None
    pdf_path = os.path.join("data", "processed", "llm_text_only_report.pdf")
    gen = PDFReportGenerator(output_path=pdf_path, images_dir="data/images")
    gen.generate_report(report_text, metadata, verbose=False)
    
    print(f"\n✓ Text-only PDF: {pdf_path}")
    assert os.path.exists(pdf_path)

"""
Test LLM Integration and PDF Generation
Uses existing metadata.json and images to test report generation
"""

import sys
import os
import json

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# --------------------------------------------------
# Imports (CORRECT)
# --------------------------------------------------
from src.llm_integration import generate_report_text
from src.pdf_generator import generate_pdf_report


# --------------------------------------------------
# Helper: Load metadata from JSON
# --------------------------------------------------
def load_metadata(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------
# TEST 1: LLM INTEGRATION
# --------------------------------------------------
def test_llm_integration():
    """Test LLM report text generation"""

    print("\n" + "=" * 70)
    print("TEST 1: LLM INTEGRATION")
    print("=" * 70)

    metadata_path = "metadata.json"

    # Check metadata exists
    if not os.path.exists(metadata_path):
        print(f"❌ ERROR: Metadata file not found: {metadata_path}")
        print("   Please run main pipeline first to generate metadata")
        return False

    print(f"✓ Found metadata: {metadata_path}")

    # Load metadata
    try:
        metadata = load_metadata(metadata_path)
        print("✓ Metadata loaded successfully")
        print(f"  - Location: {metadata.get('location', {})}")
        print(f"  - KPI count: {len(metadata.get('kpi_summary', {}))}")
        print(f"  - Band count: {len(metadata.get('band_summary', []))}")
    except Exception as e:
        print(f"❌ ERROR loading metadata: {e}")
        return False

    # Generate report text
    print("\n📝 Generating report text using LLM...")
    try:
        report_text = generate_report_text(
            metadata=metadata,
            output_path="data/processed/report_text.json",
            verbose=True,
            max_tokens=2000
        )

        print("\n✓ Report text generated successfully")
        print(f"  - Sections generated: {len(report_text)}")
        print("  - Output: data/processed/report_text.json")

        # Preview Introduction
        if "Introduction" in report_text:
            intro = report_text["Introduction"]
            preview = intro[:150] + "..." if len(intro) > 150 else intro
            print("\n📄 Introduction Preview:")
            print(f"  {preview}")

        return True

    except Exception as e:
        print("\n❌ ERROR during LLM generation")
        import traceback
        traceback.print_exc()
        return False


# --------------------------------------------------
# TEST 2: PDF GENERATION
# --------------------------------------------------
def test_pdf_generation():
    """Test PDF report generation"""

    print("\n" + "=" * 70)
    print("TEST 2: PDF GENERATION")
    print("=" * 70)

    metadata_path = "metadata.json"
    report_text_path = "data/processed/report_text.json"

    # Validate required files
    if not os.path.exists(metadata_path):
        print(f"❌ ERROR: Metadata file not found: {metadata_path}")
        return False

    if not os.path.exists(report_text_path):
        print(f"❌ ERROR: Report text file not found: {report_text_path}")
        print("   Run LLM test first")
        return False

    print(f"✓ Found metadata: {metadata_path}")
    print(f"✓ Found report text: {report_text_path}")

    # Check images
    images_dir = "data/images"
    if os.path.exists(images_dir):
        total_images = sum(
            len(files)
            for _, _, files in os.walk(images_dir)
            if any(f.endswith(".png") for f in files)
        )
        print(f"✓ Found images under {images_dir} (PNG files present)")
    else:
        print(f"⚠ WARNING: Images directory not found: {images_dir}")

    # Generate PDF
    print("\n📄 Generating PDF report...")
    try:
        pdf_path = generate_pdf_report(
            metadata_path=metadata_path,
            report_text_path=report_text_path,
            output_path="data/processed/drive_test_report.pdf",
            images_dir="data/images",
            verbose=True
        )

        if os.path.exists(pdf_path):
            size_kb = os.path.getsize(pdf_path) / 1024
            print("\n✓ PDF generated successfully!")
            print(f"  - Path: {pdf_path}")
            print(f"  - Size: {size_kb:.2f} KB")
            return True
        else:
            print("❌ ERROR: PDF file was not created")
            return False

    except Exception as e:
        print("\n❌ ERROR during PDF generation")
        import traceback
        traceback.print_exc()
        return False


# --------------------------------------------------
# MAIN TEST RUNNER
# --------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("LLM & PDF GENERATION TEST SUITE")
    print("=" * 70)
    print("\nThis test uses existing metadata.json and images")
    print("to validate only LLM integration and PDF generation.\n")

    results = {}

    # Test 1: LLM
    results["llm"] = test_llm_integration()

    # Test 2: PDF (only if LLM passed)
    if results["llm"]:
        results["pdf"] = test_pdf_generation()
    else:
        print("\n⚠ Skipping PDF test due to LLM failure")
        results["pdf"] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"LLM Integration : {'✓ PASSED' if results['llm'] else '❌ FAILED'}")
    print(f"PDF Generation  : {'✓ PASSED' if results['pdf'] else '❌ FAILED'}")
    print("=" * 70)

    if all(results.values()):
        print("\n🎉 ALL TESTS PASSED!")
        print("📄 Final PDF: data/processed/drive_test_report.pdf")
    else:
        print("\n⚠ SOME TESTS FAILED — check logs above")

    return all(results.values())


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

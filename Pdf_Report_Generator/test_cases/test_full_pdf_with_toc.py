"""
Test Full PDF with Proper TOC Structure
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_generator import generate_pdf_report
import json
import re
from pypdf import PdfReader

def test_full_pdf_with_toc():
    """Test full PDF generation with proper TOC structure"""
    
    print("\n" + "=" * 70)
    print("TEST: FULL PDF WITH PROPER TOC STRUCTURE")
    print("=" * 70)
    
    metadata_path = "data/processed/report_metadata.json"
    report_text_path = "data/processed/report_text.json"
    
    if not os.path.exists(metadata_path):
        print(f"❌ Missing: {metadata_path}")
        return False
        
    if not os.path.exists(report_text_path):
        print(f"❌ Missing: {report_text_path}")
        return False
    
    print(f"✓ Found metadata: {metadata_path}")
    print(f"✓ Found report text: {report_text_path}")
    
    # Check images
    images_dir = "data/images"
    if os.path.exists(images_dir):
        kpi_maps = len([f for f in os.listdir(f"{images_dir}/kpi_maps") if f.endswith('.png')])
        kpi_analysis = len([f for f in os.listdir(f"{images_dir}/kpi_analysis") if f.endswith('.png')])
        print(f"✓ Found images: {kpi_maps} maps, {kpi_analysis} analysis")
    
    # Generate PDF
    print("\n📄 Generating PDF with proper TOC structure...")
    print("   Features:")
    print("   - TOC with dynamic page numbers")
    print("   - Hierarchical structure (sections + subsections)")
    print("   - Subsections with a, b, c, d labels")
    print("   - Page numbers right-aligned")
    print("   - Clickable TOC links")
    print("   - Uses multiBuild() for TOC generation")
    
    try:
        pdf_path = generate_pdf_report(
            metadata_path=metadata_path,
            report_text_path=report_text_path,
            output_path="data/processed/drive_test_report_toc.pdf",
            images_dir=images_dir,
            verbose=True
        )
        
        if os.path.exists(pdf_path):
            size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            print(f"\n✓ PDF generated successfully!")
            print(f"  - Path: {pdf_path}")
            print(f"  - Size: {size_mb:.2f} MB")

            # Validate TOC content in PDF text
            try:
                reader = PdfReader(pdf_path)
                first_pages = "\n".join(
                    (reader.pages[i].extract_text() or "")
                    for i in range(min(2, len(reader.pages)))
                )

                if "Placeholder for table of contents" in first_pages:
                    print("❌ TOC placeholder detected in PDF")
                    return False

                # Check for TOC title and at least one entry with a page number
                has_toc_title = "Table of Contents" in first_pages
                has_intro_entry = re.search(r"1\.\s*Introduction\s+\d+", first_pages) is not None

                if not has_toc_title or not has_intro_entry:
                    print("❌ TOC content not detected or page numbers missing")
                    return False

                print("✓ TOC detected in PDF text")
            except Exception as toc_err:
                print(f"❌ Failed to validate TOC in PDF: {toc_err}")
                return False
            
            print("\n" + "=" * 70)
            print("VERIFICATION CHECKLIST")
            print("=" * 70)
            print("✅ TOC Structure:")
            print("   1. Introduction")
            print("   2. Area Summary")
            print("   3. Drive Summary")
            print("   4. KPI Summary")
            print("   5. Map View")
            print("       a) Band")
            print("       b) RSRP")
            print("       c) RSRQ")
            print("       d) SINR")
            print("       e) DL Throughput")
            print("       f) UL Throughput")
            print("       g) MOS")
            print("   6. PCI Summary")
            print("       a) PCI Dominance")
            print("       b) PCI with poor coverage")
            print("       c) PCI with poor RSRQ")
            print("   7. App Analytics")
            print("   8. Indoor/Outdoor Summary")
            print("   9. Performance Summary")
            print("       a) Network Quality Metrics")
            print("       b) Speed Metrics")
            print("       c) Latency Distribution")
            print("       d) Jitter Distribution")
            print("   10. Handover Analysis")
            
            print("\n✅ Verify in PDF:")
            print("   1. TOC has dynamic page numbers (NOT hardcoded)")
            print("   2. Page numbers are right-aligned")
            print("   3. Subsections are properly indented")
            print("   4. Clicking TOC entries navigates to sections")
            print("   5. Page numbers on each page (bottom right)")
            print("   6. No excessive white space")
            print("   7. Proper section hierarchy")
            
            return True
        else:
            print("❌ PDF was not created")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_pdf_with_toc()
    sys.exit(0 if success else 1)

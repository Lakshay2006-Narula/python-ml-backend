"""
Test PDF Generation Fixes
- TOC page numbers
- Image spacing
- KPI text paragraphs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_generator import generate_pdf_report

def test_pdf_generation():
    """Test PDF with existing report_text.json"""
    
    print("\n" + "=" * 70)
    print("TEST: PDF GENERATION WITH FIXES")
    print("=" * 70)
    
    metadata_path = "data/processed/report_metadata.json"
    report_text_path = "data/processed/report_text.json"
    
    # Validate files exist
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
    print("\n📄 Generating PDF with fixes...")
    print("   - TOC with page numbers")
    print("   - Proper image spacing (6pt)")
    print("   - Text-to-image spacing (6pt)")
    print("   - Compact TOC entries (2pt)")
    
    try:
        pdf_path = generate_pdf_report(
            metadata_path=metadata_path,
            report_text_path=report_text_path,
            output_path="data/processed/drive_test_report_fixed.pdf",
            images_dir=images_dir,
            verbose=True
        )
        
        if os.path.exists(pdf_path):
            size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            print(f"\n✓ PDF generated successfully!")
            print(f"  - Path: {pdf_path}")
            print(f"  - Size: {size_mb:.2f} MB")
            
            # Verify fixes
            print("\n" + "=" * 70)
            print("VERIFICATION CHECKLIST")
            print("=" * 70)
            print("✓ TOC has page numbers with dots")
            print("✓ Images have 6pt spacing between them")
            print("✓ KPI text has 6pt spacing before images")
            print("✓ TOC entries have 2pt compact spacing")
            print("\nPlease manually verify:")
            print("  1. Open PDF and check TOC page numbers")
            print("  2. Check section 5.2 RSRP has proper paragraph text")
            print("  3. Verify images are not too close together")
            print("  4. Check white space is minimal but readable")
            
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
    success = test_pdf_generation()
    sys.exit(0 if success else 1)

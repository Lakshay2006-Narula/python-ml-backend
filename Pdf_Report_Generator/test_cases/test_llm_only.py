"""
Test LLM Integration Only - No PDF
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_integration import generate_report_text
import json

def test_llm_only():
    """Test LLM generation with existing metadata"""
    
    print("\n" + "=" * 70)
    print("TEST: LLM INTEGRATION ONLY")
    print("=" * 70)
    
    metadata_path = "data/processed/report_metadata.json"
    
    if not os.path.exists(metadata_path):
        print(f"❌ Missing: {metadata_path}")
        return False
        
    print(f"✓ Found metadata: {metadata_path}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✓ Metadata loaded")
    print(f"  - KPIs: {len(metadata.get('kpi_summary', {}))}")
    print(f"  - Bands: {len(metadata.get('band_summary', []))}")
    
    # Generate report text
    print("\n📝 Calling LLM to generate report text...")
    print("   (This tests if LLM output is properly parsed)")
    
    try:
        report_text = generate_report_text(
            metadata=metadata,
            output_path="data/processed/report_text_test.json",
            verbose=True,
            max_tokens=2500
        )
        
        print(f"\n✓ LLM generation successful!")
        print(f"  - Sections: {len(report_text)}")
        
        # Check critical sections
        print("\n" + "=" * 70)
        print("CONTENT VERIFICATION")
        print("=" * 70)
        
        intro = report_text.get("Introduction", "")
        print(f"\n✓ Introduction ({len(intro)} chars):")
        print(f"  {intro[:200]}...")
        
        drive = report_text.get("Drive Summary", "")
        print(f"\n✓ Drive Summary ({len(drive)} chars):")
        print(f"  {drive}")
        
        kpi = report_text.get("KPI Summary", "")
        print(f"\n✓ KPI Summary ({len(kpi)} chars):")
        print(f"  {kpi}")
        
        rsrp = report_text.get("Map View - RSRP", "")
        print(f"\n✓ Map View - RSRP ({len(rsrp)} chars):")
        print(f"  {rsrp[:200]}...")
        
        if len(intro) < 100:
            print("\n⚠️ WARNING: Introduction is too short (fallback text?)")
            return False
            
        if "threshold" in rsrp.lower():
            print("\n⚠️ WARNING: 'threshold' word found in RSRP (should use actual values)")
            return False
        
        print("\n✓ All content looks good (report-level quality)")
        return True
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llm_only()
    sys.exit(0 if success else 1)

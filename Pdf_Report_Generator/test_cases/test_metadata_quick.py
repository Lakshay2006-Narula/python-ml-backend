import json
import pandas as pd
from src.metadata_generator import build_area_summary

# Create a minimal test with mock data
filtered_df = pd.DataFrame({
    'lat': [28.631, 28.632, 28.633],
    'lon': [77.219, 77.220, 77.221],
    'speed': [10, 20, 30]
})

print("Testing area_summary with detailed objects...")
area_summary = build_area_summary(filtered_df, top_n=2, sleep_sec=1.5)

if area_summary:
    print("\n✅ Generated area_summary:")
    print(json.dumps(area_summary, indent=2))
    
    # Check structure
    if area_summary.get('hotspots'):
        first_hotspot = area_summary['hotspots'][0]
        if isinstance(first_hotspot, dict):
            print(f"\n✅ SUCCESS: Hotspots are now detailed objects!")
            print(f"   - Has 'name': {' name' in first_hotspot}")
            print(f"   - Has 'roles': {'roles' in first_hotspot}")
            print(f"   - Has 'context': {'context' in first_hotspot}")
        else:
            print(f"\n❌ FAILED: Hotspots are still strings: {type(first_hotspot)}")
else:
    print("❌ No area_summary generated")

"""
INVESTIGATION: Why app counts changed
--------------------------------------
Checking:
1. What's in app_name column vs apps column
2. Why count changed from 281 to 655
3. Where are the other apps
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.load_data_db import load_project_data

print("=" * 80)
print("INVESTIGATING APP ANALYTICS COUNT CHANGE")
print("=" * 80)

# Load real data
raw_df, filtered_df, project_meta = load_project_data(149)

print(f"\n[TOTAL DATA]")
print(f"Total filtered rows: {len(filtered_df)}")

print(f"\n[CHECKING 'app_name' COLUMN]")
if 'app_name' in filtered_df.columns:
    app_name_not_null = filtered_df['app_name'].notna().sum()
    print(f"Rows with app_name data: {app_name_not_null}")
    if app_name_not_null > 0:
        print(f"Unique values in app_name:")
        for app in filtered_df['app_name'].dropna().unique():
            count = len(filtered_df[filtered_df['app_name'] == app])
            print(f"  {app}: {count} samples")
    else:
        print("  ❌ app_name column is EMPTY")
else:
    print("  ❌ app_name column does NOT exist")

print(f"\n[CHECKING 'apps' COLUMN]")
if 'apps' in filtered_df.columns:
    apps_not_null = filtered_df['apps'].notna().sum()
    print(f"Rows with apps data: {apps_not_null}")
    if apps_not_null > 0:
        print(f"Unique values in apps:")
        for app in filtered_df['apps'].dropna().unique():
            count = len(filtered_df[filtered_df['apps'] == app])
            print(f"  {app}: {count} samples")
    else:
        print("  ❌ apps column is EMPTY")
else:
    print("  ❌ apps column does NOT exist")

print(f"\n[ROWS WITH NO APP DATA]")
no_app_data = len(filtered_df[filtered_df['apps'].isna()])
print(f"Rows with NULL/empty apps: {no_app_data}")

print(f"\n[COMPARISON]")
print("When code checked BOTH columns (app_name first, then apps):")
print("  → app_name was EMPTY, so it fell back to 'apps'")
print("  → But maybe the logic was different?")

print(f"\n[OLD CODE LOGIC]")
print("if app_name has data: use app_name")
print("elif apps has data: use apps")
print("→ Since app_name is empty, it should have used apps anyway")

print(f"\n[EXPLANATION]")
if 'apps' in filtered_df.columns:
    apps_count = filtered_df['apps'].notna().sum()
    print(f"Total rows with apps data: {apps_count}")
    print(f"Current result shows: 655 samples for Whatsapp")
    if apps_count == 655:
        print("✅ CORRECT: 655 is the actual count in database")
        print("❓ The 281 count might have been from:")
        print("   • Different filtering")
        print("   • Different data in database at that time")
        print("   • Test data (not real database)")
    else:
        print(f"❓ Mismatch: Expected {apps_count}, showing 655")

print("\n" + "=" * 80)

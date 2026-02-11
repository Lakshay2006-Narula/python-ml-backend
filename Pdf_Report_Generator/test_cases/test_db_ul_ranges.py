"""
Test to check what UL ranges are actually in the database
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db import get_user_thresholds
import json

def test_db_ul_ranges():
    print("=" * 70)
    print("CHECKING UL RANGES IN DATABASE")
    print("=" * 70)
    
    user_id = 13  # From main.py
    
    db_row = get_user_thresholds(user_id)
    
    if not db_row:
        print(f"❌ No threshold row found for user_id={user_id}")
        return
    
    print(f"\n✅ Found threshold row for user_id={user_id}")
    
    # Check UL threshold
    ul_json = db_row.get("ul_thpt_json")
    
    print(f"\nUL JSON from DB:")
    print(f"Raw: {ul_json}")
    
    if ul_json:
        try:
            ul_ranges = json.loads(ul_json)
            print(f"\nParsed UL Ranges ({len(ul_ranges)} total):")
            for idx, r in enumerate(ul_ranges):
                print(f"  [{idx}] min={r.get('min')}, max={r.get('max')}, "
                      f"color={r.get('color')}, range='{r.get('range', 'N/A')}'")
            
            # Check for issues
            print("\n" + "-" * 70)
            print("VALIDATION:")
            print("-" * 70)
            
            for idx, r in enumerate(ul_ranges):
                min_val = r.get('min')
                max_val = r.get('max')
                
                if min_val is None or max_val is None:
                    print(f"  ❌ Range [{idx}]: Missing min or max")
                elif min_val >= max_val:
                    print(f"  ❌ Range [{idx}]: min ({min_val}) >= max ({max_val})")
                elif min_val == 0 and max_val == 0:
                    print(f"  ⚠️  Range [{idx}]: Both min and max are 0")
                else:
                    print(f"  ✅ Range [{idx}]: {min_val} to {max_val} OK")
        
        except Exception as e:
            print(f"❌ Error parsing JSON: {e}")
    else:
        print("❌ No UL JSON found in database")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_db_ul_ranges()

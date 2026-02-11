import json
from pprint import pprint

from src.db import get_user_thresholds
from src.threshold_resolver import resolve_kpi_ranges


USER_ID = 13
KPI_NAME = "RSRQ"   # change to RSRQ / SINR / DL / UL / MOS to test others


def main():
    print("=" * 60)
    print(f"TESTING THRESHOLDS FOR user_id = {USER_ID}")
    print("=" * 60)

    # 1️⃣ Raw DB fetch
    db_row = get_user_thresholds(USER_ID)

    print("\n[1] RAW DB ROW:")
    pprint(db_row)

    if not db_row:
        print("\n❌ No row returned from DB")
        return

    json_col = f"{KPI_NAME.lower()}_json"

    print(f"\n[2] EXPECTED JSON COLUMN: {json_col}")

    if json_col not in db_row:
        print(f"\n❌ Column '{json_col}' NOT FOUND in DB row")
        print("Available columns:", list(db_row.keys()))
        return

    if not db_row[json_col]:
        print(f"\n❌ Column '{json_col}' is EMPTY / NULL")
        return

    # 3️⃣ Parse JSON
    print("\n[3] RAW JSON STRING:")
    print(db_row[json_col])

    try:
        parsed = json.loads(db_row[json_col])
        print("\n[4] PARSED JSON:")
        pprint(parsed)
    except Exception as e:
        print("\n❌ JSON PARSE FAILED:", e)
        return

    # 4️⃣ Resolver output (THIS IS WHAT MAP RECEIVES)
    print("\n[5] RESOLVED RANGES (USED BY MAP):")
    resolved = resolve_kpi_ranges(
        kpi_name=KPI_NAME,
        user_id=USER_ID
    )
    pprint(resolved)

    print("\n✅ TEST COMPLETED")


if __name__ == "__main__":
    main()

"""
Validate LLM JSON output structure and content types.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_integration import generate_report_text, REQUIRED_KEYS


def test_llm_json_validation():
    metadata_path = "data/processed/report_metadata.json"
    output_path = "data/processed/report_text_test.json"

    print("\n" + "=" * 70)
    print("TEST: LLM JSON VALIDATION")
    print("=" * 70)

    if not os.path.exists(metadata_path):
        print(f"❌ Missing: {metadata_path}")
        return False

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("✓ Metadata loaded")

    try:
        report = generate_report_text(
            metadata=metadata,
            output_path=output_path,
            verbose=True,
            max_tokens=2500,
        )
    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
        return False

    if not os.path.exists(output_path):
        print("❌ Output JSON file not created")
        return False

    with open(output_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    # Validate required keys
    for key in REQUIRED_KEYS:
        if key not in saved:
            print(f"❌ Missing key in JSON: {key}")
            return False

    # Validate types
    if not isinstance(saved.get("Area Summary"), dict):
        print("❌ Area Summary is not a dict")
        return False

    for key in REQUIRED_KEYS:
        if key == "Area Summary":
            continue
        if not isinstance(saved.get(key), str):
            print(f"❌ Section '{key}' is not a string")
            return False
        if not saved.get(key).strip():
            print(f"❌ Section '{key}' is empty")
            return False

    # Basic quality check to detect fallback placeholders
    if saved.get("Introduction", "").strip().lower() in {"not available.", "area summary not available."}:
        print("❌ Introduction looks like fallback")
        return False

    print("✓ LLM JSON output validated successfully")
    return True


if __name__ == "__main__":
    success = test_llm_json_validation()
    sys.exit(0 if success else 1)

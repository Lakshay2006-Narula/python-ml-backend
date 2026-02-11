"""
TEST CASE: CDF DISTRIBUTION FROM API
====================================
This test case fetches CDF (Cumulative Distribution Function) data from API
and generates CDF distribution graphs for all KPIs.

API Endpoint Pattern:
http://192.168.1.67:5224/api/MapView/kpi-distribution?sessionIds=<session_ids>&kpi=<kpi_name>

Output:
- CDF graphs saved in: data/images/kpi_analysis/
- File naming: cdf_<kpi>.png (e.g., cdf_rsrp.png, cdf_rsrq.png)
"""

import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from src.db import get_connection
from src.kpi_config import KPI_CONFIG


# Configuration
BASE_API_URL = "http://192.168.1.67:5224/api/MapView/kpi-distribution"
PROJECT_ID = 149
OUTPUT_DIR = "data/images/kpi_analysis"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_session_ids_for_project(project_id: int) -> list:
    """Fetch session IDs from database for the given project"""
    cn = get_connection()
    
    try:
        query = f"""
        SELECT ref_session_id 
        FROM defaultdb.tbl_project 
        WHERE id = {project_id}
        """
        result = pd.read_sql(query, cn)
        
        if result.empty:
            print(f"❌ No project found with ID: {project_id}")
            return []
        
        ref_session_id = result.iloc[0]["ref_session_id"]
        
        # Parse session IDs (format: "3187,3189,3191")
        session_ids = [
            int(s.strip())
            for s in str(ref_session_id).split(",")
            if s.strip().isdigit()
        ]
        
        return session_ids
        
    finally:
        cn.close()


def fetch_cdf_data_from_api(session_ids: list, kpi: str) -> dict:
    """
    Fetch CDF distribution data from API
    
    Args:
        session_ids: List of session IDs
        kpi: KPI name (e.g., 'rsrp', 'rsrq', 'sinr')
    
    Returns:
        dict: JSON response from API or None if failed
    """
    # Convert session IDs to comma-separated string
    session_ids_str = ",".join(map(str, session_ids))
    
    # Build API URL
    url = f"{BASE_API_URL}?sessionIds={session_ids_str}&kpi={kpi}"
    
    print(f"\n📡 Fetching CDF data for {kpi.upper()}...")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Successfully fetched data")
            return data
        else:
            print(f"   ❌ Failed with status code: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"   ❌ Request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def generate_cdf_plot(cdf_data: dict, kpi: str) -> str:
    """
    Generate CDF plot from API data
    
    Args:
        cdf_data: JSON data from API containing CDF distribution
        kpi: KPI name
    
    Returns:
        str: Path to saved image file
    """
    if not cdf_data:
        print(f"   ❌ No data to plot for {kpi}")
        return None
    
    # Extract data from API response
    # Adjust this based on actual API response structure
    # Common formats:
    # 1. {"values": [...], "cdf": [...]}
    # 2. {"distribution": [{"value": x, "cdf": y}, ...]}
    # 3. Direct array of values
    
    try:
        # Try different possible response formats
        if "Data" in cdf_data:
            # API format: {"Status": ..., "KPI": ..., "Data": [...]}
            data_content = cdf_data["Data"]
            
            if isinstance(data_content, list):
                if len(data_content) == 0:
                    print(f"   ❌ Empty data array for {kpi}")
                    return None
                
                # Check if data items are objects with value/cumulative_count keys
                if isinstance(data_content[0], dict):
                    # Format from API: [{"value": x, "count": y, "percentage": z, "cumulative_count": cc}, ...]
                    if "value" in data_content[0] and "cumulative_count" in data_content[0]:
                        # Extract values and calculate CDF percentage
                        total_count = max(item["cumulative_count"] for item in data_content)
                        values = [item["value"] for item in data_content]
                        cdf = [(item["cumulative_count"] / total_count) * 100 for item in data_content]
                    elif "value" in data_content[0] and "cdf" in data_content[0]:
                        values = [item["value"] for item in data_content]
                        cdf = [item["cdf"] for item in data_content]
                    elif "Value" in data_content[0] and "CDF" in data_content[0]:
                        values = [item["Value"] for item in data_content]
                        cdf = [item["CDF"] for item in data_content]
                    else:
                        print(f"   ⚠️ Unknown data item format. Keys: {list(data_content[0].keys())}")
                        return None
                else:
                    # Format: Direct array of values - calculate CDF
                    values = sorted(data_content)
                    n = len(values)
                    cdf = [(i + 1) / n * 100 for i in range(n)]
            else:
                print(f"   ⚠️ Data is not a list: {type(data_content)}")
                return None
                
        elif "values" in cdf_data and "cdf" in cdf_data:
            # Format 1: Separate arrays
            values = cdf_data["values"]
            cdf = cdf_data["cdf"]
        elif "distribution" in cdf_data:
            # Format 2: Array of objects
            distribution = cdf_data["distribution"]
            values = [item["value"] for item in distribution]
            cdf = [item["cdf"] for item in distribution]
        elif isinstance(cdf_data, list):
            # Format 3: Direct array - calculate CDF
            values = sorted(cdf_data)
            n = len(values)
            cdf = [(i + 1) / n * 100 for i in range(n)]
        else:
            print(f"   ⚠️ Unknown API response format for {kpi}")
            print(f"   Response keys: {cdf_data.keys() if isinstance(cdf_data, dict) else 'Not a dict'}")
            return None
        
        if not values or not cdf:
            print(f"   ❌ Empty data for {kpi}")
            return None
        
        # Create CDF plot
        plt.figure(figsize=(10, 6))
        plt.plot(values, cdf, linewidth=2, color='#2E86AB', marker='o', 
                 markersize=3, markevery=max(1, len(values) // 50))
        
        plt.xlabel(f'{kpi.upper()} Value', fontsize=12, fontweight='bold')
        plt.ylabel('Cumulative Probability (%)', fontsize=12, fontweight='bold')
        plt.title(f'CDF Distribution - {kpi.upper()}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim(0, 100)
        
        # Add percentile markers
        percentiles = [10, 50, 90]
        for p in percentiles:
            # Find closest CDF value to percentile
            idx = min(range(len(cdf)), key=lambda i: abs(cdf[i] - p))
            if idx < len(values):
                plt.axhline(y=p, color='red', linestyle='--', alpha=0.3, linewidth=1)
                plt.axvline(x=values[idx], color='red', linestyle='--', alpha=0.3, linewidth=1)
                plt.text(values[idx], p + 2, f'P{p}: {values[idx]:.2f}', 
                        fontsize=9, color='red', ha='center')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(OUTPUT_DIR, f"cdf_{kpi.lower()}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved CDF plot: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"   ❌ Error generating plot for {kpi}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main test function"""
    print("=" * 80)
    print("CDF DISTRIBUTION FROM API - TEST CASE")
    print("=" * 80)
    
    # Step 1: Get session IDs for the project
    print(f"\n[1] Fetching session IDs for project {PROJECT_ID}")
    session_ids = get_session_ids_for_project(PROJECT_ID)
    
    if not session_ids:
        print("❌ No session IDs found. Exiting.")
        return
    
    print(f"   ✅ Found {len(session_ids)} sessions: {session_ids}")
    
    # Step 2: Process each KPI
    print(f"\n[2] Processing KPIs from configuration")
    
    successful_kpis = []
    failed_kpis = []
    
    for kpi_name, kpi_config in KPI_CONFIG.items():
        # Only process range-based KPIs
        if kpi_config["type"] != "range":
            print(f"\n⏭️ Skipping {kpi_name} (not a range-based KPI)")
            continue
        
        # Get the column name (API uses lowercase column names)
        kpi_column = kpi_config["column"]  # e.g., 'rsrp', 'rsrq', 'sinr'
        
        print(f"\n{'=' * 60}")
        print(f"Processing: {kpi_name} (column: {kpi_column})")
        print('=' * 60)
        
        # Fetch CDF data from API
        cdf_data = fetch_cdf_data_from_api(session_ids, kpi_column)
        
        if cdf_data:
            # Generate CDF plot
            output_file = generate_cdf_plot(cdf_data, kpi_column)
            
            if output_file:
                successful_kpis.append(kpi_name)
            else:
                failed_kpis.append(kpi_name)
        else:
            failed_kpis.append(kpi_name)
    
    # Step 3: Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {len(successful_kpis)} KPIs")
    if successful_kpis:
        for kpi in successful_kpis:
            print(f"   - {kpi}")
    
    print(f"\n❌ Failed: {len(failed_kpis)} KPIs")
    if failed_kpis:
        for kpi in failed_kpis:
            print(f"   - {kpi}")
    
    print(f"\n📁 Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

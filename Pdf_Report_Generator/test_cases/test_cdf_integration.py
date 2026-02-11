"""
Simple test to verify CDF generation integration works
"""
from src.cdf_kpi import generate_all_cdf_plots
from src.db import get_connection
import pandas as pd

# Get session IDs for project 149
PROJECT_ID = 149

cn = get_connection()
query = f"SELECT ref_session_id FROM defaultdb.tbl_project WHERE id = {PROJECT_ID}"
result = pd.read_sql(query, cn)
cn.close()

ref_session_id = result.iloc[0]["ref_session_id"]
session_ids = [int(s.strip()) for s in str(ref_session_id).split(",") if s.strip().isdigit()]

print(f"Project ID: {PROJECT_ID}")
print(f"Session IDs: {session_ids}")

# Generate CDF plots
result = generate_all_cdf_plots(
    session_ids=session_ids,
    output_dir="data/images/kpi_analysis"
)

print("\n✅ Test completed!")
print(f"Successful KPIs: {result['successful']}")
print(f"Failed KPIs: {result['failed']}")

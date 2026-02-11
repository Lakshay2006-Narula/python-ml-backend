import pandas as pd
from src.metadata_generator import build_metadata
from src.kpi_analysis import generate_kpi_summary, generate_kpi_range_summary

df = pd.read_csv('data/processed/filtered_data.csv')
kpi = generate_kpi_summary(df)
kpi_ranges = generate_kpi_range_summary(df, 13)

for k, r in kpi_ranges.items():
    kpi[k]['distribution'] = r

meta = build_metadata(
    df, 
    kpi_details=kpi, 
    drive_summary={'distance_covered': 93.01, 'total_samples': len(df), 'total_sessions': 10}
)

print("\n=== CHANGED THRESHOLDS TEST ===")
print(f"DL poor_threshold: {meta['kpi_summary']['DL']['poor_threshold']}")
print(f"DL poor_count: {meta['kpi_summary']['DL']['poor_count']}")
print(f"DL poor_percentage: {meta['kpi_summary']['DL']['poor_percentage']}%")
print(f"DL excellent_threshold_value: {meta['kpi_summary']['DL']['excellent_threshold_value']}")
print(f"DL excellent_percentage: {meta['kpi_summary']['DL']['excellent_percentage']}%")

print(f"\nUL poor_threshold: {meta['kpi_summary']['UL']['poor_threshold']}")
print(f"UL poor_percentage: {meta['kpi_summary']['UL']['poor_percentage']}%")
print(f"UL range: {meta['kpi_summary']['UL']['range_min']}-{meta['kpi_summary']['UL']['range_max']} Mbps")
print(f"UL range_percentage: {meta['kpi_summary']['UL']['range_percentage']}%")

print(f"\nMOS poor_threshold: {meta['kpi_summary']['MOS']['poor_threshold']}")
print(f"MOS poor_percentage: {meta['kpi_summary']['MOS']['poor_percentage']}%")

print(f"\nSINR poor_threshold: {meta['kpi_summary']['SINR']['poor_threshold']}")
print(f"SINR poor_percentage: {meta['kpi_summary']['SINR']['poor_percentage']}%")
print(f"SINR range: {meta['kpi_summary']['SINR']['range_min']}-{meta['kpi_summary']['SINR']['range_max']}")
print(f"SINR range_percentage: {meta['kpi_summary']['SINR']['range_percentage']}%")

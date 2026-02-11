from src.metadata_generator import build_metadata, write_metadata_file
from src.kpi_analysis import run_kpi_analysis
from src.kpi_config import KPI_CONFIG
import pandas as pd


filtered_df = pd.read_csv(r"C:\Users\91832\Desktop\Pdf_Report\data\processed\filtered_data.csv")
# filtered_df already created using polygon
user_id = "test_user"
kpi_results, drive_summary = run_kpi_analysis(filtered_df, user_id, KPI_CONFIG)

metadata = build_metadata(
    filtered_df=filtered_df,
    kpi_analysis_results=kpi_results,
    drive_summary_data=drive_summary
)

write_metadata_file(metadata, "metadata.json")
print("Metadata generation and writing completed.")
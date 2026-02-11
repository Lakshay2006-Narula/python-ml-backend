from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os

app = FastAPI()

REPORTS_DIR = "data/reports"


@app.get("/download/{report_id}")
def download_report(report_id: str):
    pdf_path = os.path.join(REPORTS_DIR, report_id, "report.pdf")

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="drive_test_report.pdf"
    )

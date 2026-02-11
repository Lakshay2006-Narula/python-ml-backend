"""
End-to-End Test for PDF Report Generator API Integration

Tests the complete flow:
1. POST /api/report/generate with project_id and user_id
2. Verify report_id is returned and status is "processing"
3. Wait for background report generation to complete
4. Check logs for completion message
5. GET /api/report/download/<report_id> successfully downloads PDF
6. Verify temp files are cleaned up (except final PDF)
"""

import requests
import json
import os
import time
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_URL = "http://localhost:5000"  # Flask backend URL
API_ENDPOINT_GENERATE = f"{BASE_URL}/api/report/generate"
MAX_WAIT_TIME = 120  # seconds - max time to wait for report generation
POLL_INTERVAL = 2  # seconds - check every 2 seconds if PDF is ready
PROJECT_ID = 148  # Test project ID - change as needed
USER_ID = 1  # Test user ID - change as needed

# ============================================================================
# TEST UTILITIES
# ============================================================================

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_result(test_name, passed, details=""):
    """Print test result"""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"[{status}] {test_name}")
    if details:
        print(f"       {details}")


def check_pdf_exists(report_id):
    """Check if PDF file exists in reports directory"""
    # The Flask app stores reports in data/reports/<report_id>/report.pdf
    pdf_path = os.path.join("data", "reports", report_id, "report.pdf")
    return os.path.exists(pdf_path)


def get_temp_files(report_id):
    """Get list of temp files for this report"""
    tmp_dir = os.path.join("data", "tmp", report_id)
    if not os.path.exists(tmp_dir):
        return []
    
    temp_files = []
    for root, dirs, files in os.walk(tmp_dir):
        for file in files:
            temp_files.append(os.path.join(root, file))
    return temp_files


def get_final_report_file(report_id):
    """Get the final report PDF file"""
    pdf_path = os.path.join("data", "reports", report_id, "report.pdf")
    if os.path.exists(pdf_path):
        return pdf_path
    return None


def wait_for_pdf_generation(report_id, max_wait=MAX_WAIT_TIME):
    """
    Poll for PDF availability.
    Returns: (success, elapsed_time, reason)
    """
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        
        if check_pdf_exists(report_id):
            return True, elapsed, "PDF file found"
        
        if elapsed > max_wait:
            return False, elapsed, f"Timeout after {max_wait}s - PDF not generated"
        
        print(f"   Polling... ({elapsed:.1f}s) - Waiting for PDF generation...", end='\r')
        time.sleep(POLL_INTERVAL)


def download_pdf(report_id, output_path=None):
    """
    Download the generated PDF.
    Returns: (success, file_path, error_message)
    """
    download_url = f"{BASE_URL}/api/report/download/{report_id}"
    
    try:
        response = requests.get(download_url, timeout=30)
        
        if response.status_code == 404:
            return False, None, "Report not found (404)"
        
        if response.status_code != 200:
            return False, None, f"HTTP {response.status_code}: {response.text}"
        
        if not output_path:
            output_path = f"downloaded_report_{report_id}.pdf"
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        file_size = os.path.getsize(output_path)
        return True, output_path, f"Downloaded {file_size} bytes"
        
    except requests.exceptions.ConnectionError:
        return False, None, "Cannot connect to Flask backend"
    except Exception as e:
        return False, None, str(e)


# ============================================================================
# TEST CASES
# ============================================================================

def test_01_post_generate_returns_report_id():
    """Test Case 1: POST /api/report/generate returns report_id with 202 status"""
    print_section("TEST 1: Generate Report - POST /api/report/generate")
    
    payload = {
        "project_id": PROJECT_ID,
        "user_id": USER_ID
    }
    
    print(f"Sending POST request to {API_ENDPOINT_GENERATE}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(API_ENDPOINT_GENERATE, json=payload, timeout=10)
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Body:")
        print(json.dumps(response.json(), indent=2))
        
        # Verify status code
        if response.status_code != 202:
            print_result("Status Code", False, f"Expected 202, got {response.status_code}")
            return None
        print_result("Status Code", True, "202 Accepted")
        
        # Verify response structure
        data = response.json()
        required_fields = ["report_id", "status", "project_id", "user_id"]
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            print_result("Response Fields", False, f"Missing: {missing_fields}")
            return None
        print_result("Response Fields", True, f"All required fields present")
        
        # Verify status is "processing"
        if data.get("status") != "processing":
            print_result("Processing Status", False, f"Expected 'processing', got '{data.get('status')}'")
            return None
        print_result("Processing Status", True, f"Status: {data['status']}")
        
        # Verify report_id format
        report_id = data.get("report_id")
        if not report_id or len(str(report_id)) < 10:
            print_result("Report ID", False, f"Invalid report_id format: {report_id}")
            return None
        print_result("Report ID", True, f"report_id: {report_id}")
        
        return report_id
        
    except requests.exceptions.ConnectionError:
        print_result("Connection", False, "Cannot connect to Flask backend at " + BASE_URL)
        print("       Make sure the Flask app is running: python app.py")
        return None
    except Exception as e:
        print_result("Request Error", False, str(e))
        return None


def test_02_report_generation_completes(report_id):
    """Test Case 2: Background report generation completes successfully"""
    print_section("TEST 2: Report Generation - Wait for Completion")
    
    print(f"Waiting for PDF generation to complete (report_id: {report_id})...")
    print(f"Max wait time: {MAX_WAIT_TIME} seconds\n")
    
    success, elapsed_time, reason = wait_for_pdf_generation(report_id)
    
    print(f"\n{'': <50}")  # Clear the polling line
    
    if success:
        print_result("Generation Completed", True, f"Completed in {elapsed_time:.2f}s")
        return True
    else:
        print_result("Generation Completed", False, reason)
        return False


def test_03_verify_pdf_accessibility(report_id):
    """Test Case 3: GET /api/report/download/<report_id> returns accessible PDF"""
    print_section("TEST 3: PDF Download - GET /api/report/download/<report_id>")
    
    print(f"Attempting to download PDF for report_id: {report_id}")
    
    success, file_path, message = download_pdf(report_id)
    
    if success:
        file_size = os.path.getsize(file_path)
        print_result("PDF Download", True, f"Downloaded {file_size} bytes")
        print_result("File Path", True, os.path.abspath(file_path))
        
        # Verify file size is reasonable (PDF should be > 50KB typically)
        if file_size > 50000:
            print_result("File Size", True, f"{file_size} bytes (reasonable size)")
        else:
            print_result("File Size", False, f"{file_size} bytes (suspiciously small)")
        
        # Verify it's a valid PDF
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header == b'%PDF':
                print_result("PDF Signature", True, "Valid PDF header found")
                return True
            else:
                print_result("PDF Signature", False, f"Invalid header: {header}")
                return False
    else:
        print_result("PDF Download", False, message)
        return False


def test_04_verify_temp_cleanup(report_id):
    """Test Case 4: Temp files are cleaned up, only final PDF remains"""
    print_section("TEST 4: File Cleanup - Verify Temp Files Management")
    
    # Check temp directory
    temp_files = get_temp_files(report_id)
    final_pdf = get_final_report_file(report_id)
    
    print(f"Report ID: {report_id}\n")
    
    # Check if final PDF exists
    if final_pdf:
        pdf_size = os.path.getsize(final_pdf)
        print_result("Final PDF Exists", True, f"Location: {final_pdf}")
        print(f"         File Size: {pdf_size} bytes")
    else:
        print_result("Final PDF Exists", False, "PDF file not found in data/reports")
        return False
    
    # Check temp directory
    print(f"\nTemp Directory Check:")
    tmp_dir = os.path.join("data", "tmp", report_id)
    
    if not os.path.exists(tmp_dir):
        print_result("Temp Directory Cleaned", True, "Temp directory removed/not present")
        return True
    
    if temp_files:
        print_result("Temp Directory Cleaned", False, f"Still contains {len(temp_files)} files:")
        for file in temp_files[:5]:  # Show first 5
            print(f"         - {file}")
        if len(temp_files) > 5:
            print(f"         ... and {len(temp_files) - 5} more files")
        return False
    else:
        print_result("Temp Directory Cleaned", True, "No temp files remaining")
        return True


def check_logs_for_completion(report_id):
    """Check Flask logs for completion message"""
    print_section("TEST 5: Log Verification")
    
    print(f"Looking for completion logs for report_id: {report_id}")
    print(f"\nNote: This test provides informational output.")
    print(f"You should see logs like:")
    print(f"  [Report] Starting generation: project_id={PROJECT_ID}, user_id={USER_ID}, report_id={report_id}")
    print(f"  [Report] Completed generation: report_id={report_id}")
    print(f"\nThese should appear in the Flask app console output.")
    
    return True


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all test cases"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  PDF REPORT GENERATOR - END-TO-END INTEGRATION TEST".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    print(f"\nConfiguration:")
    print(f"  Base URL: {BASE_URL}")
    print(f"  Project ID: {PROJECT_ID}")
    print(f"  User ID: {USER_ID}")
    print(f"  Max Wait Time: {MAX_WAIT_TIME}s")
    
    # Test 1: Generate Report
    report_id = test_01_post_generate_returns_report_id()
    if not report_id:
        print("\n" + "="*80)
        print("TESTS STOPPED - Cannot proceed without valid report_id")
        print("="*80)
        return False
    
    # Test 2: Wait for Generation
    success = test_02_report_generation_completes(report_id)
    if not success:
        print("\n⚠️  Warning: Report generation did not complete within timeout")
        print("       Continuing with download test anyway...")
    
    # Test 3: Download PDF
    pdf_success = test_03_verify_pdf_accessibility(report_id)
    
    # Test 4: Verify Cleanup
    cleanup_success = test_04_verify_temp_cleanup(report_id)
    
    # Test 5: Check Logs
    check_logs_for_completion(report_id)
    
    # Summary
    print_section("TEST SUMMARY")
    
    summary = {
        "report_id": report_id,
        "post_generate": True,  # Test 1 passed
        "generation_completed": success,
        "pdf_accessible": pdf_success,
        "cleanup_verified": cleanup_success,
    }
    
    print(json.dumps(summary, indent=2))
    
    all_passed = all([
        report_id is not None,
        success,
        pdf_success,
        cleanup_success
    ])
    
    print("\n")
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nThe PDF Report Generator has been successfully integrated into the Flask backend!")
    else:
        print("✗ SOME TESTS FAILED")
        if not success:
            print("\n  Issue: Report generation did not complete.")
            print("  - Check if the database is accessible")
            print("  - Verify project_id exists in the database")
            print("  - Check Flask app logs for detailed error messages")
        if not pdf_success:
            print("\n  Issue: PDF download failed.")
            print("  - Ensure the PDF was generated successfully")
            print("  - Check file permissions in data/reports directory")
        if not cleanup_success:
            print("\n  Issue: Temp files not cleaned up.")
            print("  - Set REPORT_KEEP_TMP=1 to keep temp files for debugging")
            print("  - Check report_engine/main.py cleanup configuration")
    
    print("\n" + "="*80)
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

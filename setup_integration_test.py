"""
SETUP INSTRUCTIONS FOR END-TO-END API TEST

The test_api_report_integration.py requires the Flask backend to be running with 
a properly configured database. Follow these steps:

1. SET UP ENVIRONMENT
   Create a .env file in the project root with your database connection:
   
   DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<dbname>
   FLASK_ENV=development
   SECRET_KEY=your-secret-key

   Or set it directly in PowerShell before running:
   $env:DATABASE_URL="postgresql://..."

2. VERIFY DATABASE
   - Ensure the project_id (default: 148) exists in your database
   - Ensure the user_id (default: 1) exists in your database
   - Verify database connectivity

3. START THE FLASK APP
   In one terminal:
   $env:DATABASE_URL="YOUR_DATABASE_URL"
   python app.py
   
   Expected output:
   * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)

4. RUN THE TEST
   In another terminal:
   python test_api_report_integration.py
   
   The test will:
   - Send POST request to /api/report/generate
   - Wait for PDF generation (up to 120 seconds)
   - Download the PDF via /api/report/download/<report_id>
   - Verify file cleanup

5. CUSTOMIZE TEST PARAMETERS
   Edit test_api_report_integration.py to change:
   - BASE_URL: Flask app URL (default: http://localhost:5000)
   - PROJECT_ID: Test project (default: 148)
   - USER_ID: Test user (default: 1)
   - MAX_WAIT_TIME: Generation timeout in seconds (default: 120)

TROUBLESHOOTING
==============

Q: "Cannot connect to Flask backend"
A: Flask app is not running or not on the expected port.
   Ensure you started: python app.py
   Check that it says "Running on http://127.0.0.1:5000"

Q: "Report not found (404)" after generation timeout
A: Report generation is taking too long or failing silently.
   - Check Flask app console for errors
   - Verify project_id and user_id exist in database
   - Look for logs containing [Report] messages
   - Increase MAX_WAIT_TIME in the test

Q: "Invalid PDF header" or file corruption
A: The PDF generation failed but returned an error file.
   - Check Flask logs for detailed error messages
   - Verify all required data exists in the database
   - Check that Playwright and other dependencies are installed

Q: DATABASE_URL not configured
A: You must set the DATABASE_URL environment variable.
   Example with PostgreSQL:
   $env:DATABASE_URL="postgresql://user:password@localhost:5432/mydb"
   python app.py

RUNNING IN CI/CD PIPELINE
=========================
For automated testing:

1. Set DATABASE_URL as environment variable in your CI system
2. Ensure database migrations are run: flask db upgrade
3. Seed test data (project_id=148, user_id=1 if needed)
4. Run: python test_api_report_integration.py
5. Check exit code: 0 = success, 1 = failure

"""

import os
import sys

print(__doc__)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("To run the integration test with proper setup:")
    print("="*80)
    print("\n1. Set DATABASE_URL environment variable")
    print("2. Run: python app.py")
    print("3. In another terminal, run: python test_api_report_integration.py")
    print("\n"+"="*80)

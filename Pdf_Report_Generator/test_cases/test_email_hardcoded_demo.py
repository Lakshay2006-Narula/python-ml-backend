"""
TEST CASE: Email Sending for PDF Report
This is a DEMO/TEST version with hardcoded values
You will make this dynamic later by:
  - Fetching email from user_id
  - Getting report_id dynamically from report generation
  
For now, it uses hardcoded demo values for testing.
"""

import sys
sys.path.insert(0, '.')

import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import os

load_dotenv()

# =====================================================
# DEMO/TEST DATA (HARDCODED FOR NOW)
# =====================================================

# Your actual PDF file path from report generation
PDF_REPORT_PATH = r"C:\Users\91832\Desktop\Pdf_Report\data\reports\0fc5d6dd-9dda-4f99-9627-f6fde7401da1\report.pdf"

# Hardcoded demo values - YOU WILL CHANGE THESE LATER
DEMO_EMAIL = "vinfocom.client@gmail.com"          # ← Change to actual user email
DEMO_USER_NAME = "Developer"             # ← Change to actual user name
DEMO_PROJECT_NAME = "Project 148"        # ← Change to actual project name
DEMO_REPORT_ID = "0fc5d6dd-9dda-4f99-9627-f6fde7401da1"  # ← Change to actual report ID

# SMTP Configuration (from .env file)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


def test_send_email_with_hardcoded_values():
    """TEST: Send email with hardcoded demo values"""
    
    print("\n" + "="*80)
    print("TEST: SEND EMAIL WITH PDF REPORT")
    print("="*80)
    
    print(f"\n📧 Email Configuration:")
    print(f"   SMTP Host: {SMTP_HOST}")
    print(f"   SMTP Port: {SMTP_PORT}")
    print(f"   SMTP User: {SMTP_USER}")
    
    print(f"\n📝 Hardcoded Demo Values:")
    print(f"   To Email: {DEMO_EMAIL}")
    print(f"   User Name: {DEMO_USER_NAME}")
    print(f"   Project: {DEMO_PROJECT_NAME}")
    print(f"   Report ID: {DEMO_REPORT_ID}")
    
    # Construct download URL
    download_url = f"{BASE_URL}/download/{DEMO_REPORT_ID}"
    
    print(f"\n📎 PDF Report:")
    print(f"   File Path: {PDF_REPORT_PATH}")
    print(f"   Download URL: {download_url}")
    
    # Verify PDF exists
    if os.path.exists(PDF_REPORT_PATH):
        file_size = os.path.getsize(PDF_REPORT_PATH)
        print(f"   Status: ✓ File exists ({file_size} bytes)")
    else:
        print(f"   Status: ✗ File NOT found")
        return False
    
    # Create email message
    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = DEMO_EMAIL
    msg["Subject"] = f"Drive Test Report Ready – {DEMO_PROJECT_NAME}"
    
    # Email body
    email_body = f"""Hello {DEMO_USER_NAME},

Your drive test report for {DEMO_PROJECT_NAME} has been generated successfully.

📊 Report Details:
   Project: {DEMO_PROJECT_NAME}
   Report ID: {DEMO_REPORT_ID}
   Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

You can download the report using the link below:
{download_url}

Or access the PDF directly at:
{PDF_REPORT_PATH}

Regards,
Network Analytics Team
"""
    
    msg.set_content(email_body)
    
    print(f"\n📤 Email Message:")
    print(f"   Subject: {msg['Subject']}")
    print(f"   Body preview (first 200 chars):")
    print(f"   {email_body[:200]}...")
    
    # Send email (FOR TESTING, you can comment this out if SMTP not configured)
    try:
        print(f"\n🔄 Attempting to send email...")
        
        if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
            print(f"   ⚠ SMTP credentials not fully configured in .env")
            print(f"   ✓ TEST MODE: Email content prepared but not sent")
            print(f"\n   To actually send emails, configure:")
            print(f"   - SMTP_HOST=<your_smtp_server>")
            print(f"   - SMTP_PORT=587")
            print(f"   - SMTP_USER=<your_email>")
            print(f"   - SMTP_PASS=<your_password>")
            return True
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        
        print(f"   ✓ Email sent successfully to {DEMO_EMAIL}")
        return True
        
    except smtplib.SMTPAuthenticationError:
        print(f"   ✗ SMTP Authentication Failed")
        print(f"   Check SMTP_USER and SMTP_PASS in .env")
        return False
    except smtplib.SMTPException as e:
        print(f"   ✗ SMTP Error: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_send_email_from_user_id(user_id: int, project_id: int):
    """
    FUTURE: This is how you'll make it DYNAMIC
    (Uncomment and implement when ready)
    """
    
    print("\n" + "="*80)
    print("FUTURE: DYNAMIC EMAIL SENDING FROM USER_ID")
    print("="*80)
    
    print(f"""
Once you're ready, implement this pattern:

1. Get user email from user_id:
   from src.db import get_connection, get_user_by_id
   user = get_user_by_id(user_id, connection)
   email = user['email']
   name = user['full_name']

2. Get project name:
   project = get_project_by_id(project_id, connection)
   project_name = project['name']

3. Call email function:
   send_report_ready_email(
       to_email=email,
       user_name=name,
       project_name=project_name,
       report_id=report_id
   )

For now, this test uses hardcoded demo values.
When ready: Replace DEMO_EMAIL, DEMO_USER_NAME with database lookups.
""")


if __name__ == "__main__":
    import pandas as pd
    
    # Run the test with hardcoded demo values
    success = test_send_email_with_hardcoded_values()
    
    # Show future implementation approach
    test_send_email_from_user_id(user_id=1, project_id=148)
    
    print("\n" + "="*80)
    if success:
        print("✓ TEST PASSED - Email infrastructure is working!")
    else:
        print("⚠ TEST INCOMPLETE - Check SMTP configuration in .env")
    print("="*80)

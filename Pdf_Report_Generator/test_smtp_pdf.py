import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

#  CHANGE THIS PATH if your PDF is elsewhere
PDF_PATH = r"C:\Users\91832\Desktop\Pdf_Report\data\processed\drive_test_report_toc.pdf"


def test_email_with_pdf():
    if not os.path.exists(PDF_PATH):
        print(f" PDF not found at: {PDF_PATH}")
        return

    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = SMTP_USER   # send to yourself first
    msg["Subject"] = "SMTP PDF Attachment Test"
    msg.set_content("This is a test email with PDF attachment sent via Gmail SMTP.")

    with open(PDF_PATH, "rb") as f:
        pdf_data = f.read()

    msg.add_attachment(
        pdf_data,
        maintype="application",
        subtype="pdf",
        filename=os.path.basename(PDF_PATH)
    )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        print(" PDF email sent successfully")

    except Exception as e:
        print(" Failed to send PDF email")
        print(e)


if __name__ == "__main__":
    test_email_with_pdf()

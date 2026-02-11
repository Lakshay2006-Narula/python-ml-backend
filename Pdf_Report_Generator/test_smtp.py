import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")


def test_smtp_connection():
    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = SMTP_USER      # send to yourself first
    msg["Subject"] = "SMTP Test Email"
    msg.set_content("Hello! This is a test email from Python SMTP.")

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        print(" SMTP connection successful. Test email sent.")

    except Exception as e:
        print(" SMTP test failed")
        print(e)


if __name__ == "__main__":
    test_smtp_connection()

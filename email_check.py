import imaplib
import email
from email.header import decode_header
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

EMAIL_SERVER = os.environ.get("EMAIL_SERVER", "imap.gmail.com")
EMAIL_USERNAME = os.environ.get("EMAIL_USERNAME3")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD3")
TRADE_SIGNAL_FROM = os.environ.get("TRADE_SIGNAL_FROM", "alerts@tradingview.com")
MAILBOX = "INBOX"

def connect_to_email():
    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        return mail
    except Exception as e:
        print(f"Failed to connect: {e}")
        return None

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    paragraphs = soup.find_all("p")
    for p in paragraphs:
        text = p.get_text(strip=True)
        if 'entry' in text:
            return text

def get_latest_email_body(mail, sender_email):
    try:
        mail.select(MAILBOX)
        status, messages = mail.search(None, f'(FROM "{sender_email}" UNSEEN)')
        if status != 'OK' or not messages[0]:
            print("No new emails found.")
            return None

        email_ids = messages[0].split()
        latest_email_id = email_ids[-1]

        status, msg_data = mail.fetch(latest_email_id, '(RFC822)')
        if status != 'OK':
            print("Failed to fetch email.")
            return None

        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        subject = decode_header(msg["Subject"])[0][0]
        if isinstance(subject, bytes):
            subject = subject.decode()

        html_body = None

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/html":
                    html_body = part.get_payload(decode=True).decode()
                    break
        else:
            if msg.get_content_type() == "text/html":
                html_body = msg.get_payload(decode=True).decode()

        if html_body:
            extracted_text = extract_text_from_html(html_body)
            print(f"Subject: {subject}")
            print("Extracted <p> text:")
            print(extracted_text)
            return extracted_text
        else:
            print("No HTML body found.")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    mail = connect_to_email()
    if mail:
        get_latest_email_body(mail, TRADE_SIGNAL_FROM)
        mail.logout()

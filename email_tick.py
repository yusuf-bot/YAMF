import imaplib
import email
from email.header import decode_header
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Email configuration
EMAIL_ADDRESS = os.environ.get('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')
IMAP_SERVER = os.environ.get('IMAP_SERVER', 'imap.gmail.com')
IMAP_PORT = int(os.environ.get('IMAP_PORT', 993))

# Search criteria
TARGET_SUBJECT = os.environ.get('TARGET_SUBJECT', 'Your specific subject')
TARGET_CONTENT = os.environ.get('TARGET_CONTENT', 'Your specific content')

# Check frequency in seconds
CHECK_INTERVAL = int(os.environ.get('CHECK_INTERVAL', 60))

def clean_text(text):
    """Clean and decode text from email"""
    if isinstance(text, bytes):
        text = text.decode()
    return text.strip()

def get_email_content(msg):
    """Extract the content from an email message"""
    content = ""
    if msg.is_multipart():
        # If the message has multiple parts, get the text from each part
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            # Skip attachments
            if "attachment" in content_disposition:
                continue
                
            # Get the body of the email
            if content_type == "text/plain":
                body = part.get_payload(decode=True)
                content += clean_text(body)
    else:
        # If the message is not multipart, just get the payload
        content = clean_text(msg.get_payload(decode=True))
    
    return content

def check_for_new_emails():
    """Connect to the email server and check for new emails matching criteria"""
    try:
        # Connect to the IMAP server
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("INBOX")
        
        # Search for unread emails
        status, messages = mail.search(None, "UNSEEN")
        print(messages)
        if status != "OK":
            print("No messages found!")
            return False
            
        # Get the list of email IDs
        email_ids = messages[0].split()
        
        if not email_ids:
            print("No new messages.")
            return False
            
        print(f"Found {len(email_ids)} new messages. Checking for target content...")
        
        # Process each email
        for email_id in email_ids:
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            
            if status != "OK":
                continue
                
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)
            print(msg)
            # Get subject
            subject = decode_header(msg["Subject"])[0][0]
            print(subject)
            if isinstance(subject, bytes):
                subject = subject.decode()
                
            # Check if subject matches
            if TARGET_SUBJECT.lower() in subject.lower():
                # Get content and check if it matches
                content = get_email_content(msg)
                if TARGET_CONTENT.lower() in content.lower():
                    print(f"Found matching email! Subject: {subject}")
                    print(f"Content preview: {content[:100]}...")
                    return True
        
        return False
        
    except Exception as e:
        print(f"Error checking emails: {e}")
        return False
    finally:
        try:
            mail.close()
            mail.logout()
        except:
            pass

def main():
    print(f"Starting email checker. Looking for emails with subject containing '{TARGET_SUBJECT}'")
    print(f"and content containing '{TARGET_CONTENT}'")
    print(f"Checking every {CHECK_INTERVAL} seconds...")
    
    while True:
        print("\nChecking for new emails...")
        if check_for_new_emails():
            print("Target email found!")
            # You can add additional actions here, such as:
            # - Send a notification
            # - Run a specific function
            # - Trigger another script
        else:
            print("No matching emails found.")
            
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
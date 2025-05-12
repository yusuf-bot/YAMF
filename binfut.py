import os
import time
import logging
import traceback
from datetime import datetime, timezone
import imaplib
import email
from email.header import decode_header
from bs4 import BeautifulSoup
import ccxt
from dotenv import load_dotenv
import re
import threading
from flask import Flask, request, jsonify

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Environment config
EMAIL_SERVER = os.environ['IMAP_SERVER']
EMAIL_USERNAME = os.environ['EMAIL_USERNAME3']
EMAIL_PASSWORD = os.environ['EMAIL_PASSWORD3']
TRADE_SIGNAL_FROM = os.environ['TRADE_SIGNAL_FROM']

API_KEY = os.environ['BINANCE_API_KEY']
API_SECRET = os.environ['BINANCE_API_SECRET']

# Initialize Binance Futures
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'options': {'defaultType': 'future'}
})
exchange.set_sandbox_mode(True)  # Set to False for live

PROCESSED_EMAIL_IDS = set()
POSITIONS = {}
START_DATETIME = datetime.now(timezone.utc)
LAST_CHECK_TIME = datetime.now(timezone.utc)
BOT_STATUS = "running"

# Initialize Flask app
app = Flask(__name__)

# === Email Functions ===

def connect_to_email():
    mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
    mail.login(EMAIL_USERNAME, EMAIL_PASSWORD)
    logger.info("Connected to email server.")
    return mail

def get_unseen_trade_emails(mail):
    
    mail.select("tv")
    status, messages = mail.search(None, f'(FROM "{TRADE_SIGNAL_FROM}")')
    if status != 'OK':
        logger.warning("Failed to search emails.")
        return []

    email_ids = messages[0].split()
    new_emails = []
    
    for eid in email_ids:
        eid_str = eid.decode('utf-8')
        if eid_str in PROCESSED_EMAIL_IDS:
            continue
        
        # Fetch the email's date header
        status, data = mail.fetch(eid, '(BODY.PEEK[HEADER.FIELDS (DATE)])')
        if status != 'OK' or not data or not data[0]:
            continue

        raw_date = data[0][1].decode(errors='ignore').strip()
        match = re.search(r'Date:\s*(.*)', raw_date, re.IGNORECASE)
        if not match:
            continue
   
        try:
            email_datetime = email.utils.parsedate_to_datetime(match.group(1)).astimezone(timezone.utc)
            if email_datetime > START_DATETIME:
                new_emails.append(eid)
                logger.info("Fetching unseen trade emails.")
        except Exception as e:
            logger.warning(f"Failed to parse email date: {raw_date} | Error: {e}")

    return new_emails


def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    paragraphs = soup.find_all("p")
    for p in paragraphs:
        text = p.get_text(strip=True)
        if 'entry' in text.lower():  # Basic signal keyword filter
            return text
    return None

def get_email_body_by_id(mail, eid):
    try:
        status, msg_data = mail.fetch(eid, '(RFC822)')
        if status != 'OK':
            logger.error("Failed to fetch email.")
            return None

        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

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
            return extract_text_from_html(html_body)
        else:
            logger.warning("No HTML body found.")
            return None
    except Exception as e:
        logger.error(f"Error: {e}")
        return None

# === Trading Logic ===

def parse_trade_signal(text):
    parts = text.strip().split()
    logger.info(f"Signal parts: {parts}")
    if len(parts) != 7:
        raise ValueError("Signal format invalid")

    try:
        size = float(parts[4])
        size = size if size > 0 else 0.01
    except ValueError:
        size = 0.01

    return {
        'symbol': parts[1].upper().replace('/', ''),
        'direction': parts[2].lower(),
        'price': float(parts[3]),
        'size': size,
        'trail_offset': float(parts[5]),
        'trail_amount': float(parts[6])
    }

def place_market_order(symbol, direction, amount):
    return exchange.create_market_order(symbol=symbol, side=direction, amount=amount)

def place_trailing_stop(symbol, direction, entry_price, trail_offset, trail_point, amount):
    side = 'sell' if direction == 'buy' else 'buy'

    current_price = exchange.fetch_ticker(symbol)['last']
    activation_price = current_price - trail_point if direction == 'sell' else entry_price + trail_point
    max_attempts = 5
    position_size = 0

    callback_rate = (trail_offset / entry_price) * 100
    callback_rate = max(0.1, min(callback_rate, 5.0))
    logger.info(f"Callback rate: {callback_rate}")
    logger.info(f"Placing trailing stop for {symbol} with activation price {activation_price} and callback rate {callback_rate}")
    logger.info(f"current: {current_price} {(activation_price/current_price)*100}%")

    for attempt in range(max_attempts):
        try:
            positions = exchange.fetch_positions()
            for pos in positions:
                if pos['symbol'] == 'ETH/USDT:USDT':
                    size = float(pos['contracts'])
                    if size > 0:
                        position_size = size
                        break

            if position_size > 0:
                break
            time.sleep(1)

        except Exception as e:
            logger.warning(f"Attempt {attempt+1}: Error fetching position size: {e}")
            time.sleep(1)

    if position_size == 0:
        logger.warning(f"No open position to place trailing stop for {symbol}")
        return None

    try:
        order = exchange.create_order(
            symbol=symbol,
            type='TRAILING_STOP_MARKET',
            side=side,
            amount=amount,
            params={
                'activationPrice': round(float(activation_price), 2),
                'callbackRate': round(float(callback_rate), 2),
                'reduceOnly': True
            }
        )
        logger.info(f"Trailing stop order placed: {order}")
        return order

    except Exception as e:
        logger.error(f"Error placing trailing stop: {e}")

        # Fallback to market order if order would immediately trigger
        if 'Order would immediately trigger' in str(e):
            logger.warning(f"Trailing stop would immediately trigger. Closing position with market order.")
            try:
                close_order = place_market_order(symbol, direction, amount)
                logger.info(f"Market close order result: {close_order}")
                return close_order
            except Exception as e2:
                logger.error(f"Failed to close position with market order: {e2}")
                return None

        return None

# === Main Trading Handler ===

def process_signal(text):
    try:
        signal = parse_trade_signal(text)
        symbol = f"{signal['symbol']}"

        logger.info(f"Processing signal: {signal}")

        market_order = place_market_order(symbol, signal['direction'], signal['size'])
        logger.info(f"Market order result: {market_order}")

        trailing_order = place_trailing_stop(
            symbol, signal['direction'], signal['price'], signal['trail_offset'], signal['trail_amount'], signal['size']
        )

        POSITIONS[signal['symbol']] = {
            'direction': signal['direction'],
            'entry_price': signal['price'],
            'size': signal['size'],
            'trailing_order_id': trailing_order.get('id') if trailing_order else None
        }
        return True
    except Exception as e:
        logger.error(f"Error processing signal: {e}")
        logger.error(traceback.format_exc())
        return False

# === Email Checking Function ===

def check_emails():
    global LAST_CHECK_TIME, BOT_STATUS
    
    try:
        LAST_CHECK_TIME = datetime.now(timezone.utc)
        mail = connect_to_email()
        new_emails = get_unseen_trade_emails(mail)
        logger.info(f"New emails: {new_emails}")
        
        for eid in new_emails:
            eid_str = eid.decode('utf-8')
            text = get_email_body_by_id(mail, eid)
            if text:
                logger.info(f"Parsed signal: {text}")
                process_signal(text)
                PROCESSED_EMAIL_IDS.add(eid_str)
        
        mail.logout()
        return True
    except Exception as e:
        logger.error(f"Email check error: {e}")
        BOT_STATUS = "error"
        return False

# === Background Task ===

def background_task():
    global BOT_STATUS
    logger.info("Starting background email checking task")
    BOT_STATUS = "running"
    
    while True:
        try:
            check_emails()
            time.sleep(10)
        except Exception as e:
            logger.error(f"Background task error: {e}")
            BOT_STATUS = "error"
            time.sleep(30)  # Wait longer on error

# === Flask Routes ===

@app.route('/health', methods=['GET'])
def health_check():
    try:
        time_since_last_check = (datetime.now(timezone.utc) - LAST_CHECK_TIME).total_seconds()
        return jsonify({
            'status': BOT_STATUS,
            'uptime': (datetime.now(timezone.utc) - START_DATETIME).total_seconds(),
            'last_check': LAST_CHECK_TIME.isoformat(),
            'seconds_since_last_check': time_since_last_check,
            'positions': len(POSITIONS),
            'processed_emails': len(PROCESSED_EMAIL_IDS)
        }), 200 if BOT_STATUS == "running" and time_since_last_check < 60 else 500
    except Exception as e:
        logger.error(f"/health error: {e}")
        return jsonify({'error': str(e)}), 500

        
@app.route('/positions', methods=['GET'])
def get_positions():
    """Get current positions"""
    return jsonify(POSITIONS)

@app.route('/check-now', methods=['POST'])
def trigger_check():
    """Manually trigger an email check"""
    success = check_emails()
    return jsonify({'success': success})

@app.route('/webhook', methods=['POST'])
def webhook():
    """Webhook endpoint for direct signal processing"""
    try:
        data = request.json
        if not data or 'signal' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
            
        signal_text = data['signal']
        logger.info(f"Received webhook signal: {signal_text}")
        
        success = process_signal(signal_text)
        return jsonify({'success': success})
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({'error': str(e)}), 500

# === Main Function ===

def main():
    logger.info("Starting trading bot with Flask server")
    
    # Start the background task in a separate thread
    bg_thread = threading.Thread(target=background_task, daemon=True)
    bg_thread.start()
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
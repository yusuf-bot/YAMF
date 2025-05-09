import imaplib
import email
import os
import hmac
import hashlib
import time
import json
import requests
from datetime import datetime, timezone, timedelta
import email.utils
from email.header import decode_header
import re
import traceback
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trade_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Load environment variables from .env file
load_dotenv()

# Global storage for processed emails
PROCESSED_EMAIL_IDS = set()

EMAIL_SERVER = os.environ.get("EMAIL_SERVER", "imap.gmail.com")
EMAIL_USERNAME = os.environ.get("EMAIL_USERNAME")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
TRADE_SIGNAL_FROM = os.environ.get("TRADE_SIGNAL_FROM", "sender@example.com")
CHECK_INTERVAL = 10  # seconds

# Binance API credentials
API_KEY = os.environ.get("BINANCE_API_KEY")
API_SECRET = os.environ.get("BINANCE_API_SECRET")
BINANCE_BASE_URL = os.environ.get("BINANCE_BASE_URL", "https://fapi.binance.com")  # Futures API endpoint
USE_TESTNET = os.environ.get("USE_TESTNET", "True").lower() == "true"

if USE_TESTNET:
    BINANCE_BASE_URL = "https://testnet.binancefuture.com"  # Futures testnet API endpoint
    logger.info("Using Binance Futures TESTNET")
else:
    logger.info("Using Binance Futures PRODUCTION")

# Global storage to track positions
POSITIONS = {}

# Store the timestamp when the script starts - use UTC to avoid timezone issues
START_DATETIME = datetime.now(timezone.utc)
logger.info(f"Script started at: {START_DATETIME.strftime('%Y-%m-%d %H:%M:%S UTC')}")

def connect_to_email():
    """Connect to the email server and return the connection object"""
    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        return mail
    except Exception as e:
        logger.error(f"Error connecting to email: {str(e)}")
        return None

def get_emails_since_script_start(mail_connection, sender_email):
    """Get emails from the specified sender received since the script started"""
    try:
        mail_connection.select("tv")
        
        # First get all emails from the sender
        status, messages = mail_connection.search(None, f'(FROM "{sender_email}")')
        
        if status != 'OK':
            logger.warning("No messages found or error in search")
            return []
        
        all_email_ids = messages[0].split()
        
        # If no emails, return empty list
        if not all_email_ids:
            return []
        
        # Filter for emails received after script start time
        recent_email_ids = []
        for email_id in all_email_ids:
            # Skip if already processed
            email_id_str = email_id.decode('utf-8')
            if email_id_str in PROCESSED_EMAIL_IDS:
                continue
                
            # Get internal date of email
            status, date_data = mail_connection.fetch(email_id, '(INTERNALDATE)')
            if status == 'OK':
                email_date_str = date_data[0].decode('utf-8')
                match = re.search(r'INTERNALDATE "([^"]+)"', email_date_str)
                if match:
                    date_str = match.group(1)
                    try:
                        email_date = email.utils.parsedate_to_datetime(date_str)
                        # Make sure email_date has timezone info
                        if email_date.tzinfo is None:
                            email_date = email_date.replace(tzinfo=timezone.utc)
                        
                        # Compare with script start time
                        if email_date > START_DATETIME:
                            recent_email_ids.append(email_id)
                            logger.info(f"New email found: ID {email_id}, received at {email_date}")
                    except Exception as parse_error:
                        logger.error(f"Error parsing email date: {str(parse_error)}")
        
        logger.info(f"Found {len(recent_email_ids)} new emails since {START_DATETIME.strftime('%Y-%m-%d %H:%M:%S')}")
        return recent_email_ids
    except Exception as e:
        logger.error(f"Error searching emails: {str(e)}")
        return []
        
def get_email_content(mail_connection, email_id):
    """Get the content of an email"""
    try:
        status, msg_data = mail_connection.fetch(email_id, '(RFC822)')
        
        if status != 'OK':
            logger.warning(f"Failed to fetch email {email_id}")
            return None
        
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)
        
        # Extract subject for logging
        subject = decode_email_header(msg["Subject"])
        
        # Get email received date for detailed logging
        received_date = None
        date_header = msg.get("Date")
        if date_header:
            try:
                received_date = email.utils.parsedate_to_datetime(date_header)
                logger.info(f"Processing email: '{subject}' received at {received_date.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception:
                logger.info(f"Processing email: '{subject}' (date parsing failed)")
        else:
            logger.info(f"Processing email: '{subject}' (no date header)")
        
        
        # Mark as read
        mail_connection.store(email_id, '+FLAGS', '\\Seen')
        
        return subject
    except Exception as e:
        logger.error(f"Error getting email content: {str(e)}")
        return None

def decode_email_header(header):
    """Decode email header properly"""
    if not header:
        return ""
    
    decoded_header = decode_header(header)
    header_parts = []
    
    for content, encoding in decoded_header:
        if isinstance(content, bytes):
            if encoding:
                header_parts.append(content.decode(encoding))
            else:
                header_parts.append(content.decode('utf-8', errors='replace'))
        else:
            header_parts.append(content)
    
    return " ".join(header_parts)

def parse_trade_signal(email_body):
    """Parse the email content to extract trading signal"""
    try:
        # Clean the email body (remove extra whitespace, etc.)
        email_body = email_body.split()

        ticker = email_body[1].strip()
        action = email_body[2].strip().lower()
        quantity = float(email_body[3].strip())
        price = float(email_body[4].strip())
        position = email_body[5].strip().lower()
        leverage = 5
 
        
        # Validate the data
        if action not in ["buy", "sell"]:
            raise ValueError(f"Invalid action: {action}")
        
        if position not in ["long", "short", "flat"]:
            raise ValueError(f"Invalid market position: {position}")
        
        return {
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "price": price,
            "position": position,
            "leverage": leverage
        }
    except Exception as e:
        logger.error(f"Error parsing trade signal: {str(e)}")
        return None

def binance_futures_request(endpoint, method='GET', params=None):
    """Make a request to Binance Futures API with authentication"""
    if params is None:
        params = {}
    
    # Add timestamp for signature
    params['timestamp'] = str(int(time.time() * 1000))
    
    # Generate signature
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    params['signature'] = signature
    
    # Headers
    headers = {
        "X-MBX-APIKEY": API_KEY
    }
    
    url = f"{BINANCE_BASE_URL}{endpoint}"
    
    try:
        if method == 'GET':
            response = requests.get(url, params=params, headers=headers)
        elif method == 'POST':
            response = requests.post(url, params=params, headers=headers)
        elif method == 'DELETE':
            response = requests.delete(url, params=params, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Check for error status code
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        error_data = {}
        try:
            error_data = response.json()
        except:
            pass
        
        logger.error(f"HTTP error: {http_err}, Response: {error_data}")
        return {"error": str(http_err), "binance_error": error_data}
    except Exception as e:
        logger.error(f"Error making request to Binance: {str(e)}")
        return {"error": str(e)}

def set_leverage(symbol, leverage):
    """Set leverage for a specific symbol"""
    endpoint = "/fapi/v1/leverage"
    params = {
        "symbol": symbol.replace(".P", ""),  # Remove .P suffix for API
        "leverage": leverage
    }
    
    return binance_futures_request(endpoint, method='POST', params=params)

def get_futures_position_info():
    """Get current positions information"""
    endpoint = "/fapi/v2/positionRisk"
    return binance_futures_request(endpoint)

def execute_futures_order(symbol, side, position_side, quantity):
    """Execute a futures order on Binance"""
    symbol = symbol.replace(".P", "")  # Remove .P suffix for API calls
    
    # Set order params
    endpoint = "/fapi/v1/order"
    params = {
        "symbol": symbol,
        "side": side.upper(),  # SELL or BUY
        "type": "MARKET",
        "quantity": f"1",  # Format to 3 decimal places
        "reduceOnly": "false"
    }
    
    # If we're using hedge mode, specify the position side
    if position_side:
        params["positionSide"] = position_side.upper()  # LONG or SHORT
    
    return binance_futures_request(endpoint, method='POST', params=params)

def get_futures_account_balance():
    """Get futures account balance"""
    endpoint = "/fapi/v2/balance"
    return binance_futures_request(endpoint)

def check_position_mode():
    """Check if hedge mode is enabled (dual position side)"""
    endpoint = "/fapi/v1/positionSide/dual"
    result = binance_futures_request(endpoint)
    
    return result.get("dualSidePosition", False)

def set_position_mode(dual_position=True):
    """Set position mode (hedge or one-way)"""
    endpoint = "/fapi/v1/positionSide/dual"
    params = {
        "dualSidePosition": "true" if dual_position else "false"
    }
    
    return binance_futures_request(endpoint, method='POST', params=params)

def process_trading_signal(signal):
    """Process the trading signal and execute trade"""
    try:
        ticker = signal["ticker"]
        action = signal["action"]
        quantity = signal["quantity"]
        price = signal["price"]
        position = signal["position"]
        leverage = signal["leverage"]
        
        standard_ticker = ticker.replace(".P", "")  # Remove .P suffix for API calls
        
        # Log the parsed data
        logger.info(f"Signal: {ticker}, {action}, {quantity} contracts at {price}, position: {position}, leverage: {leverage}")
        
        # Check position mode (hedge or one-way)
        is_hedge_mode = check_position_mode()
        logger.info(f"Hedge mode is {'enabled' if is_hedge_mode else 'disabled'}")
        
        if not is_hedge_mode:
            # Try to enable hedge mode, but don't fail if it can't be changed
            logger.info("Attempting to set position mode to hedge mode (dual position side)")
            try:
                set_position_mode(True)
            except Exception as e:
                logger.warning(f"Could not change position mode: {str(e)}. Continuing with current mode.")
        
        # Set leverage
        leverage_result = set_leverage(standard_ticker, leverage)
        logger.info(f"Leverage set result: {json.dumps(leverage_result)}")
        
        if "error" in leverage_result:
            return {
                "ticker": ticker,
                "action": action,
                "quantity": quantity,
                "price": price,
                "position": position,
                "status": "FAILED",
                "reason": f"Failed to set leverage: {leverage_result.get('error')}"
            }
        
        # Get current positions
        positions = get_futures_position_info()
        current_position = None
        
        for pos in positions:
            if pos.get("symbol") == standard_ticker:
                current_position = pos
                break
        
        # Determine position side based on hedge mode
        position_side = None
        if is_hedge_mode:
            if position == "long":
                position_side = "LONG"
            elif position == "short":
                position_side = "SHORT"
        
    
        # Determine the side (BUY/SELL) based on action and position
        order_side = action.upper()
        
        # Execute the trade
        order_result = execute_futures_order(standard_ticker, order_side, position_side, quantity)
        
        # Get updated account and position information
        updated_positions = get_futures_position_info()
        balance = get_futures_account_balance()
        
        # Update our position tracking
        for pos in updated_positions:
            if pos.get("symbol") == standard_ticker:
                position_size = float(pos.get("positionAmt", 0))
                entry_price = float(pos.get("entryPrice", 0))
                
                if position_size != 0:
                    # We have an active position
                    POSITIONS[ticker] = {
                        "position": "long" if position_size > 0 else "short",
                        "quantity": abs(position_size),
                        "entry_price": entry_price,
                        "leverage": leverage,
                        "entry_time": datetime.now(timezone.utc).isoformat()
                    }
                elif ticker in POSITIONS:
                    # Position was closed
                    del POSITIONS[ticker]
        
        # Format the balance data for easier reading
        formatted_balance = {}
        for bal in balance:
            symbol = bal.get("asset")
            free = float(bal.get("balance", "0"))
            if free > 0:
                formatted_balance[symbol] = free
        
        result = {
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "price": price,
            "position": position,
            "leverage": leverage,
            "order_result": order_result,
            "current_balance": formatted_balance,
            "active_positions": list(POSITIONS.keys())
        }
        
        # Save positions and processed emails after each trade
        save_positions()
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing trading signal: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def save_positions():
    """Save positions to a file for persistence between runs"""
    try:
        with open('positions.json', 'w') as f:
            json.dump(POSITIONS, f)
    except Exception as e:
        logger.error(f"Error saving positions: {str(e)}")

def load_positions():
    """Load positions from a file"""
    global POSITIONS
    try:
        if os.path.exists('positions.json'):
            with open('positions.json', 'r') as f:
                POSITIONS = json.load(f)
                logger.info(f"Loaded positions: {json.dumps(POSITIONS)}")
    except Exception as e:
        logger.error(f"Error loading positions: {str(e)}")

def save_processed_emails():
    """Save processed email IDs to a file for persistence between runs"""
    try:
        with open('processed_emails.json', 'w') as f:
            json.dump(list(PROCESSED_EMAIL_IDS), f)
    except Exception as e:
        logger.error(f"Error saving processed emails: {str(e)}")

def load_processed_emails():
    """Load processed email IDs from a file"""
    global PROCESSED_EMAIL_IDS
    try:
        if os.path.exists('processed_emails.json'):
            with open('processed_emails.json', 'r') as f:
                PROCESSED_EMAIL_IDS = set(json.load(f))
                logger.info(f"Loaded {len(PROCESSED_EMAIL_IDS)} processed email IDs")
    except Exception as e:
        logger.error(f"Error loading processed emails: {str(e)}")

def main():
    """Main function to run the email checking and trading loop"""
    logger.info("Starting email trading bot")
    
    # Load positions and processed emails from file
    load_positions()
    load_processed_emails()
    
    # Validate environment variables
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        logger.error("Email credentials not set. Please set EMAIL_USERNAME and EMAIL_PASSWORD environment variables.")
        return
    
    if not API_KEY or not API_SECRET:
        logger.error("Binance API credentials not set. Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")
        return
        
    logger.info(f"Monitoring emails from: {TRADE_SIGNAL_FROM}")
    logger.info(f"Only checking emails received after: {START_DATETIME.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Check interval: {CHECK_INTERVAL} seconds")
 
    while True:
        try:
         
            # Connect to email
            mail = connect_to_email()
            
            if not mail:
                logger.error("Failed to connect to email server. Retrying in 60 seconds...")
                time.sleep(60)
                continue
            
  
            
            # Check for trade signal emails from specific sender received since script start
            email_ids = get_emails_since_script_start(mail, TRADE_SIGNAL_FROM)
            
            if not email_ids:
                logger.info(f"No new emails found from {TRADE_SIGNAL_FROM}")
            else:
                logger.info(f"Found {len(email_ids)} new emails to process")
            
            # Process emails
            for email_id in email_ids:
                # Get email content
                email_content = get_email_content(mail, email_id)
                if not email_content:
                    logger.warning(f"Could not get content for email ID {email_id}")
                    continue
                
                # Parse the trade signal
                signal = parse_trade_signal(email_content)
                if not signal:
                    logger.warning(f"Could not parse trade signal from email ID {email_id}")
                    continue
                
                # Process the trading signal
                result = process_trading_signal(signal)
                logger.info(f"Trade result: {json.dumps(result)}")
                
                # Mark this email as processed
                PROCESSED_EMAIL_IDS.add(email_id.decode('utf-8'))
                
                # Save positions and processed emails after each trade
                save_positions()
                save_processed_emails()
            
            # Logout from email
            mail.logout()
          
            # Calculate next check time
            next_check_time = datetime.now() + timedelta(seconds=CHECK_INTERVAL)
            logger.info(f"Next check scheduled for {next_check_time.strftime('%H:%M:%S')}")
            
            # Wait before checking again
            time.sleep(CHECK_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            logger.error(traceback.format_exc())
            time.sleep(60)  # Wait longer on error
            
            # Wait longer on error
if __name__ == "__main__":
    main()
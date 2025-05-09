import imaplib
import email
import os
import hmac
import hashlib
import time
import json
import requests
from datetime import datetime,timezone
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
        mail_connection.select("INBOX")
        
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
        email_body = email_body.strip()
        lines = email_body.split()
        
        if len(lines) >= 5:
            ticker = lines[1].strip()
            action = lines[2].strip().lower()  # Normalize to lowercase
            contracts = float(lines[3].strip())
            price = float(lines[4].strip())
            market_position = lines[5].strip().lower()  # Normalize to lowercase
            
            # Validate the data
            if action not in ["buy", "sell"]:
                raise ValueError(f"Invalid action: {action}")
            
            if market_position not in ["long", "short", "flat"]:
                raise ValueError(f"Invalid market position: {market_position}")
            
            return {
                "ticker": ticker,
                "action": action,
                "contracts": contracts,
                "price": price,
                "market_position": market_position
            }
        else:
            raise ValueError(f"Invalid email format. Expected at least 5 lines, got {len(lines)}")
    except Exception as e:
        logger.error(f"Error parsing trade signal: {str(e)}")
        return None

def execute_binance_order(symbol, side, quantity, api_key, api_secret):
    """Execute an order on Binance testnet"""
    base_url = "https://testnet.binance.vision"
    endpoint = "/api/v3/order"
    timestamp = str(int(time.time() * 1000))
    
    # Prepare parameters
    params = {
        "symbol": symbol,
        "side": side.upper(),
        "type": "MARKET",
        "quantity": quantity,
        "timestamp": timestamp
    }
    
    # Generate signature
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    params["signature"] = signature
    
    # Headers
    headers = {
        "X-MBX-APIKEY": api_key
    }
    
    # Send request
    try:
        response = requests.post(f"{base_url}{endpoint}", params=params, headers=headers)
        return response.json()
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error executing Binance order: {error_msg}")
        return {"error": error_msg, "message": "Request failed but order may have been processed"}

def get_binance_balance(api_key, api_secret):
    """Get account balance from Binance testnet"""
    base_url = "https://testnet.binance.vision"
    endpoint = "/api/v3/account"
    timestamp = str(int(time.time() * 1000))
    
    # Prepare parameters
    params = {
        "timestamp": timestamp
    }
    
    # Generate signature
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    params["signature"] = signature
    
    # Headers
    headers = {
        "X-MBX-APIKEY": api_key
    }
    
    # Send request
    try:
        response = requests.get(f"{base_url}{endpoint}", params=params, headers=headers)
        
        # Process balance data
        result = response.json()
        balances = result.get("balances", [])
        
        # Format the balance data for easier reading
        formatted_balance = {}
        for balance in balances:
            symbol = balance.get("asset")
            free = float(balance.get("free", "0"))
            if free > 0:
                formatted_balance[symbol] = free
        
        return formatted_balance
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error getting Binance balance: {error_msg}")
        return {"error": error_msg, "message": "Failed to get balance"}

def process_trading_signal(signal):
    """Process the trading signal and execute trade"""
    try:
        ticker = signal["ticker"]
        action = signal["action"]
        contracts = signal["contracts"]
        price = signal["price"]
        market_position = signal["market_position"]
        
        
        # Log the parsed data
        logger.info(f"Signal: {ticker}, {action}, {contracts} contracts at {price}, position: {market_position}")
        logger.info(f"Is entry: {market_position != 'flat'}, Is exit: {market_position == 'flat'}")
        logger.info(f"Current positions: {json.dumps(POSITIONS)}")
        
        # Skip if it's an exit without a corresponding entry
        if action == "sell" and market_position == 'flat' and ticker not in POSITIONS:
            logger.warning("Attempted to exit long position that doesn't exist")
            return {
                "ticker": ticker,
                "action": action,
                "contracts": contracts,
                "price": price,
                "market_position": market_position,
                "status": "SKIPPED",
                "reason": "Attempted to sell/exit long position that doesn't exist"
            }
        
        if action == "buy" and market_position == 'flat' and ticker not in POSITIONS:
            logger.warning("Attempted to exit short position that doesn't exist")
            return {
                "ticker": ticker,
                "action": action,
                "contracts": contracts,
                "price": price,
                "market_position": market_position,
                "status": "SKIPPED",
                "reason": "Attempted to buy/exit short position that doesn't exist"
            }
        
        # Execute the trade
        order_result = execute_binance_order(ticker, action, contracts, API_KEY, API_SECRET)
        
        # Update position tracking
        if market_position != 'flat':
            POSITIONS[ticker] = {
                "position": market_position,
                "contracts": contracts,
                "entry_price": price,
                "entry_time": datetime.now(timezone.utc).isoformat()
            }
       
            trade_type = "ENTRY"
        elif market_position == 'flat':
            # Calculate P&L if it's an exit
            entry_data = POSITIONS.get(ticker, {})
            entry_price = entry_data.get("entry_price", price)
            entry_contracts = entry_data.get("contracts", 0)
            
            if entry_data.get("position") == "long":
                pnl = (price - entry_price) * contracts
            else:  # short
                pnl = (entry_price - price) * contracts
            
            # Get account balance for equity calculation
            balance = get_binance_balance(API_KEY, API_SECRET)

            # Check if we got a valid response
            if isinstance(balance, dict) and "error" not in balance:
                # Display BTC and USDT separately
                btc_balance = balance.get("ETH", 0)
                usdt_balance = balance.get("USDT", 0)
                
                
            else:
                # Handle error case
                error_message = balance.get("message", "Unknown error") if isinstance(balance, dict) else "Invalid response"
                print(f"Error retrieving balance: {error_message}")
            # Log PnL and equity information
            logger.info(f"===== TRADE CLOSED =====")
            logger.info(f"Symbol: {ticker}")
            logger.info(f"Position: {entry_data.get('position', 'unknown')}")
            logger.info(f"Entry Price: {entry_price}")
            logger.info(f"Exit Price: {price}")
            logger.info(f"Contracts: {contracts}")
            logger.info(f"P&L: {pnl:.2f}")
            logger.info(f"ETH equity: {btc_balance:.8f}")
            logger.info(f"USDT equity: {usdt_balance:.2f}")
            logger.info(f"========================")
            
            # Remove the position from tracking
            if ticker in POSITIONS:
                del POSITIONS[ticker]
            
            trade_type = "EXIT"
        else:
            # It's a position adjustment (scaling in/out)
            if ticker in POSITIONS and POSITIONS[ticker]["position"] == market_position:
                # Adjust the position size
                old_contracts = POSITIONS[ticker]["contracts"]
                old_price = POSITIONS[ticker]["entry_price"]
                
                # Calculate new average entry price
                if (action == "buy" and market_position == "long") or (action == "sell" and market_position == "short"):
                    # Scaling in
                    new_contracts = old_contracts + contracts
                    avg_price = ((old_contracts * old_price) + (contracts * price)) / new_contracts
                    
                    POSITIONS[ticker]["contracts"] = new_contracts
                    POSITIONS[ticker]["entry_price"] = avg_price
                else:
                    # Scaling out
                    new_contracts = old_contracts - contracts
                    if new_contracts <= 0:
                        # Full exit
                        if ticker in POSITIONS:
                            del POSITIONS[ticker]
                    else:
                        # Partial exit
                        POSITIONS[ticker]["contracts"] = new_contracts
            
            trade_type = "ADJUSTMENT"
        
        # Get account balance after trade
        balance = get_binance_balance(API_KEY, API_SECRET)
        
        result = {
            "ticker": ticker,
            "action": action,
            "trade_type": trade_type,
            "contracts": contracts,
            "price": price,
            "market_position": market_position,
            "order_result": order_result,
            "current_balance": balance,
            "active_positions": list(POSITIONS.keys())
        }
        
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
            start_time = time.time()
            logger.debug("Starting email check cycle")
            
            # Connect to email
            connect_start = time.time()
            mail = connect_to_email()
            connect_time = time.time() - connect_start
            logger.debug(f"Email connection time: {connect_time:.2f} seconds")
            
            if not mail:
                logger.error("Failed to connect to email server. Retrying in 60 seconds...")
                time.sleep(60)
                continue
            
            # Check for trade signal emails from specific sender received since script start
            search_start = time.time()
            email_ids = get_emails_since_script_start(mail, TRADE_SIGNAL_FROM)
            search_time = time.time() - search_start
            logger.debug(f"Email search time: {search_time:.2f} seconds")
            
            # Process emails
            process_start = time.time()
            for email_id in email_ids:
                # Get email content
                email_content = get_email_content(mail, email_id)
                if not email_content:
                    continue
                
                # Parse the trade signal
                signal = parse_trade_signal(email_content)
                if not signal:
                    continue
                
                # Process the trading signal
                result = process_trading_signal(signal)
                
                # Mark this email as processed
                PROCESSED_EMAIL_IDS.add(email_id.decode('utf-8'))
                
                # Save positions and processed emails after each trade
                save_positions()
                save_processed_emails()
            
            process_time = time.time() - process_start
            logger.debug(f"Email processing time: {process_time:.2f} seconds")
            
            # Logout from email
            mail.logout()
            
            # Calculate total cycle time
            cycle_time = time.time() - start_time
            logger.debug(f"Total email check cycle time: {cycle_time:.2f} seconds")
            
            # Calculate remaining time to sleep
            sleep_time = max(0.1, CHECK_INTERVAL - cycle_time)
            logger.debug(f"Sleeping for {sleep_time:.2f} seconds")
            
            # Wait before checking again
            time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            logger.error(traceback.format_exc())
            time.sleep(60)  # Wait longer on error

            
if __name__ == "__main__":
    main()
import imaplib
import email
import os
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
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType

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
EMAIL_USERNAME = os.environ.get("EMAIL_USERNAME2")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD2")
TRADE_SIGNAL_FROM = os.environ.get("TRADE_SIGNAL_FROM", "sender@example.com")
CHECK_INTERVAL = 10  # seconds

# Alpaca API credentials
API_KEY = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_SECRET_KEY")
PAPER = True  # Set to False for live trading

# Initialize Alpaca client
trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)

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
        email_body = email_body.strip()
        lines = email_body.split()
        print(lines)
        
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

def execute_alpaca_order(symbol, side, quantity):
    """Execute an order on Alpaca"""
    try:
        # Format the symbol for Alpaca (add USD suffix for crypto if needed)
        formatted_symbol = 'ETH/USD'
        
        # Create order request
        order_request = MarketOrderRequest(
            symbol=formatted_symbol,
            qty=quantity,
            side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC
        )
        
        # Submit the order
        logger.info(f"Submitting order to {side} {quantity} of {formatted_symbol} at market")
        order = trading_client.submit_order(order_request)
        
        return {
            "order_id": order.id,
            "symbol": formatted_symbol,
            "side": side,
            "quantity": quantity,
            "type": "market",
            "status": order.status
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error executing Alpaca order: {error_msg}")
        return {"error": error_msg, "message": "Request failed but order may have been processed"}

def get_alpaca_balance():
    """Get account balance from Alpaca"""
    try:
        # Get account information
        account = trading_client.get_account()
        
        # Format the balance data for easier reading
        formatted_balance = {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "currency": "USD"
        }
        
        # Get positions for more detailed balance
        positions = trading_client.get_all_positions()
        
        # Add positions to the balance
        position_data = {}
        for position in positions:
            symbol = position.symbol
            position_data[symbol] = {
                "quantity": float(position.qty),
                "market_value": float(position.market_value),
                "avg_entry_price": float(position.avg_entry_price),
                "current_price": float(position.current_price),
                "unrealized_pl": float(position.unrealized_pl)
            }
        
        formatted_balance["positions"] = position_data
        
        return formatted_balance
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error getting Alpaca balance: {error_msg}")
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
        order_result = execute_alpaca_order(ticker, action, contracts)
        
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
            balance = get_alpaca_balance()
            
            # Check if we got a valid response
            if isinstance(balance, dict) and "error" not in balance:
                equity = balance.get("equity", 0)
                cash = balance.get("cash", 0)
            else:
                # Handle error case
                error_message = balance.get("message", "Unknown error") if isinstance(balance, dict) else "Invalid response"
                logger.error(f"Error retrieving balance: {error_message}")
                equity = 0
                cash = 0
                
            # Log PnL and equity information
            logger.info(f"===== TRADE CLOSED =====")
            logger.info(f"Symbol: {ticker}")
            logger.info(f"Position: {entry_data.get('position', 'unknown')}")
            logger.info(f"Entry Price: {entry_price}")
            logger.info(f"Exit Price: {price}")
            logger.info(f"Contracts: {contracts}")
            logger.info(f"P&L: {pnl:.2f}")
            logger.info(f"Current Equity: {equity:.2f}")
            logger.info(f"Current Cash: {cash:.2f}")
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
        balance = get_alpaca_balance()
        
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
        with open('positions_alpaca.json', 'w') as f:
            json.dump(POSITIONS, f)
    except Exception as e:
        logger.error(f"Error saving positions: {str(e)}")

def load_positions():
    """Load positions from a file"""
    global POSITIONS
    try:
        if os.path.exists('positions_alpaca.json'):
            with open('positions_alpaca.json', 'r') as f:
                POSITIONS = json.load(f)
                logger.info(f"Loaded positions: {json.dumps(POSITIONS)}")
    except Exception as e:
        logger.error(f"Error loading positions: {str(e)}")

def save_processed_emails():
    """Save processed email IDs to a file for persistence between runs"""
    try:
        with open('processed_alpaca.json', 'w') as f:
            json.dump(list(PROCESSED_EMAIL_IDS), f)
    except Exception as e:
        logger.error(f"Error saving processed emails: {str(e)}")

def load_processed_emails():
    """Load processed email IDs from a file"""
    global PROCESSED_EMAIL_IDS
    try:
        if os.path.exists('processed_alpaca.json'):
            with open('processed_alpaca.json', 'r') as f:
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
        logger.error("Alpaca API credentials not set. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
        return
        
    logger.info(f"Monitoring emails from: {TRADE_SIGNAL_FROM}")
    logger.info(f"Only checking emails received after: {START_DATETIME.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Check interval: {CHECK_INTERVAL} seconds")
    
    
    while True:
        try:
      
            mail = connect_to_email()
      
            
            if not mail:
                logger.error("Failed to connect to email server. Retrying in 60 seconds...")
                time.sleep(60)
                continue
            
       
            # Check for trade signal emails from specific sender received since script start
            search_start = time.time()
            email_ids = get_emails_since_script_start(mail, TRADE_SIGNAL_FROM)
            search_time = time.time() - search_start
            
            if not email_ids:
                logger.info(f"No new emails found from {TRADE_SIGNAL_FROM} (search took {search_time:.2f} seconds)")
            
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

if __name__ == "__main__":
    main()
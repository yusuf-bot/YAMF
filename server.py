import requests
import hmac
import hashlib
import time
import json
import pandas as pd
import os
from datetime import datetime
import schedule
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Your Binance Testnet API credentials
# Use os.getenv to retrieve the variables from environment
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# Make sure API credentials are loaded
if not API_KEY or not API_SECRET:
    print("ERROR: API credentials not found in environment variables.")
    print("Make sure your .env file exists with BINANCE_API_KEY and BINANCE_API_SECRET set.")
    exit(1)

# Directory for CSV files
DATA_DIR = "binance_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_timestamp():
    """Get current timestamp in milliseconds"""
    return int(time.time() * 1000)

def generate_signature(query_string):
    """Generate HMAC SHA256 signature"""
    return hmac.new(
        API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def binance_request(endpoint, method="GET", params=None):
    """Make an authenticated request to Binance Testnet API"""
    base_url = "https://testnet.binance.vision/api/v3"
    url = f"{base_url}{endpoint}"
    
    # Add timestamp to params
    if params is None:
        params = {}
    
    params['timestamp'] = get_timestamp()
    
    # Generate signature
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = generate_signature(query_string)
    params['signature'] = signature
    
    # Set headers
    headers = {
        "X-MBX-APIKEY": API_KEY
    }
    
    # Make request
    if method == "GET":
        response = requests.get(url, params=params, headers=headers)
    elif method == "POST":
        response = requests.post(url, params=params, headers=headers)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Check for errors
    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        return None
    
    return response.json()

def get_account_balance():
    """Get account balance from Binance Testnet"""
    endpoint = "/account"
    result = binance_request(endpoint)
    
    if result is None:
        return None
    
    # Extract balance information
    balances = result.get("balances", [])
    
    # Filter out zero balances
    non_zero_balances = []
    for balance in balances:
        asset = balance.get("asset")
        free = float(balance.get("free", 0))
        locked = float(balance.get("locked", 0))
        total = free + locked
        
        if total > 0:
            non_zero_balances.append({
                "Asset": asset,
                "Free": free,
                "Locked": locked,
                "Total": total
            })
    
    return non_zero_balances

def get_recent_trades(symbol="BTCUSDT", limit=500):
    """Get recent trades for a specific symbol"""
    endpoint = "/myTrades"
    params = {
        "symbol": symbol,
        "limit": limit
    }
    
    result = binance_request(endpoint, params=params)
    
    if result is None or not result:
        return None
    
    # Format trade data
    trades = []
    for trade in result:
        trades.append({
            "Symbol": trade.get("symbol"),
            "Order ID": trade.get("orderId"),
            "Price": float(trade.get("price")),
            "Quantity": float(trade.get("qty")),
            "Quote Quantity": float(trade.get("quoteQty")),
            "Commission": float(trade.get("commission")),
            "Commission Asset": trade.get("commissionAsset"),
            "Time": datetime.fromtimestamp(trade.get("time") / 1000),
            "IsBuyer": trade.get("isBuyer"),
            "IsMaker": trade.get("isMaker")
        })
    
    return trades

def get_open_orders():
    """Get all open orders"""
    endpoint = "/openOrders"
    result = binance_request(endpoint)
    
    if result is None:
        return None
    
    # Format open orders data
    orders = []
    for order in result:
        orders.append({
            "Symbol": order.get("symbol"),
            "Order ID": order.get("orderId"),
            "Price": float(order.get("price")),
            "Original Quantity": float(order.get("origQty")),
            "Executed Quantity": float(order.get("executedQty")),
            "Status": order.get("status"),
            "Type": order.get("type"),
            "Side": order.get("side"),
            "Time": datetime.fromtimestamp(order.get("time") / 1000)
        })
    
    return orders

def get_all_symbols():
    """Get all trading symbols from Binance"""
    url = "https://testnet.binance.vision/api/v3/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    
    symbols = []
    for symbol_data in data.get("symbols", []):
        status = symbol_data.get("status")
        symbol = symbol_data.get("symbol")
        
        if status == "TRADING" and symbol.endswith("USDT"):
            symbols.append(symbol)
    
    return symbols

def update_portfolio_data():
    """Update all portfolio data and save to CSV files"""
    print(f"Updating portfolio data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get account balance
    balances = get_account_balance()
    if balances:
        df_balances = pd.DataFrame(balances)
        # Add timestamp
        df_balances['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to CSV with timestamp in filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{DATA_DIR}/balance_{timestamp}.csv"
        df_balances.to_csv(filename, index=False)
        
        # Also save to a "latest" file that gets overwritten
        df_balances.to_csv(f"{DATA_DIR}/balance_latest.csv", index=False)
        
        print(f"Balance data saved to {filename}")
    else:
        print("Failed to retrieve balance data")
    
    # Get open orders
    open_orders = get_open_orders()
    if open_orders:
        df_orders = pd.DataFrame(open_orders)
        # Save to CSV with timestamp in filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{DATA_DIR}/open_orders_{timestamp}.csv"
        df_orders.to_csv(filename, index=False)
        
        # Also save to a "latest" file that gets overwritten
        df_orders.to_csv(f"{DATA_DIR}/open_orders_latest.csv", index=False)
        
        print(f"Open orders data saved to {filename}")
    else:
        print("No open orders or failed to retrieve data")
    
    # Get trades for major trading pairs
    symbols = get_all_symbols()
    all_trades = []
    
    # Limit to top 10 common trading pairs to avoid excessive API calls
    trading_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']
    trading_symbols = [s for s in trading_symbols if s in symbols][:5]
    
    for symbol in trading_symbols:
        trades = get_recent_trades(symbol)
        if trades:
            all_trades.extend(trades)
            print(f"Retrieved {len(trades)} trades for {symbol}")
        time.sleep(1)  # Avoid API rate limits
    
    if all_trades:
        df_trades = pd.DataFrame(all_trades)
        # Sort by time
        df_trades = df_trades.sort_values(by='Time', ascending=False)
        
        # Save to CSV with timestamp in filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{DATA_DIR}/trades_{timestamp}.csv"
        df_trades.to_csv(filename, index=False)
        
        # Also save to a "latest" file that gets overwritten
        df_trades.to_csv(f"{DATA_DIR}/trades_latest.csv", index=False)
        
        print(f"Trade data saved to {filename}")
    else:
        print("No trades found or failed to retrieve data")

def run_scheduler():
    """Run the portfolio update on a schedule"""
    # Update immediately on start
    update_portfolio_data()
    
    # Schedule updates every hour
    schedule.every(1).hour.do(update_portfolio_data)
    
    print("Portfolio tracker running. Updates scheduled every hour.")
    print(f"Data files will be saved to {os.path.abspath(DATA_DIR)}")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

def display_portfolio_summary():
    """Display a summary of the current portfolio"""
    print("\n===== BINANCE TESTNET PORTFOLIO SUMMARY =====\n")
    
    # Get account balance
    balances = get_account_balance()
    if balances:
        print("ACCOUNT BALANCE:")
        balance_df = pd.DataFrame(balances)
        print(balance_df.to_string(index=False))
    else:
        print("Failed to retrieve balance data")
    
    # Get open orders
    print("\nOPEN ORDERS:")
    open_orders = get_open_orders()
    if open_orders:
        orders_df = pd.DataFrame(open_orders)
        print(orders_df.to_string(index=False))
    else:
        print("No open orders found")
    
    # Get recent trades for BTC
    print("\nRECENT TRADES (BTCUSDT):")
    trades = get_recent_trades("BTCUSDT", limit=10)
    if trades:
        trades_df = pd.DataFrame(trades)
        print(trades_df[["Symbol", "Price", "Quantity", "Time", "Side"]].to_string(index=False))
    else:
        print("No recent trades found")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Binance Testnet Portfolio Tracker")
    parser.add_argument("--display", action="store_true", help="Display current portfolio summary")
    parser.add_argument("--update", action="store_true", help="Update portfolio data once")
    parser.add_argument("--schedule", action="store_true", help="Run scheduled updates every hour")
    
    args = parser.parse_args()
    
    if args.display:
        display_portfolio_summary()
    elif args.update:
        update_portfolio_data()
    elif args.schedule:
        run_scheduler()
    else:
        print("Please specify an action: --display, --update, or --schedule")
        print("Example: python binance_portfolio_tracker.py --schedule")
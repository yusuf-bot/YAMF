import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Load environment variables
load_dotenv()

# Get API keys from environment variables
API_KEY = os.environ.get('ALPACA_API_KEY')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')

# Initialize Alpaca trading client (paper trading by default)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

def place_stop_order(symbol, qty, stop_price, side="buy"):
    """
    Place a stop order for a stock
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
        qty: Quantity of shares
        stop_price: Stop price to trigger the order
        side: 'buy' or 'sell'
    """
    try:
        # Create a stop order
        order_data = StopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,  # Good Till Canceled
            stop_price=stop_price
        )
        
        # Submit the order
        order = trading_client.submit_order(order_data=order_data)
        print(f"Stop {side} order placed for {qty} shares of {symbol} at ${stop_price}")
        print(f"Order ID: {order.id}")
        return order.id
    
    except Exception as e:
        print(f"Error placing stop order: {e}")
        return None

def get_account_info():
    """Get account information"""
    account = trading_client.get_account()
    print(f"Account ID: {account.id}")
    print(f"Cash: ${float(account.cash)}")
    print(f"Equity: ${float(account.equity)}")
    print(f"Buying Power: ${float(account.buying_power)}")

if __name__ == "__main__":
    # Get account information
    get_account_info()
    
    # Example: Place a stop buy order for AAPL
    symbol = "AMD"
    qty = 1  # Number of shares
    current_price = 175.0  # You would typically get this from market data
    stop_price = current_price + 2.0  # Stop price $2 above current price
    
    place_stop_order(symbol, qty, stop_price, "buy")
    
    # Example: Place a stop sell order (stop loss)
    # place_stop_order("MSFT", 1, 350.0, "sell")
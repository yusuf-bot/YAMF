from datetime import datetime, timedelta, time as dt_time
import pytz
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.environ.get('ALPACA_API_KEY')
secret_key = os.environ.get('ALPACA_SECRET_KEY')

# Initialize client
trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)

def is_market_open():
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # Check if it's a weekday
    if now.weekday() >= 5:
        print(f"It's a weekend (weekday: {now.weekday()})")
        return False
    
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    current_time = now.time()
    
    print(f"Current time (ET): {current_time}")
    print(f"Market hours: {market_open} - {market_close}")
    
    if market_open <= current_time <= market_close:
        today = now.date()
        calendar_request = GetCalendarRequest(
            start=today.isoformat(),
            end=today.isoformat()
        )
        calendar = trading_client.get_calendar(calendar_request)
        is_trading_day = len(calendar) > 0
        print(f"Is trading day: {is_trading_day}")
        return is_trading_day
        
    print("Outside market hours")
    return False

def get_next_market_open():
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # Get calendar for the next 5 trading days
    start_date = now.date()
    end_date = (now + timedelta(days=5)).date()
    
    calendar_request = GetCalendarRequest(
        start=start_date.isoformat(),
        end=end_date.isoformat()
    )
    calendar = trading_client.get_calendar(calendar_request)
    
    print(f"\nChecking market calendar from {start_date} to {end_date}")
    
    for day in calendar:
        market_open_time = datetime.combine(
            day.date,
            dt_time(9, 30)
        ).replace(tzinfo=eastern)
        
        print(f"Found trading day: {day.date}, Market opens at: {market_open_time}")
        
        if market_open_time > now:
            return market_open_time
    
    print("No trading days found in calendar, calculating next weekday")
    next_day = now
    while True:
        next_day = next_day + timedelta(days=1)
        if next_day.weekday() < 5:
            next_open = datetime.combine(
                next_day.date(),
                dt_time(9, 30)
            ).replace(tzinfo=eastern)
            return next_open

if __name__ == "__main__":
    print("Testing market hours functions...")
    print("\nChecking if market is currently open:")
    is_open = is_market_open()
    print(f"Market is {'open' if is_open else 'closed'}")
    
    print("\nGetting next market open time:")
    next_open = get_next_market_open()
    print(f"Next market opens at: {next_open}")
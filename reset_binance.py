#!/usr/bin/env python3
"""
Reset Binance Testnet Balance to exactly 100 USDT and 100 BTC
"""
import os
import time
import hmac
import hashlib
import requests
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# Binance testnet API URLs
BASE_URL = "https://testnet.binance.vision"

def sign_request(params):
    """Generate HMAC SHA256 signature for API request"""
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    return hmac.new(
        API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def api_request(method, endpoint, params=None):
    """Make an API request to Binance testnet"""
    if params is None:
        params = {}
    
    # Add timestamp to params
    params['timestamp'] = str(int(time.time() * 1000))
    
    # Generate signature
    signature = sign_request(params)
    params['signature'] = signature
    
    # Set headers
    headers = {"X-MBX-APIKEY": API_KEY}
    
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method.upper() == 'GET':
            response = requests.get(url, params=params, headers=headers)
        elif method.upper() == 'POST':
            response = requests.post(url, params=params, headers=headers)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, params=params, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error in {method} {endpoint}: {str(e)}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
        return None

def get_account_balance():
    """Get current account balance"""
    result = api_request('GET', '/api/v3/account')
    if not result:
        return {}
    
    balances = {}
    for asset in result.get('balances', []):
        symbol = asset.get('asset')
        free = float(asset.get('free', '0'))
        locked = float(asset.get('locked', '0'))
        total = free + locked
        if total > 0:
            balances[symbol] = {'free': free, 'locked': locked, 'total': total}
    
    return balances

def get_symbol_price(symbol):
    """Get current price for a symbol"""
    result = api_request('GET', '/api/v3/ticker/price', {'symbol': symbol})
    if result and 'price' in result:
        return float(result['price'])
    return None

def cancel_all_orders():
    """Cancel all open orders"""
    print("Cancelling all open orders...")
    result = api_request('DELETE', '/api/v3/openOrders')
    return result

def reset_balance():
    """Reset balance to exactly 100 USDT and 100 BTC"""
    print("Starting balance reset process...")
    
    # Step 1: Cancel all open orders
    cancel_all_orders()
    
    # Step 2: Get current balances
    balances = get_account_balance()
    print(f"Current balances: {', '.join([f'{sym}: {data['total']:.8f}' for sym, data in balances.items()])}")
    
    # Step 3: Get account asset balance reset endpoint details
    # For testnet, we can use a special endpoint to reset the account
    print("Resetting account to default values...")
    result = api_request('POST', '/api/v3/asset/reset-api-default')
    
    # Wait for reset to complete
    time.sleep(2)
    
    # Step 4: Verify new balances
    new_balances = get_account_balance()
    btc_balance = new_balances.get('BTC', {}).get('total', 0)
    usdt_balance = new_balances.get('USDT', {}).get('total', 0)
    
    print(f"\nReset completed!")
    print(f"Final balances:")
    print(f"BTC: {btc_balance:.8f}")
    print(f"USDT: {usdt_balance:.2f}")
    
    if abs(btc_balance - 100) < 0.001 and abs(usdt_balance - 100) < 0.1:
        print("✅ Successfully reset to 100 BTC and 100 USDT")
    else:
        print("⚠️ Balance reset completed but values are not exactly 100 BTC and 100 USDT")

if __name__ == "__main__":
    # Check if API keys are loaded
    if not API_KEY or not API_SECRET:
        print("Error: API_KEY and API_SECRET must be set in your .env file")
        exit(1)
    
    reset_balance()
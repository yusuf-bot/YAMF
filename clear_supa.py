import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("TABLE_NAME", "YAMF")
if not SUPABASE_URL or not SUPABASE_API_KEY or not SUPABASE_TABLE:
    raise ValueError("Missing required environment variables.")

# Supabase REST endpoint
endpoint = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"

# Headers
headers = {
    "apikey": SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

# Payload to update fields
payload = {
    "order_tracker": False,
    "qty_tracker": 0.0
}

# WHERE clause: update all rows where id > 0
params = {
    "id": "gt.0"  # change this if your table uses another unique key or zero/negative ids
}

# Send PATCH request
response = requests.patch(endpoint, headers=headers, params=params, json=payload)

if response.status_code in [200, 204]:
    print("Rows updated successfully.")
else:
    print(f"Failed to update rows: {response.status_code}")
    print(response.text)
import os
import sys
import requests
from dotenv import load_dotenv

def clear_and_setup_rows(num_rows):
    """
    Clear all existing rows and create specified number of new rows with default values.
    
    Args:
        num_rows (int): Number of rows to create
    """
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

    print(f"Starting process for {num_rows} rows...")

    # Step 1: Delete all existing rows
    print("Deleting all existing rows...")
    delete_params = {
        "id": "gte.0"  # Delete all rows (assuming id >= 0)
    }
    
    delete_response = requests.delete(endpoint, headers=headers, params=delete_params)
    
    if delete_response.status_code in [200, 204]:
        print("All existing rows deleted successfully.")
    else:
        print(f"Failed to delete existing rows: {delete_response.status_code}")
        print(delete_response.text)
        return False

    # Step 2: Insert new rows with default values
    if num_rows > 0:
        print(f"Inserting {num_rows} new rows...")
        
        # Create payload for new rows
        new_rows = []
        for i in range(num_rows):
            new_rows.append({
                "id": i,
                "order_tracker": False,
                "qty_tracker": 0.0
            })
        
        # Insert new rows
        insert_response = requests.post(endpoint, headers=headers, json=new_rows)
        
        if insert_response.status_code in [200, 201]:
            print(f"Successfully inserted {num_rows} new rows.")
            return True
        else:
            print(f"Failed to insert new rows: {insert_response.status_code}")
            print(insert_response.text)
            return False
    else:
        print("No new rows to insert (num_rows = 0).")
        return True

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python script.py <number_of_rows>")
        print("Example: python script.py 10")
        sys.exit(1)
    
    try:
        num_rows = int(sys.argv[1])
        if num_rows < 0:
            print("Error: Number of rows must be non-negative.")
            sys.exit(1)
        
        success = clear_and_setup_rows(num_rows)
        if success:
            print("Operation completed successfully!")
        else:
            print("Operation failed!")
            sys.exit(1)
            
    except ValueError:
        print("Error: Please provide a valid integer for number of rows.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
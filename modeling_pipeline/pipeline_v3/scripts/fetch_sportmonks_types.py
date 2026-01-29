import os
import json
import requests
from pathlib import Path

# Manually load .env
try:
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value
except Exception as e:
    print(f"Warning: Could not load .env: {e}")

API_KEY = os.getenv('SPORTMONKS_API_KEY')
BASE_URL = os.getenv('SPORTMONKS_BASE_URL', 'https://api.sportmonks.com/v3/football')

def fetch_types():
    """Fetch types from SportMonks API (with pagination)."""
    url = "https://api.sportmonks.com/v3/core/types"
    print(f"Fetching types from {url}...")
    
    all_types = []
    page = 1
    
    while True:
        print(f"Fetching page {page}...")
        response = requests.get(url, params={'api_token': API_KEY, 'page': page})
        
        if response.status_code != 200:
            print(f"Error fetching types: {response.status_code}")
            print(response.text)
            break

        data = response.json()
        items = data.get('data', [])
        all_types.extend(items)
        
        # Check pagination
        meta = data.get('pagination', {})
        if not meta:
            # Maybe old format or single page?
            break
            
        current_page = meta.get('current_page')
        last_page = meta.get('total_pages') # or 'last_page'
        
        if last_page is None:
             # Fallback: check if we got full page
             if not items: 
                 print("No items found, stopping.")
                 break
             # Verify if we have 'links.next'
             # For now, just assume if we got items, try next page, but protect against infinite loop
             if len(items) < data.get('pagination', {}).get('per_page', 25):
                 print("Last page reached (based on item count).")
                 break
        elif current_page >= last_page:
            break
            
        page += 1
        
    print(f"Found {len(all_types)} types total.")
    
    # Save to file
    output_file = Path('data/reference/sportmonks_types.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_types, f, indent=2)
        
    print(f"Saved types to {output_file}")
    
    # Analyze specifically
    print("Searching for known IDs:")
    target_ids = [45, 50, 41, 42, 80, 82, 52]
    
    # Create lookup map
    id_map = {t.get('id'): t for t in all_types}
    
    for tid in target_ids:
        type_obj = id_map.get(tid)
        if type_obj:
            print(f"ID {tid}: Name='{type_obj.get('name')}', Code='{type_obj.get('code')}'")
        else:
            print(f"ID {tid}: NOT FOUND")
            
    # Also search strictly for 'Possession' or 'Shots'
    print("\nSearching for 'Possession' keyword:")
    for t in all_types:
        if 'possession' in t.get('name', '').lower() or 'possession' in t.get('code', '').lower():
             print(f"ID {t.get('id')}: {t.get('name')} ({t.get('code')})")

if __name__ == "__main__":
    fetch_types()

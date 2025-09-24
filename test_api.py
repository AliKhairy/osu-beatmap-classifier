import requests
import os

# --- CONFIGURATION ---
TOKEN = os.getenv("ECHO_API_TOKEN")  # Replace with your actual Echo API Token
BASE_URL = 'https://www.echosu.com/api/beatmaps/'
HEADERS = {'Authorization': f'Token {TOKEN}'}

# --- Request Page 1 ---
print("Requesting Page 1...")
try:
    resp1 = requests.get(BASE_URL, headers=HEADERS, params={'page': 1})
    resp1.raise_for_status()
    page1_data = resp1.json()
    if page1_data:
        print(f"  Received {len(page1_data)} maps on page 1.")
        print(f"  First map on page 1: {page1_data[0].get('title')}")
    else:
        print("  Received an empty list for page 1.")
except Exception as e:
    print(f"  Error on page 1: {e}")

print("-" * 20)

# --- Request Page 2 ---
print("Requesting Page 2...")
try:
    resp2 = requests.get(BASE_URL, headers=HEADERS, params={'page': 2})
    resp2.raise_for_status()
    page2_data = resp2.json()
    if page2_data:
        print(f"  Received {len(page2_data)} maps on page 2.")
        print(f"  First map on page 2: {page2_data[0].get('title')}")
    else:
        print("  Received an empty list for page 2.")
except Exception as e:
    print(f"  Error on page 2: {e}")
import requests
import json
import urllib3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Disable warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BRIDGE_IP = os.getenv("BRIDGE_IP")  # Replace with your bridge IP if different
API_URL = f"https://{BRIDGE_IP}/api"

# The devicetype can be any string, format is usually "application_name#username_or_device"
DATA_TO_SEND = {
    "devicetype": "my_python_app#test_user"
}

print(f"Preparing to send POST request to: {API_URL}")
print(f"With data: {json.dumps(DATA_TO_SEND)}")
print("\n" + "="*60)
print("IMPORTANT: PRESS THE PHYSICAL LINK BUTTON ON YOUR HUE BRIDGE NOW!")
print("You have about 30 seconds.")
print("="*60 + "\n")

input("Press Enter here after you have pressed the button on the bridge (or if you already did just before running this script)... ")

try:
    # verify=False is used because Hue bridges use self-signed certificates
    response = requests.post(API_URL, json=DATA_TO_SEND, verify=False, timeout=10)
    
    print(f"\nResponse Status Code: {response.status_code}")
    print("Response Content:")
    # Try to pretty-print if it's JSON, otherwise print raw text
    try:
        response_json = response.json()
        print(json.dumps(response_json, indent=4))
        
        # Extract the key if successful
        if response.status_code == 200 and isinstance(response_json, list) and len(response_json) > 0:
            if "success" in response_json[0] and "username" in response_json[0]["success"]:
                app_key = response_json[0]["success"]["username"]
                print("\n" + "-"*60)
                print(f"SUCCESS! Your new Hue Application Key is: {app_key}")
                print("Copy this key and paste it into the HUE_APP_KEY variable in your test.py script.")
                print("-"*60)
            elif "error" in response_json[0]:
                 print("\n" + "!"*60)
                 print(f"API Error received: {response_json[0]['error'].get('description')}")
                 print("This usually means you didn't press the link button in time, or an invalid request.")
                 print("!"*60)
    except json.JSONDecodeError:
        print(response.text) # Print raw text if not valid JSON
        
except requests.exceptions.SSLError as e:
    print(f"SSL Error: {e}")
    print("This can happen with self-signed certificates. 'verify=False' was used, but check network/firewall if it persists.")
except requests.exceptions.Timeout:
    print(f"Request Timeout: The request to {API_URL} timed out. Is the bridge IP {BRIDGE_IP} correct and bridge online?")
except requests.exceptions.ConnectionError as e:
    print(f"Connection Error: Failed to connect to {API_URL}. Check IP and network.")
    print(f"Details: {e}")
except requests.exceptions.RequestException as e:
    print(f"An unexpected error occurred: {e}") 
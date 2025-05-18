import requests
import json
import urllib3
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Disable InsecureRequestWarning: Unverified HTTPS request is being made to host.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Configuration Loading and Validation ---
BRIDGE_IP = os.getenv("BRIDGE_IP")
HUE_APP_KEY = os.getenv("HUE_APP_KEY")

CONFIG_VALID = True
if not BRIDGE_IP:
    print("ERROR: BRIDGE_IP not found in environment variables. Ensure it's set in your .env file.")
    CONFIG_VALID = False
if not HUE_APP_KEY:
    print("ERROR: HUE_APP_KEY not found in environment variables. Ensure it's set in your .env file.")
    CONFIG_VALID = False

if CONFIG_VALID:
    BASE_URL_V2 = f"https://{BRIDGE_IP}/clip/v2"
    HEADERS_V2 = {
        "hue-application-key": HUE_APP_KEY,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
else:
    BASE_URL_V2 = None
    HEADERS_V2 = {}
    print("Essential configuration (BRIDGE_IP or HUE_APP_KEY) missing. Hue interactions will be disabled.")

# Specific Device/Service IDs from reference/hue_device_room_reference.md
OFFICE_LAMP_LIGHT_SERVICE_ID = "86c959b1-2c17-472d-bbe2-3dbb728d0df7"

# --- Helper Function ---
def _send_light_command(light_service_id: str, payload: dict) -> bool:
    """
    Sends a command (payload) to a specific light service.

    Args:
        light_service_id: The ID of the light service to control.
        payload: The JSON payload for the command (e.g., {"on": {"on": True}}).

    Returns:
        True if the command was likely successful (200 OK), False otherwise.
    """
    if not CONFIG_VALID:
        print("ERROR: Configuration is invalid. Cannot send command. Check .env for BRIDGE_IP and HUE_APP_KEY.")
        return False
    
    # CONFIG_VALID is True, so BASE_URL_V2 and HEADERS_V2 are properly initialized.
    url = f"{BASE_URL_V2}/resource/light/{light_service_id}"
    print(f"Sending PUT to {url} with payload: {json.dumps(payload)}")

    try:
        response = requests.put(url, headers=HEADERS_V2, json=payload, verify=False, timeout=10)
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
        
        response_data = response.json()
        if "errors" in response_data and response_data["errors"]:
            print(f"API Error(s) when sending command to {light_service_id}:")
            for error in response_data["errors"]:
                print(f"  - Description: {error.get('description', 'No description')}")
            return False
        
        print(f"Command successful for light {light_service_id}. Response: {response_data}")
        return True
        
    except requests.exceptions.SSLError as e:
        print(f"SSL Error: {e}")
        return False
    except requests.exceptions.Timeout:
        print(f"Request Timeout: The request to {url} timed out.")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: Failed to connect to {url}. Details: {e}")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} {e.response.reason} for URL {url}")
        try:
            error_details = e.response.json()
            print("Error details from bridge:")
            if "errors" in error_details:
                for err_item in error_details["errors"]:
                    print(f"  - {err_item.get('description')}")
            else:
                print(f"  {json.dumps(error_details)}")
        except json.JSONDecodeError:
            print(f"  Response content: {e.response.text}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"An unexpected error occurred: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error processing response. URL: {url}, Details: {e}")
        print(f"Response text: {response.text if 'response' in locals() else 'Response object not available'}")
        return False

# --- Public Functions ---
def turn_office_light_on() -> bool:
    """Turns the Office Lamp on."""
    if not CONFIG_VALID:
        print("Office light control disabled: Configuration invalid. Check .env for BRIDGE_IP and HUE_APP_KEY.")
        return False
    print("Attempting to turn Office Lamp ON...")
    payload = {"on": {"on": True}}
    return _send_light_command(OFFICE_LAMP_LIGHT_SERVICE_ID, payload)

def turn_office_light_off() -> bool:
    """Turns the Office Lamp off."""
    if not CONFIG_VALID:
        print("Office light control disabled: Configuration invalid. Check .env for BRIDGE_IP and HUE_APP_KEY.")
        return False
    print("Attempting to turn Office Lamp OFF...")
    payload = {"on": {"on": False}}
    return _send_light_command(OFFICE_LAMP_LIGHT_SERVICE_ID, payload)

# --- Test Block ---
if __name__ == "__main__":
    if not CONFIG_VALID:
        print("Cannot run tests: Configuration invalid. Check .env file for BRIDGE_IP and HUE_APP_KEY.")
    else:
        print("Testing Office Lamp controls...")

        # Example: Turn light on
        if turn_office_light_on():
            print("Office Lamp should be ON.")
        else:
            print("Failed to turn Office Lamp ON.")

        # Wait for a bit (optional, for visual confirmation)
        time.sleep(3)

        # Example: Turn light off
        if turn_office_light_off():
            print("Office Lamp should be OFF.")
        else:
            print("Failed to turn Office Lamp OFF.")

        print("\nTest finished. Uncomment sections in __main__ to test OFF or add delays.") 
import requests
import json
import urllib3
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
OFFICE_LAMP_LIGHT_SERVICE_ID = "86c959b1-2c17-472d-bbe2-3dbb728d0df7" # Replace with your actual ID if different

def _send_light_command(light_service_id: str, payload: dict, action_description: str) -> dict:
    """
    Sends a command (payload) to a specific light service.

    Args:
        light_service_id: The ID of the light service to control.
        payload: The JSON payload for the command (e.g., {"on": {"on": True}}).
        action_description: A description of the action being performed (e.g., "turn Office Lamp ON").

    Returns:
        A dictionary with "status" and "message".
    """
    if not CONFIG_VALID:
        message = f"Configuration is invalid. Cannot {action_description}. Check .env for BRIDGE_IP and HUE_APP_KEY."
        print(f"ERROR: {message}")
        return {"status": "error", "message": message}

    url = f"{BASE_URL_V2}/resource/light/{light_service_id}"
    print(f"Sending PUT to {url} with payload: {json.dumps(payload)} for action: {action_description}")

    try:
        response = requests.put(url, headers=HEADERS_V2, json=payload, verify=False, timeout=10)
        response.raise_for_status()

        response_data = response.json()
        if "errors" in response_data and response_data["errors"]:
            error_descriptions = [err.get('description', 'No description') for err in response_data["errors"]]
            message = f"API Error(s) when trying to {action_description}: {'; '.join(error_descriptions)}"
            print(message)
            return {"status": "error", "message": message}

        message = f"Successfully {action_description}."
        print(f"Command successful for light {light_service_id}. Response: {response_data}")
        return {"status": "success", "message": message}

    except requests.exceptions.SSLError as e:
        message = f"SSL Error while trying to {action_description}: {e}"
        print(message)
        return {"status": "error", "message": message}
    except requests.exceptions.Timeout:
        message = f"Request Timeout: The request to {url} timed out while trying to {action_description}."
        print(message)
        return {"status": "error", "message": message}
    except requests.exceptions.ConnectionError as e:
        message = f"Connection Error: Failed to connect to {url} while trying to {action_description}. Details: {e}"
        print(message)
        return {"status": "error", "message": message}
    except requests.exceptions.HTTPError as e:
        error_message_detail = f"{e.response.status_code} {e.response.reason}"
        try:
            error_details_json = e.response.json()
            if "errors" in error_details_json:
                descriptions = [err.get('description') for err in error_details_json["errors"]]
                error_message_detail += f" - Details: {'; '.join(filter(None, descriptions))}"
            else:
                error_message_detail += f" - {json.dumps(error_details_json)}"
        except json.JSONDecodeError:
            error_message_detail += f" - Response: {e.response.text}"
        message = f"HTTP Error while trying to {action_description}: {error_message_detail}"
        print(message)
        return {"status": "error", "message": message}
    except requests.exceptions.RequestException as e:
        message = f"An unexpected error occurred while trying to {action_description}: {e}"
        print(message)
        return {"status": "error", "message": message}
    except json.JSONDecodeError as e:
        resp_text = response.text if 'response' in locals() else 'Response object not available'
        message = f"JSON Decode Error processing response for {action_description}. URL: {url}, Details: {e}. Response text: {resp_text}"
        print(message)
        return {"status": "error", "message": message}

def turn_office_light_on() -> dict:
    """
    Turns the Office Lamp on.
    Returns a dictionary with "status" ('success' or 'error') and "message".
    """
    if not CONFIG_VALID:
        message = "Office light control disabled: Configuration invalid. Check .env for BRIDGE_IP and HUE_APP_KEY."
        print(message)
        return {"status": "error", "message": message}
    print("Attempting to turn Office Lamp ON...")
    payload = {"on": {"on": True}}
    return _send_light_command(OFFICE_LAMP_LIGHT_SERVICE_ID, payload, "turn Office Lamp ON")

def turn_office_light_off() -> dict:
    """
    Turns the Office Lamp off.
    Returns a dictionary with "status" ('success' or 'error') and "message".
    """
    if not CONFIG_VALID:
        message = "Office light control disabled: Configuration invalid. Check .env for BRIDGE_IP and HUE_APP_KEY."
        print(message)
        return {"status": "error", "message": message}
    print("Attempting to turn Office Lamp OFF...")
    payload = {"on": {"on": False}}
    return _send_light_command(OFFICE_LAMP_LIGHT_SERVICE_ID, payload, "turn Office Lamp OFF")

if __name__ == "__main__":
    if not CONFIG_VALID:
        print("Cannot run tests: Configuration invalid. Check .env file for BRIDGE_IP and HUE_APP_KEY.")
    else:
        print("Testing Office Lamp controls (from tools.py)...")

        # Test turning light on
        on_result = turn_office_light_on()
        print(f"Turn ON result: {on_result}")

        if on_result["status"] == "success":
            print("Office Lamp should be ON.")
            # Wait for a bit (optional, for visual confirmation)
            import time
            time.sleep(3)

            # Test turning light off
            off_result = turn_office_light_off()
            print(f"Turn OFF result: {off_result}")
            if off_result["status"] == "success":
                print("Office Lamp should be OFF.")
            else:
                print("Failed to turn Office Lamp OFF.")
        else:
            print("Failed to turn Office Lamp ON, skipping OFF test.")

        print("Test finished.")

import requests
import json
import urllib3 # For disabling SSL warnings
import os
from dotenv import load_dotenv
load_dotenv()

# Disable InsecureRequestWarning: Unverified HTTPS request is being made to host.
# Hue bridges often use self-signed certificates.
# In a production environment, proper certificate handling should be used.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BRIDGE_IP = os.getenv("BRIDGE_IP") # User provided IP
# IMPORTANT: You need to generate an application key on your Hue Bridge
# and replace "YOUR_HUE_APP_KEY" with it.
# This typically involves a process like the following (details may vary, consult official Hue API docs):
# 1. Make a POST request to https://<YOUR_BRIDGE_IP>/api with a JSON body
#    like: {"devicetype": "my_python_app#test_user"}
# 2. Within about 30 seconds of sending the request (or just before),
#    PRESS THE PHYSICAL LINK BUTTON on your Hue Bridge.
# 3. The response to your POST request should contain the new application key (often called "username").
#    Example success response: [{"success":{"username":"YOUR_NEW_KEY_HERE"}}]
# This "username" is what you use as the 'hue-application-key' for V2 API calls.
# Store this key securely.
HUE_APP_KEY = os.getenv("HUE_APP_KEY") # !!! User has updated this !!!

BASE_URL_V2 = f"https://{BRIDGE_IP}/clip/v2"

HEADERS_V2 = {
    "hue-application-key": HUE_APP_KEY,
    "Accept": "application/json"
}

def get_hue_resources(resource_type):
    """
    Fetches resources of a given type (e.g., 'light', 'room', 'device')
    from the Hue Bridge V2 API.
    Returns a list of resources or None if an error occurs or key is missing.
    """
    if HUE_APP_KEY == "YOUR_HUE_APP_KEY" or not HUE_APP_KEY:
        print("-" * 60)
        print("ACTION REQUIRED: Philips Hue Application Key is Missing/Not Set")
        print("-" * 60)
        print(f"Please update the 'HUE_APP_KEY' variable in 'test.py'.")
        print("You need to generate this key from your Hue Bridge.")
        print("Instructions:")
        print(f"  1. Ensure your Hue Bridge IP address is correctly set to: {BRIDGE_IP}")
        print("  2. To generate a key (this is the V1 method, but key is often reusable for V2):")
        print(f"     a. Send a POST request to: https://{BRIDGE_IP}/api")
        print("     b. The request body should be JSON, for example: ")
        print("        '''{'devicetype': 'my_hue_script#some_identifier'}'''")
        print("     c. IMPORTANT: Within 30 seconds of sending this POST request (or just before),")
        print("        you MUST press the physical round link button on top of your Hue Bridge.")
        print("     d. The JSON response from the POST request will look like:")
        print("        '''[{'success':{'username':'SOME_LONG_RANDOM_STRING_KEY'}}] '''")
        print("        The value associated with 'username' is your HUE_APP_KEY.")
        print("  3. Replace 'YOUR_HUE_APP_KEY' in the script with this new key.")
        print("For the most accurate V2 API key generation, consult the official Philips Hue developer portal.")
        print("-" * 60)
        return None

    url = f"{BASE_URL_V2}/resource/{resource_type}"
    print(f"\nAttempting to GET: {url}")
    try:
        response = requests.get(url, headers=HEADERS_V2, verify=False, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "errors" in data and data["errors"]:
            print(f"API Error(s) when fetching {resource_type}:")
            for error in data["errors"]:
                print(f"  - Description: {error.get('description', 'No description')}")
            return None
        return data.get("data", [])
    except requests.exceptions.SSLError as e:
        print(f"SSL Error: {e}")
        print("This is often due to the Hue Bridge's self-signed certificate. The script uses 'verify=False' to attempt to bypass this.")
        print("If this error persists, there might be other SSL/TLS configuration issues or network interception.")
        return None
    except requests.exceptions.Timeout:
        print(f"Request Timeout: The request to {url} timed out.")
        print(f"Ensure the Hue Bridge at IP {BRIDGE_IP} is online and responsive.")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: Failed to connect to {url}.")
        print(f"Is the IP address '{BRIDGE_IP}' correct? Is the Hue Bridge connected to the network?")
        print(f"Details: {e}")
        return None
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
        if e.response.status_code == 401 or e.response.status_code == 403:
            print("This (401/403 error) strongly suggests an issue with the 'HUE_APP_KEY'.")
            print("Please ensure it's correct, valid, and authorized on the bridge.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An unexpected error occurred: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: Failed to parse the response from the Hue Bridge as JSON.")
        print(f"URL: {url}")
        print(f"Response text: {response.text if 'response' in locals() else 'Response object not available'}")
        print(f"Details: {e}")
        return None

if __name__ == "__main__":
    print("Philips Hue Bridge Interaction Script")
    print(f"Target Bridge IP: {BRIDGE_IP}")
    if HUE_APP_KEY == "YOUR_HUE_APP_KEY" or not HUE_APP_KEY:
        print("HUE_APP_KEY: NOT CONFIGURED (see instructions in script and output)")
    else:
        print(f"HUE_APP_KEY: {'*' * (len(HUE_APP_KEY) - 4)}{HUE_APP_KEY[-4:] if len(HUE_APP_KEY) > 4 else '****'}")

    print("\n--- Fetching All Devices (for name lookups) ---")
    all_devices = get_hue_resources("device")
    device_name_map = {}
    if all_devices:
        for dev in all_devices:
            device_id = dev.get("id")
            device_name = dev.get("metadata", {}).get("name", "N/A")
            if device_id:
                device_name_map[device_id] = device_name
        print(f"Found {len(device_name_map)} devices and mapped their names.")
    else:
        print("Could not fetch devices. Light and room names might be missing.")

    print("\n--- Fetching Lights (Light Services) ---")
    light_services = get_hue_resources("light")
    if light_services is not None:
        if light_services:
            print(f"Found {len(light_services)} light service(s):")
            for light_service in light_services:
                service_id = light_service.get("id")
                owner_device_id = light_service.get("owner", {}).get("rid")
                owner_device_name = device_name_map.get(owner_device_id, "Unknown Device")
                is_on = light_service.get("on", {}).get("on", "Unknown")
                brightness = light_service.get("dimming", {}).get("brightness", "N/A")
                
                print(f"  Light Service ID: {service_id}")
                print(f"    Owner: {owner_device_name} (Device ID: {owner_device_id})")
                print(f"    On: {is_on}")
                print(f"    Brightness: {brightness}{'%' if isinstance(brightness, (int, float)) else ''}")
        else:
            print("No light services found. This could be normal, or an issue with the key/permissions.")
    else:
        print("Failed to retrieve light services. Check HUE_APP_KEY, bridge IP, and connectivity.")
        print("Review any error messages printed above.")

    print("\n--- Fetching Rooms ---")
    rooms = get_hue_resources("room")
    if rooms is not None:
        if rooms:
            print(f"Found {len(rooms)} room(s):")
            for room in rooms:
                room_id = room.get("id")
                room_name = room.get("metadata", {}).get("name", "N/A")
                children_devices = room.get("children", [])
                
                print(f"  Room ID: {room_id}")
                print(f"    Name: {room_name}")
                
                device_names_in_room = []
                for child in children_devices:
                    if child.get("rtype") == "device":
                        child_device_id = child.get("rid")
                        child_device_name = device_name_map.get(child_device_id, f"Unknown Device (ID: {child_device_id})")
                        device_names_in_room.append(child_device_name)
                
                if device_names_in_room:
                    print(f"    Contains {len(device_names_in_room)} device(s):")
                    for name in device_names_in_room:
                        print(f"      - {name}")
                else:
                    print("    Contains 0 devices (or device details not found).")
        else:
            print("No rooms found. This could be normal, or an issue with the key/permissions.")
    else:
        print("Failed to retrieve rooms. Check HUE_APP_KEY, bridge IP, and connectivity.")
        print("Review any error messages printed above.")

    print("\n" + "-"*60)
    print("Script execution finished.")
    print("Please review the output above. It should now include device names.")
    print("If this looks correct, we can proceed to create the reference documentation.")
    print("-"*60)

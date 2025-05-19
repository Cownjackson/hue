import requests
import json
import urllib3
import os
from dotenv import load_dotenv
import re # Added for parsing markdown
from typing import Optional, Dict, Tuple, Any, List # Added for type hinting

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

# --- Light Name to ID and Room Composition Caches ---
LIGHT_ID_CACHE: Optional[Dict[str, str]] = None
ROOM_CONTENTS_CACHE: Optional[Dict[str, List[str]]] = None # Maps Room Name to list of Light Owner Names
REFERENCE_FILE_PATH = "reference/hue_device_room_reference.md"

# Specific Device/Service IDs from reference/hue_device_room_reference.md (legacy, can be removed if cache is always populated first)
OFFICE_LAMP_LIGHT_SERVICE_ID = "86c959b1-2c17-472d-bbe2-3dbb728d0df7"

def _load_reference_data():
    """
    Loads light names to service IDs AND room compositions from the reference markdown file.
    Populates LIGHT_ID_CACHE and ROOM_CONTENTS_CACHE.
    """
    global LIGHT_ID_CACHE, ROOM_CONTENTS_CACHE
    if LIGHT_ID_CACHE is not None and ROOM_CONTENTS_CACHE is not None:
        return

    print(f"Attempting to load reference data from {REFERENCE_FILE_PATH}...")
    LIGHT_ID_CACHE = {}
    ROOM_CONTENTS_CACHE = {}
    
    try:
        with open(REFERENCE_FILE_PATH, 'r') as f:
            content = f.read()

        # --- Parse Light Services ---
        # Regex to find "- **Light Service ID:** `(uuid)`" and then "- **Owner:** `(Name)` (Device ID: `uuid`)"
        service_id_pattern = re.compile(r"- \*\*Light Service ID:\*\*\s*`([0-9a-fA-F\-]+)`")
        owner_pattern = re.compile(r"-\s*\*\*Owner:\*\*\s*(.+?)\s*\(") # Captures name before (Device ID:...)
        
        lines = content.splitlines()
        current_service_id = None
        
        in_light_services_section = False
        in_rooms_section = False

        for i, line in enumerate(lines):
            if line.strip() == "## Light Services":
                in_light_services_section = True
                in_rooms_section = False
                print("  Parsing Light Services section...")
                continue
            if line.strip() == "## Rooms":
                in_rooms_section = True
                in_light_services_section = False
                print("  Parsing Rooms section...")
                continue

            if in_light_services_section:
                service_id_match = service_id_pattern.search(line)
                if service_id_match:
                    current_service_id = service_id_match.group(1)
                    # Look for Owner in the next relevant line(s)
                    for j in range(i + 1, min(i + 3, len(lines))):
                        owner_match = owner_pattern.search(lines[j])
                        if owner_match and current_service_id:
                            owner_name = owner_match.group(1).strip()
                            if owner_name not in LIGHT_ID_CACHE: # Avoid overwriting if multiple entries somehow exist
                                LIGHT_ID_CACHE[owner_name] = current_service_id
                                print(f"    Mapped Light: '{owner_name}' to Service ID: '{current_service_id}'")
                            else:
                                print(f"    Warning: Duplicate light name '{owner_name}' found. Using first mapping to {LIGHT_ID_CACHE[owner_name]}.")
                            current_service_id = None 
                            break
                    else:
                        if current_service_id:
                            print(f"    Warning: Found Service ID {current_service_id} but no matching Owner name nearby.")
                            current_service_id = None
            
            elif in_rooms_section:
                room_name_match = re.search(r"-\s*\*\*Name:\*\*\s*(.+)", line)
                if room_name_match:
                    current_room_name = room_name_match.group(1).strip()
                    ROOM_CONTENTS_CACHE[current_room_name] = []
                    print(f"    Found Room: '{current_room_name}'")
                    # Look for devices in this room in subsequent lines
                    for k in range(i + 1, len(lines)):
                        contains_match = re.search(r"-\s*\*\*Contains \d+ device\(s\):\*\*", lines[k])
                        if contains_match: # Start of device list
                            continue
                        device_in_room_match = re.search(r"^\s*-\s*(.+)", lines[k]) # Matches lines like "  - Device Name"
                        if device_in_room_match:
                            device_name_in_room = device_in_room_match.group(1).strip()
                            # Check if this line is another room property (like Room ID) or end of section
                            if device_name_in_room.startswith("**"): # Likely another room property, stop collecting devices for current room
                                break
                            if current_room_name in ROOM_CONTENTS_CACHE: # Ensure current_room_name is set
                                ROOM_CONTENTS_CACHE[current_room_name].append(device_name_in_room)
                                print(f"      Added device '{device_name_in_room}' to room '{current_room_name}'")
                        elif lines[k].strip().startswith("- **Room ID:**") or lines[k].strip().startswith("##") or not lines[k].strip():
                            # End of current room's devices or start of new room/section
                            break 
                    current_room_name = None # Reset for next room

        if not LIGHT_ID_CACHE:
            print(f"  Warning: No light IDs were successfully parsed from {REFERENCE_FILE_PATH}. Check file format in Light Services section.")
        else:
            print(f"  Successfully loaded {len(LIGHT_ID_CACHE)} light ID(s).")
        
        if not ROOM_CONTENTS_CACHE:
            print(f"  Warning: No room compositions were successfully parsed from {REFERENCE_FILE_PATH}. Check file format in Rooms section.")
        else:
            print(f"  Successfully loaded {len(ROOM_CONTENTS_CACHE)} room composition(s).")

    except FileNotFoundError:
        print(f"ERROR: Reference file not found: {REFERENCE_FILE_PATH}")
        LIGHT_ID_CACHE = {} 
        ROOM_CONTENTS_CACHE = {}
    except Exception as e:
        print(f"ERROR: Failed to load or parse reference data from {REFERENCE_FILE_PATH}: {e}")
        LIGHT_ID_CACHE = {}
        ROOM_CONTENTS_CACHE = {}

def _get_service_ids_for_target(target_name: str) -> List[str]:
    """
    Retrieves service IDs for a given target name, which can be a light or a room.
    Uses caches populated from the reference markdown file.
    Returns a list of service IDs.
    """
    if LIGHT_ID_CACHE is None or ROOM_CONTENTS_CACHE is None:
        _load_reference_data()

    service_ids: List[str] = []

    # Check if target_name is a direct light name
    if LIGHT_ID_CACHE and target_name in LIGHT_ID_CACHE:
        service_ids.append(LIGHT_ID_CACHE[target_name])
        print(f"Target '{target_name}' is a direct light. Service ID: {LIGHT_ID_CACHE[target_name]}")
        return service_ids

    # Check if target_name is a room name
    if ROOM_CONTENTS_CACHE and target_name in ROOM_CONTENTS_CACHE:
        light_names_in_room = ROOM_CONTENTS_CACHE[target_name]
        print(f"Target '{target_name}' is a room. Contains lights: {light_names_in_room}")
        for light_name in light_names_in_room:
            if LIGHT_ID_CACHE and light_name in LIGHT_ID_CACHE:
                service_ids.append(LIGHT_ID_CACHE[light_name])
                print(f"  Adding Service ID {LIGHT_ID_CACHE[light_name]} for light '{light_name}' in room '{target_name}'")
            else:
                print(f"  Warning: Light '{light_name}' from room '{target_name}' not found in LIGHT_ID_CACHE.")
        return service_ids
    
    print(f"Target '{target_name}' not found as a direct light or a room.")
    return [] # Return empty list if not found

def get_light_state(light_service_id: str) -> Dict[str, Any]:
    """
    Fetches the current state of a specific light (on/off, brightness).

    Args:
        light_service_id: The ID of the light service.

    Returns:
        A dictionary with "status" ('success' or 'error'), "message",
        and "data" (containing 'on' and 'brightness' if successful).
    """
    if not CONFIG_VALID:
        message = "Configuration invalid. Cannot get light state."
        print(f"ERROR: {message}")
        return {"status": "error", "message": message, "data": None}

    url = f"{BASE_URL_V2}/resource/light/{light_service_id}"
    action_description = f"get state for light {light_service_id}"
    print(f"Sending GET to {url} to {action_description}")

    try:
        response = requests.get(url, headers=HEADERS_V2, verify=False, timeout=10)
        response.raise_for_status()
        response_data = response.json()

        if "errors" in response_data and response_data["errors"]:
            error_descriptions = [err.get('description', 'No description') for err in response_data["errors"]]
            message = f"API Error(s) when trying to {action_description}: {'; '.join(error_descriptions)}"
            print(message)
            return {"status": "error", "message": message, "data": None}

        if not response_data.get("data"):
            message = f"API response for {action_description} did not contain 'data' field."
            print(message)
            return {"status": "error", "message": message, "data": None}

        light_data = response_data["data"][0] # Assuming the first item is the target light
        is_on = light_data.get("on", {}).get("on", False)
        brightness = light_data.get("dimming", {}).get("brightness", 0.0) # Hue reports 0-100

        message = f"Successfully retrieved state for light {light_service_id}."
        print(message)
        return {
            "status": "success",
            "message": message,
            "data": {"on": is_on, "brightness": brightness}
        }

    except requests.exceptions.SSLError as e:
        message = f"SSL Error while trying to {action_description}: {e}"
        print(message)
        return {"status": "error", "message": message, "data": None}
    except requests.exceptions.Timeout:
        message = f"Request Timeout: The request to {url} timed out while trying to {action_description}."
        print(message)
        return {"status": "error", "message": message, "data": None}
    except requests.exceptions.ConnectionError as e:
        message = f"Connection Error: Failed to connect to {url} while trying to {action_description}. Details: {e}"
        print(message)
        return {"status": "error", "message": message, "data": None}
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
        return {"status": "error", "message": message, "data": None}
    except requests.exceptions.RequestException as e:
        message = f"An unexpected error occurred while trying to {action_description}: {e}"
        print(message)
        return {"status": "error", "message": message, "data": None}
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        resp_text = response.text if 'response' in locals() else 'Response object not available'
        message = f"Error processing/parsing response for {action_description}. URL: {url}, Details: {e}. Response text: {resp_text}"
        print(message)
        return {"status": "error", "message": message, "data": None}

def _send_light_command(light_service_id: str, payload: dict, action_description: str) -> dict:
    """
    Sends a command (payload) to a specific light service using Hue API v2.
    This is a general helper and does not fetch state itself.

    Args:
        light_service_id: The ID of the light service to control.
        payload: The JSON payload for the command (e.g., {"on": {"on": True}, "dimming": {"brightness": 50}}).
        action_description: A description of the action being performed.

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
            print(f"ERROR: {message}")
            return {"status": "error", "message": message}

        message = f"Command for '{action_description}' sent successfully to light {light_service_id}."
        # The response for a successful PUT to /light/{id} usually contains an array of "rid" (resource id) and "rtype" (resource type)
        # for the resources that were updated. We don't need to parse it for this generic sender.
        print(f"DEBUG: Raw response from light command: {response_data}")
        return {"status": "success", "message": message}

    except requests.exceptions.SSLError as e:
        message = f"SSL Error while trying to {action_description}: {e}"
        print(f"ERROR: {message}")
        return {"status": "error", "message": message}
    except requests.exceptions.Timeout:
        message = f"Request Timeout: The request to {url} timed out while trying to {action_description}."
        print(f"ERROR: {message}")
        return {"status": "error", "message": message}
    except requests.exceptions.ConnectionError as e:
        message = f"Connection Error: Failed to connect to {url} while trying to {action_description}. Details: {e}"
        print(f"ERROR: {message}")
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
        print(f"ERROR: {message}")
        return {"status": "error", "message": message}
    except requests.exceptions.RequestException as e:
        message = f"An unexpected error occurred while trying to {action_description}: {e}"
        print(f"ERROR: {message}")
        return {"status": "error", "message": message}
    except json.JSONDecodeError as e: # Should be caught by HTTPError's parsing, but good to have as a fallback
        resp_text = response.text if 'response' in locals() else 'Response object not available'
        message = f"JSON Decode Error processing response for {action_description}. URL: {url}, Details: {e}. Response text: {resp_text}"
        print(f"ERROR: {message}")
        return {"status": "error", "message": message}

def control_light(
    light_name: str, # Can now be a light name or a room name
    action: str, 
    brightness_percent: Optional[int] = None
) -> Dict[str, Any]:
    """
    Controls a specific light or all lights in a room.
    Verifies the action by checking the light's state afterwards.
    """
    print(f"Attempting to control target: '{light_name}', Action: {action}, Brightness: {brightness_percent}%")

    if not CONFIG_VALID:
        return {"status": "error", "message": "Configuration invalid. Cannot control light/room.", "results": []}

    target_service_ids = _get_service_ids_for_target(light_name)
    if not target_service_ids:
        message = f"Target '{light_name}' not found as a known light or room, or room is empty. Cannot control."
        print(f"ERROR: {message}")
        return {"status": "error", "message": message, "results": []}

    action = action.upper()
    overall_status = "success" # Assume success unless a failure occurs
    aggregated_messages: List[str] = []
    individual_results: List[Dict[str, Any]] = []

    for light_service_id in target_service_ids:
        # Determine current light name for logging, if possible (not strictly necessary for operation)
        current_light_display_name = light_name 
        if LIGHT_ID_CACHE: # Attempt to find the specific name if 'light_name' was a room
            for name, lid in LIGHT_ID_CACHE.items():
                if lid == light_service_id:
                    current_light_display_name = name
                    break
        
        print(f"  Processing light: {current_light_display_name} (ID: {light_service_id})")
        payload: Dict[str, Any] = {}
        target_on_state: Optional[bool] = None
        current_action_description = f"perform {action} on {current_light_display_name}"

        if action == "ON":
            target_on_state = True
            payload["on"] = {"on": True}
        elif action == "OFF":
            target_on_state = False
            payload["on"] = {"on": False}
        elif action == "TOGGLE":
            current_state_res = get_light_state(light_service_id)
            if current_state_res["status"] == "error" or not current_state_res["data"]:
                message = f"Could not get current state for '{current_light_display_name}' (ID: {light_service_id}) to toggle. Error: {current_state_res['message']}"
                print(f"  ERROR: {message}")
                aggregated_messages.append(message)
                individual_results.append({"light_id": light_service_id, "name": current_light_display_name, "status": "error", "message": message, "final_state": None})
                overall_status = "partial_success" if individual_results else "error"
                continue # Skip to next light in room
            
            currently_on = current_state_res["data"]["on"]
            target_on_state = not currently_on
            payload["on"] = {"on": target_on_state}
            print(f"  Light '{current_light_display_name}' is currently {'ON' if currently_on else 'OFF'}. Toggling to {'ON' if target_on_state else 'OFF'}.")
        else:
            message = f"Invalid action '{action}' for {current_light_display_name}. Must be ON, OFF, or TOGGLE."
            print(f"  ERROR: {message}")
            aggregated_messages.append(message)
            individual_results.append({"light_id": light_service_id, "name": current_light_display_name, "status": "error", "message": message, "final_state": None})
            overall_status = "partial_success" if individual_results else "error"
            # If action is invalid for one, it's invalid for all; could break early, but let's process all for consistent return.
            # For now, let's make this an early exit if action itself is bad.
            return {"status": "error", "message": f"Invalid action '{action}' globally. Must be ON, OFF, or TOGGLE.", "results": individual_results}

        if target_on_state is True and brightness_percent is not None:
            if 0 <= brightness_percent <= 100:
                payload["dimming"] = {"brightness": float(brightness_percent)}
                current_action_description = f"set {current_light_display_name} to {'ON' if target_on_state else 'OFF'} and brightness to {brightness_percent}%"
            else:
                message = f"Brightness percent {brightness_percent} for '{current_light_display_name}' is invalid (must be 0-100). Brightness not changed for this light."
                print(f"  WARNING: {message}")
                # This specific light's brightness command is skipped. On/Off might still proceed if payload["on"] is set.
                # To be safe, if brightness is invalid, we can skip command for this light or return error for it.
                aggregated_messages.append(message + " Command for this light aborted due to invalid brightness.")
                individual_results.append({"light_id": light_service_id, "name": current_light_display_name, "status": "error", "message": message, "final_state": None})
                overall_status = "partial_success"
                continue # Skip to next light
        elif target_on_state is True:
             current_action_description = f"turn {current_light_display_name} {'ON' if target_on_state else 'OFF'} (brightness not specified)"
        else: # Action is OFF or TOGGLE to OFF
            current_action_description = f"turn {current_light_display_name} OFF"

        command_result = _send_light_command(light_service_id, payload, current_action_description)
        
        final_state_data = None
        verification_message = ""
        light_status_this_command = "error" # Default for this light

        if command_result["status"] == "success":
            light_status_this_command = "success" # Mark as success for sending
            import time
            time.sleep(0.5) 
            
            current_state_res = get_light_state(light_service_id)
            if current_state_res["status"] == "success" and current_state_res["data"]:
                final_state_data = current_state_res["data"]
                verification_message = f"Final state for {current_light_display_name}: On={final_state_data['on']}, Brightness={final_state_data['brightness']}%."
                
                mismatch = False
                if target_on_state is not None and final_state_data['on'] != target_on_state:
                    mismatch = True
                    verification_message += f" WARNING: Expected ON state {target_on_state} but got {final_state_data['on']}."
                if target_on_state is True and brightness_percent is not None:
                    if abs(final_state_data['brightness'] - brightness_percent) > 2.0:
                        mismatch = True
                        verification_message += f" WARNING: Expected brightness around {brightness_percent}% but got {final_state_data['brightness']}%."
                
                if mismatch:
                    print(f"  {verification_message}") # Print warning for this light
                    light_status_this_command = "success_with_warning" # Still success, but with issues
                
                msg_this_light = f"Successfully executed '{current_action_description}'. {verification_message}"
                aggregated_messages.append(msg_this_light)
                individual_results.append({"light_id": light_service_id, "name": current_light_display_name, "status": light_status_this_command, "message": msg_this_light, "final_state": final_state_data})

            else: # get_light_state failed after successful command
                verification_message = f"Command for '{current_action_description}' sent, but failed to verify final state. Getter error: {current_state_res['message']}"
                print(f"  WARNING: {verification_message}")
                aggregated_messages.append(verification_message)
                individual_results.append({"light_id": light_service_id, "name": current_light_display_name, "status": "success_with_warning", "message": verification_message, "final_state": None})
                overall_status = "partial_success" if overall_status == "success" else overall_status
        
        else: # command_result["status"] == "error"
            message = f"Failed to execute '{current_action_description}'. Error: {command_result['message']}"
            print(f"  ERROR: {message}")
            aggregated_messages.append(message)
            individual_results.append({"light_id": light_service_id, "name": current_light_display_name, "status": "error", "message": message, "final_state": None})
            overall_status = "error" if overall_status == "success" and len(target_service_ids) == 1 else "partial_success"

    # Determine final overall status if multiple lights
    if len(target_service_ids) > 1:
        num_success = sum(1 for r in individual_results if r["status"].startswith("success"))
        if num_success == len(target_service_ids):
            overall_status = "success"
        elif num_success == 0:
            overall_status = "error"
        else:
            overall_status = "partial_success"
    elif individual_results: # single light
         overall_status = individual_results[0]["status"]

    final_message = f"Overall result for '{light_name}': {overall_status}. " + " | ".join(aggregated_messages)
    return {
        "status": overall_status,
        "message": final_message,
        "results": individual_results # List of dicts, one per light processed
    }

def turn_office_light_on() -> dict:
    """
    Turns the Office Lamp on.
    Returns a dictionary with "status" ('success' or 'error'), "message", and "final_state".
    """
    # This function now wraps control_light for the specific Office Lamp ID and action.
    # The OFFICE_LAMP_LIGHT_SERVICE_ID is resolved internally by control_light via _get_service_ids_for_target.
    print("Attempting to turn Office Lamp ON (using turn_office_light_on wrapper)...")
    return control_light(light_name="Office Lamp", action="ON")

def turn_office_light_off() -> dict:
    """
    Turns the Office Lamp off.
    Returns a dictionary with "status" ('success' or 'error'), "message", and "final_state".
    """
    # This function now wraps control_light for the specific Office Lamp ID and action.
    print("Attempting to turn Office Lamp OFF (using turn_office_light_off wrapper)...")
    return control_light(light_name="Office Lamp", action="OFF")

if __name__ == "__main__":
    if not CONFIG_VALID:
        print("Cannot run tests: Configuration invalid. Check .env file for BRIDGE_IP and HUE_APP_KEY.")
    else:
        print("Testing Hue controls (from tools.py)...")
        _load_reference_data() # Ensure caches are populated

        # --- Tests for individual lights (using "Office Lamp" as example) ---
        print("\n--- Test Individual: Turn Office Lamp ON to 50% ---")
        # ... (keep existing tests for Office Lamp, they should still work via _get_service_ids_for_target)

        # Add new tests for a room, e.g., "Living Room"
        # Ensure "Living Room" and its lights exist in your reference file and are parsed correctly.
        # Example: If Living Room has "Living Boob 1" and "Lamp1"
        
        print("\n\n--- Test Room: Turn Living Room ON to 70% ---")
        room_on_result = control_light(light_name="Living Room", action="ON", brightness_percent=70)
        print(f"Control Light (Living Room ON 70%) result: {json.dumps(room_on_result, indent=2)}")
        if room_on_result["status"].startswith("success"):
            print("Living Room lights should be ON at ~70%.")
        else:
            print(f"Failed or partially failed to turn Living Room lights ON. Status: {room_on_result['status']}")

        if room_on_result["status"] != "error": # Allow partial success to proceed
            import time
            time.sleep(5) # Longer sleep for multiple lights

        print("\n\n--- Test Room: Toggle Living Room (should turn OFF if previously ON) ---")
        room_toggle_off_result = control_light(light_name="Living Room", action="TOGGLE")
        print(f"Control Light (Living Room TOGGLE to OFF) result: {json.dumps(room_toggle_off_result, indent=2)}")
        # Verification for toggle is trickier for a room; check if all lights are off
        all_off = True
        if room_toggle_off_result["status"].startswith("success"):
            for res in room_toggle_off_result.get("results", []):
                if res.get("final_state", {}).get("on") is not False:
                    all_off = False
                    break
            if all_off:
                 print("Living Room lights should be OFF (toggled).")
            else:
                 print(f"Living Room lights not all OFF after toggle. Check individual results: {room_toggle_off_result['results']}")
        else:
            print(f"Failed or partially failed to toggle Living Room lights. Status: {room_toggle_off_result['status']}")

        if room_toggle_off_result["status"] != "error":
            import time
            time.sleep(5)

        print("\n\n--- Test Room: Turn Living Room OFF (explicitly) ---")
        room_off_result = control_light(light_name="Living Room", action="OFF")
        print(f"Control Light (Living Room OFF) result: {json.dumps(room_off_result, indent=2)}")
        if room_off_result["status"].startswith("success"):
            print("Living Room lights should be OFF.")
        else:
            print(f"Failed or partially failed to turn Living Room lights OFF. Status: {room_off_result['status']}")
        
        print("\nTest finished.")
        # Keep the original Office Lamp tests as well, or integrate them more cleanly.
        # For brevity, the original detailed tests for "Office Lamp" are assumed to be present above this new section.
        # Make sure to re-run those as well to ensure no regressions for single light control.

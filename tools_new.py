import requests
import json
import urllib3
import os
from dotenv import load_dotenv
from typing import Optional, Dict, List, Any

# Load environment variables from .env file
load_dotenv()

# Disable InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Configuration ---
BRIDGE_IP = os.getenv("BRIDGE_IP")
HUE_APP_KEY = os.getenv("HUE_APP_KEY")

CONFIG_VALID = True
if not BRIDGE_IP:
    print("ERROR: BRIDGE_IP not found in environment variables. Ensure it's set in your .env file.")
    CONFIG_VALID = False
if not HUE_APP_KEY:
    print("ERROR: HUE_APP_KEY not found in environment variables. Ensure it's set in your .env file.")
    CONFIG_VALID = False

BASE_URL_V2 = f"https://{BRIDGE_IP}/clip/v2" if CONFIG_VALID else None
HEADERS_V2 = {
    "hue-application-key": HUE_APP_KEY,
    "Accept": "application/json",
    "Content-Type": "application/json", # Added for PUT requests later
} if CONFIG_VALID else {}

# --- Hue System Profile Cache ---
# This will store the structured data about devices, lights, and rooms.
HUE_SYSTEM_PROFILE: Optional[Dict[str, Dict[str, Any]]] = None
# Example structure:
# HUE_SYSTEM_PROFILE = {
#     "devices": {"device_id_1": {"id": "...", "name": "...", "product_name": "..."}},
#     "lights": {"light_service_id_1": {"id": "...", "name": "Light derived name", "on": True, "brightness": 50.0, "owner_device_id": "...", "owner_device_name": "..."}},
#     "rooms": {"room_id_1": {"id": "...", "name": "...", "light_service_ids": ["ls_id_1", ...], "light_names": ["Light derived name 1", ...]}},
# }

def _fetch_hue_api_resources(resource_type: str) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches all resources of a given type (e.g., 'device', 'light', 'room')
    from the Hue Bridge V2 API.
    """
    if not CONFIG_VALID:
        print(f"ERROR: Configuration invalid. Cannot fetch Hue resource: {resource_type}")
        return None

    url = f"{BASE_URL_V2}/resource/{resource_type}"
    # print(f"Fetching Hue resource: {url}") # Verbose, can be enabled for debugging
    try:
        response = requests.get(url, headers=HEADERS_V2, verify=False, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "errors" in data and data["errors"]:
            error_descriptions = [err.get('description', 'No description') for err in data["errors"]]
            print(f"API Error(s) when fetching {resource_type}: {'; '.join(error_descriptions)}")
            return None
        return data.get("data", [])
    except requests.exceptions.RequestException as e:
        print(f"Request Exception while fetching {resource_type} from {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error while fetching {resource_type}. URL: {url}, Details: {e}. Response: {response.text if 'response' in locals() else 'N/A'}")
        return None

def build_hue_system_profile(force_refresh: bool = False) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Builds a comprehensive profile of the Hue system by fetching data for
    devices, lights, and rooms. Caches the profile.

    Args:
        force_refresh: If True, fetches data from the API even if a profile is cached.

    Returns:
        The system profile dictionary, or None if critical data cannot be fetched.
    """
    global HUE_SYSTEM_PROFILE
    if HUE_SYSTEM_PROFILE is not None and not force_refresh:
        print("Returning cached Hue system profile.")
        return HUE_SYSTEM_PROFILE

    print("Building Hue system profile from API...")
    if not CONFIG_VALID:
        print("ERROR: Configuration invalid. Cannot build Hue system profile.")
        return None

    raw_devices = _fetch_hue_api_resources("device")
    raw_lights = _fetch_hue_api_resources("light") # These are light services
    raw_rooms = _fetch_hue_api_resources("room")

    if raw_devices is None or raw_lights is None or raw_rooms is None:
        print("ERROR: Failed to fetch one or more critical resources (devices, lights, or rooms). Profile not built.")
        return None

    profile: Dict[str, Dict[str, Any]] = {
        "devices": {},
        "lights": {},
        "rooms": {}
    }

    # 1. Process devices
    for dev in raw_devices:
        dev_id = dev.get("id")
        if dev_id:
            profile["devices"][dev_id] = {
                "id": dev_id,
                "name": dev.get("metadata", {}).get("name", "Unknown Device"),
                "product_name": dev.get("product_data", {}).get("product_name", "N/A")
            }
    print(f"  Profiled {len(profile['devices'])} devices.")

    # 2. Process light services
    for light_service in raw_lights:
        ls_id = light_service.get("id")
        owner_rid = light_service.get("owner", {}).get("rid") # This is the device ID
        
        if ls_id and owner_rid:
            owner_device_info = profile["devices"].get(owner_rid)
            # Light services often don't have their own "friendly name" directly.
            # We'll use the owning device's name as the primary name for the light.
            light_name = owner_device_info["name"] if owner_device_info else "Unnamed Light"
            
            profile["lights"][ls_id] = {
                "id": ls_id,
                "name": light_name, # Using device name as light name
                "on": light_service.get("on", {}).get("on", False),
                "brightness": light_service.get("dimming", {}).get("brightness", 0.0),
                # Add other relevant states like color if needed in future
                "owner_device_id": owner_rid,
                "owner_device_name": owner_device_info["name"] if owner_device_info else "Unknown Owner"
            }
    print(f"  Profiled {len(profile['lights'])} light services.")

    # 3. Process rooms and link light services to them
    for room in raw_rooms:
        room_id = room.get("id")
        if room_id:
            room_name = room.get("metadata", {}).get("name", "Unknown Room")
            child_rids_in_room: List[str] = [] # Device RIDs that are children of the room
            
            for child_ref in room.get("children", []):
                if child_ref.get("rtype") == "device":
                    child_rids_in_room.append(child_ref.get("rid"))
            
            light_service_ids_in_room: List[str] = []
            light_names_in_room: List[str] = []

            # Find all light services whose owner_device_id is in child_rids_in_room
            for ls_id, light_data in profile["lights"].items():
                if light_data["owner_device_id"] in child_rids_in_room:
                    light_service_ids_in_room.append(ls_id)
                    light_names_in_room.append(light_data["name"]) # Use the derived light name

            profile["rooms"][room_id] = {
                "id": room_id,
                "name": room_name,
                "device_ids": child_rids_in_room, # Store device IDs directly in room
                "light_service_ids": sorted(list(set(light_service_ids_in_room))), # Unique, sorted
                "light_names": sorted(list(set(light_names_in_room))) # Unique, sorted
            }
    print(f"  Profiled {len(profile['rooms'])} rooms and linked their lights.")

    HUE_SYSTEM_PROFILE = profile
    print("Hue system profile successfully built and cached.")
    return HUE_SYSTEM_PROFILE

# --- Getter functions for easy access to the profile ---
def get_hue_profile(force_refresh: bool = False) -> Optional[Dict[str, Dict[str, Any]]]:
    """Ensures the profile is built and returns it."""
    if HUE_SYSTEM_PROFILE is None or force_refresh:
        build_hue_system_profile(force_refresh=True)
    return HUE_SYSTEM_PROFILE

def get_light_by_name(light_name: str, profile: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """Gets a light's details by its name from the profile."""
    if profile is None:
        profile = get_hue_profile()
    if not profile or "lights" not in profile:
        return None
    for light_data in profile["lights"].values():
        if light_data["name"].lower() == light_name.lower():
            return light_data
    return None

def get_room_by_name(room_name: str, profile: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """Gets a room's details by its name from the profile."""
    if profile is None:
        profile = get_hue_profile()
    if not profile or "rooms" not in profile:
        return None
    for room_data in profile["rooms"].values():
        if room_data["name"].lower() == room_name.lower():
            return room_data
    return None

# --- Main test block ---
if __name__ == "__main__":
    print("Building and displaying Hue System Profile...")
    profile = build_hue_system_profile(force_refresh=True)

    if profile:
        print("\n--- System Profile Summary ---")
        print(f"Total Devices: {len(profile['devices'])}")
        print(f"Total Light Services: {len(profile['lights'])}")
        print(f"Total Rooms: {len(profile['rooms'])}")

        print("\n--- Devices ---")
        for dev_id, dev_data in profile["devices"].items():
            print(f"  ID: {dev_id}, Name: {dev_data['name']}, Product: {dev_data['product_name']}")

        print("\n--- Light Services ---")
        for ls_id, ls_data in profile["lights"].items():
            print(f"  ID: {ls_id}, Name: {ls_data['name']} (Owner: {ls_data['owner_device_name']}), On: {ls_data['on']}, Brightness: {ls_data['brightness']}%")
        
        print("\n--- Rooms ---")
        for room_id, room_data in profile["rooms"].items():
            print(f"  ID: {room_id}, Name: {room_data['name']}")
            print(f"    Light Names: {', '.join(room_data['light_names'] if room_data['light_names'] else ['None'])}")
            # print(f"    Light Service IDs: {', '.join(room_data['light_service_ids'] if room_data['light_service_ids'] else ['None'])}")
            # print(f"    Device IDs in room: {', '.join(room_data['device_ids'] if room_data['device_ids'] else ['None'])}")

        print("\n--- Testing Getters ---")
        # Assuming you have an "Office Lamp" and a "Living Room"
        office_lamp = get_light_by_name("Office Lamp", profile=profile)
        if office_lamp:
            print(f"Found Office Lamp by name: {office_lamp['name']}, ID: {office_lamp['id']}, On: {office_lamp['on']}")
        else:
            print("Office Lamp not found by name (this is okay if it doesn't exist or naming differs).")

        living_room = get_room_by_name("Living Room", profile=profile)
        if living_room:
            print(f"Found Living Room by name: {living_room['name']}, ID: {living_room['id']}")
            print(f"  Lights in Living Room: {', '.join(living_room['light_names'])}")
        else:
            print("Living Room not found by name (this is okay if it doesn't exist).")
            
    else:
        print("Failed to build Hue system profile. Cannot display details.")

    print("\nProfile building test finished.")

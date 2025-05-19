import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import os
import traceback
from dotenv import load_dotenv

# For serving the HTML file
from fastapi.responses import FileResponse 

from google import genai
from google.genai import types

# Import tools and their specific functions
from tools import turn_office_light_on, turn_office_light_off

load_dotenv()

# Configuration from google_ai_sample.py
MODEL = "models/gemini-1.5-flash-latest"

# Initialize the Gemini client
# Ensure your GEMINI_API_KEY is set in your .env file
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    print("ERROR: GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    # Potentially exit or raise an error if the key is critical for startup
    # For now, we'll let it proceed, but client initialization might fail.

genai_client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=gemini_api_key,
)

# --- TEMPORARY: List available models ---
print("--- Listing Available Gemini Models (v1beta) ---")
try:
    for m in genai_client.models.list():
        # print(m)
        # We are looking for models that support 'bidiGenerateContent' or similar live features
        # The exact way to check this via model properties isn't directly obvious from `m` alone
        # So we'll print name and display name and manually inspect or look for clues.
        print(f"Model Name: {m.name}, Display Name: {m.display_name}, Supported Generation Methods: {m.supported_generation_methods}")
except Exception as e:
    print(f"Error listing models: {e}")
print("--- Finished Listing Models ---")
# --- END TEMPORARY ---

# Define the tools for the voice agent, similar to google_ai_sample.py
TOOLS_CONFIG = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="turn_office_light_on",
                description="Turns the office lamp on.",
                parameters=types.Schema(type=types.Type.OBJECT, properties={}),
            ),
        ]
    ),
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="turn_office_light_off",
                description="Turns the office lamp off.",
                parameters=types.Schema(type=types.Type.OBJECT, properties={}),
            ),
        ]
    )
]

LIVE_CONNECT_CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
        )
    ),
    tools=TOOLS_CONFIG,
)

app = FastAPI()

async def handle_gemini_tool_call(session, server_tool_call: types.LiveServerToolCall):
    """
    Handles tool calls received from the Gemini API.
    Adapted from google_ai_sample.py.
    """
    print(f"Server tool call received: {server_tool_call}")
    function_responses = []
    if server_tool_call.function_calls:
        for fc in server_tool_call.function_calls:
            tool_id = fc.id
            tool_name = fc.name
            print(f"Executing tool: {tool_name} (ID: {tool_id})")
            tool_result_dict = None
            try:
                if tool_name == "turn_office_light_on":
                    tool_result_dict = turn_office_light_on()
                elif tool_name == "turn_office_light_off":
                    tool_result_dict = turn_office_light_off()
                else:
                    tool_result_dict = {"status": "error", "message": f"Tool '{tool_name}' not found."}
                    print(tool_result_dict["message"])
                
                function_responses.append(types.FunctionResponse(
                    id=tool_id,
                    name=tool_name,
                    response=tool_result_dict
                ))
            except Exception as e:
                print(f"An exception occurred while executing tool {tool_name} (ID: {tool_id}): {e}")
                function_responses.append(types.FunctionResponse(
                    id=tool_id,
                    name=tool_name,
                    response={"status": "error", "message": f"Exception during {tool_name} execution: {str(e)}"}
                ))

    if function_responses:
        print(f"Sending tool responses to Gemini: {function_responses}")
        await session.send_tool_response(function_responses=function_responses)
    else:
        print("No function calls were processed or found in the server_tool_call message.")


class WebAudioHandler:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.audio_from_gemini_queue = asyncio.Queue()  # Audio from Gemini to client (bytes)
        self.text_from_gemini_queue = asyncio.Queue()   # Text from Gemini to client (str)
        self.audio_to_gemini_queue = asyncio.Queue()    # Audio from client to Gemini (PCM bytes chunks)
        self.text_to_gemini_queue = asyncio.Queue()     # Text from client to Gemini (str)

        self.gemini_session = None
        self._stop_event = asyncio.Event()
        self._tasks = [] # To keep track of created tasks for cleanup

    async def _send_audio_to_client_task(self):
        # Renamed from _send_audio_to_client to avoid conflict if we add a direct send method
        try:
            while not self._stop_event.is_set():
                try:
                    audio_chunk = await asyncio.wait_for(self.audio_from_gemini_queue.get(), timeout=0.1)
                    if audio_chunk is None: break # Sentinel
                    if self.websocket.application_state == WebSocketState.CONNECTED:
                        await self.websocket.send_bytes(audio_chunk)
                    else:
                        print("WebSocket not connected in _send_audio_to_client_task, stopping.")
                        self._stop_event.set()
                        break
                except asyncio.TimeoutError:
                    continue
        except Exception as e:
            if not self._stop_event.is_set(): print(f"Error in _send_audio_to_client_task: {e}")
        finally:
            print("_send_audio_to_client_task finished.")

    async def _send_text_to_client_task(self):
        # Renamed from _send_text_to_client
        try:
            while not self._stop_event.is_set():
                try:
                    text_message = await asyncio.wait_for(self.text_from_gemini_queue.get(), timeout=0.1)
                    if text_message is None: break # Sentinel
                    if self.websocket.application_state == WebSocketState.CONNECTED:
                        await self.websocket.send_text(text_message)
                    else:
                        print("WebSocket not connected in _send_text_to_client_task, stopping.")
                        self._stop_event.set()
                        break
                except asyncio.TimeoutError:
                    continue
        except Exception as e:
            if not self._stop_event.is_set(): print(f"Error in _send_text_to_client_task: {e}")
        finally:
            print("_send_text_to_client_task finished.")

    async def _process_client_audio_to_gemini_task(self):
        try:
            while not self._stop_event.is_set():
                try:
                    audio_chunk_bytes = await asyncio.wait_for(self.audio_to_gemini_queue.get(), timeout=0.1)
                    if audio_chunk_bytes is None: break # Sentinel
                    
                    if self.gemini_session and not self._stop_event.is_set():
                        try:
                            # Send the WebM/Opus chunk directly to Gemini
                            # print(f"Sending audio chunk to Gemini, size: {len(audio_chunk_bytes)}, type: audio/webm")
                            part = types.Part(inline_data=types.Blob(data=audio_chunk_bytes, mime_type="audio/webm")) # Changed mime_type
                            await self.gemini_session.send_realtime_input(media=part.inline_data)
                        
                        except Exception as e_send:
                            print(f"Error sending audio chunk to Gemini: {e_send}")
                            traceback.print_exc()
                            continue

                    elif self._stop_event.is_set():
                        break
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    if not self._stop_event.is_set():
                        print(f"Error in _process_client_audio_to_gemini_task main loop: {e}")
                        traceback.print_exc()
                    await asyncio.sleep(0.01) 
        finally:
            print("_process_client_audio_to_gemini_task finished.")

    async def _process_client_text_to_gemini_task(self):
        try:
            while not self._stop_event.is_set():
                try:
                    text_input = await asyncio.wait_for(self.text_to_gemini_queue.get(), timeout=0.1)
                    if text_input is None: break # Sentinel
                    
                    text_to_send = text_input if text_input else "." 
                    user_content = types.Content(role="user", parts=[types.Part(text=text_to_send)])
                    
                    if self.gemini_session and not self._stop_event.is_set():
                        await self.gemini_session.send_client_content(turns=user_content, turn_complete=True)
                        print(f"Sent to Gemini (from web client): '{text_to_send}'")
                    elif self._stop_event.is_set():
                        break
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    if not self._stop_event.is_set():
                        print(f"Error sending client text content to Gemini: {e}")
        finally:
            print("_process_client_text_to_gemini_task finished.")

    async def _process_gemini_responses_task(self):
        try:
            while not self._stop_event.is_set() and self.gemini_session:
                try:
                    response_message_iterator = self.gemini_session.receive()
                    response_message = await asyncio.wait_for(response_message_iterator.__anext__(), timeout=0.5) 
                    
                    # DETAILED LOGGING OF GEMINI RESPONSE
                    print(f"--- RAW GEMINI RESPONSE MESSAGE ---")
                    print(f"Type: {type(response_message)}")
                    print(f"Content: {response_message}")
                    # --- END DETAILED LOGGING ---

                    if self._stop_event.is_set(): break

                    if server_tool_call := response_message.tool_call:
                        print("--- GEMINI TOOL CALL RECEIVED ---") # Added log
                        if not self._stop_event.is_set():
                            await handle_gemini_tool_call(self.gemini_session, server_tool_call)
                        continue

                    if server_content := response_message.server_content:
                        print("--- GEMINI SERVER CONTENT RECEIVED ---") # Added log
                        if model_turn_content := server_content.model_turn:
                            print(f"Model Turn Content: {model_turn_content}") # Added log
                            for part_idx, part in enumerate(model_turn_content.parts):
                                print(f"  Part {part_idx}: {part}") # Log each part
                                if self._stop_event.is_set(): break
                                if part.text:
                                    print(f"    Putting TEXT to client queue: '{part.text[:50]}...'") # Log text
                                    await self.text_from_gemini_queue.put(part.text)
                                elif part.inline_data and part.inline_data.data:
                                    print(f"    Putting AUDIO to client queue, size: {len(part.inline_data.data)}") # Log audio
                                    await self.audio_from_gemini_queue.put(part.inline_data.data)
                                elif exec_result := part.code_execution_result: 
                                    outcome_str = exec_result.outcome if exec_result.outcome else "UNSPECIFIED"
                                    output_str = exec_result.output if exec_result.output else ""
                                    msg = f"\n[Code Execution Result From Server]: Outcome: {outcome_str}, Output: \"{output_str}\""
                                    await self.text_from_gemini_queue.put(msg)
                        
                        if server_content.interrupted:
                            msg = "\n[Playback Interrupted by Server Signal]"
                            await self.text_from_gemini_queue.put(msg)
                            while not self.audio_from_gemini_queue.empty():
                                try: self.audio_from_gemini_queue.get_nowait()
                                except asyncio.QueueEmpty: break
                
                except asyncio.TimeoutError:
                    continue 
                except StopAsyncIteration: 
                    if not self._stop_event.is_set(): 
                        print("Gemini session.receive() iterator finished unexpectedly.")
                        self._stop_event.set() 
                    break
                except Exception as e:
                    if not self._stop_event.is_set():
                        print(f"Error in _process_gemini_responses_task: {e}")
                        traceback.print_exc()
                        self._stop_event.set() 
                    break 
        finally:
            print("_process_gemini_responses_task finished.")

    async def run_main_interaction_loop(self):
        """
        Connects to Gemini and runs all processing tasks.
        """
        self._stop_event.clear()
        self._tasks = [] 

        # Check the initially loaded API key string, not an attribute of the client
        if not gemini_api_key:
            print("Cannot start Gemini interaction: API key string was not loaded from environment.")
            await self.text_from_gemini_queue.put("Server Error: Gemini API key not configured.")
            await self.audio_from_gemini_queue.put(None) 
            await self.text_from_gemini_queue.put(None)
            return

        print("Attempting to connect to Gemini Live API...")
        try:
            async with genai_client.aio.live.connect(model=MODEL, config=LIVE_CONNECT_CONFIG) as session:
                self.gemini_session = session
                print("Gemini session connected for WebSocket client.")
                await self.text_from_gemini_queue.put("System: Connected to Assistant.") 

                async with asyncio.TaskGroup() as tg:
                    self._tasks.append(tg.create_task(self._send_audio_to_client_task(), name="send_audio_to_ws"))
                    self._tasks.append(tg.create_task(self._send_text_to_client_task(), name="send_text_to_ws"))
                    self._tasks.append(tg.create_task(self._process_client_audio_to_gemini_task(), name="client_audio_to_gemini"))
                    self._tasks.append(tg.create_task(self._process_client_text_to_gemini_task(), name="client_text_to_gemini"))
                    self._tasks.append(tg.create_task(self._process_gemini_responses_task(), name="gemini_responses_processor"))
                    print("WebAudioHandler: All core tasks created in TaskGroup.")

                print("WebAudioHandler: TaskGroup finished (e.g. _stop_event was set).")

        except Exception as e:
            print(f"Error connecting to or during Gemini session: {e}")
            traceback.print_exc()
            await self.text_from_gemini_queue.put(f"Server Error: Could not connect to Assistant. {type(e).__name__}")
            self._stop_event.set() 
        finally:
            print("WebAudioHandler: run_main_interaction_loop ending.")
            self.gemini_session = None 
            self._stop_event.set() 
            
            await self.audio_from_gemini_queue.put(None)
            await self.text_from_gemini_queue.put(None)
            await self.audio_to_gemini_queue.put(None)
            await self.text_to_gemini_queue.put(None)
            
            print("WebAudioHandler: run_main_interaction_loop finished cleanup.")


    async def handle_client_data(self, data):
        """
        Process data received from the WebSocket client and put it into appropriate queues.
        """
        if self._stop_event.is_set():
            return

        if isinstance(data, bytes):
            await self.audio_to_gemini_queue.put(data)
        elif isinstance(data, str):
            print(f"Received text from client: {data}")
            await self.text_to_gemini_queue.put(data)

    async def stop_handler_tasks(self):
        print("WebAudioHandler: Initiating stop_handler_tasks...")
        if not self._stop_event.is_set():
            self._stop_event.set()
        await asyncio.sleep(0.5) 
        print("WebAudioHandler: stop_handler_tasks completed signal.")


@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected.")
    
    handler = WebAudioHandler(websocket)
    gemini_interaction_task = asyncio.create_task(handler.run_main_interaction_loop(), name="gemini_interaction")

    try:
        while True: 
            if handler._stop_event.is_set() or gemini_interaction_task.done():
                print("Handler interaction stopped or task ended, breaking client receive loop.")
                break

            try:
                data = await asyncio.wait_for(websocket.receive(), timeout=0.5) 
                
                if data.get("type") == "websocket.disconnect":
                    print("WebSocket client initiated disconnect via message.")
                    break 
                
                if "text" in data:
                    await handler.handle_client_data(data["text"])
                elif "bytes" in data:
                    await handler.handle_client_data(data["bytes"])

            except asyncio.TimeoutError:
                continue 
            except WebSocketDisconnect:
                print("WebSocket client disconnected (WebSocketDisconnect exception).")
                break 
            except Exception as e: 
                print(f"Error in WebSocket receive loop: {e}")
                traceback.print_exc()
                break 

    except Exception as e: 
        print(f"Outer error in WebSocket endpoint: {e}")
        traceback.print_exc()
    finally:
        print("WebSocket endpoint: Cleaning up connection...")
        
        await handler.stop_handler_tasks()
        
        if gemini_interaction_task and not gemini_interaction_task.done():
            print("WebSocket endpoint: Waiting for Gemini interaction task to complete...")
            try:
                await asyncio.wait_for(gemini_interaction_task, timeout=5.0) 
                print("WebSocket endpoint: Gemini interaction task completed.")
            except asyncio.TimeoutError:
                print("WebSocket endpoint: Gemini interaction task timed out during cleanup. Cancelling.")
                gemini_interaction_task.cancel()
                try:
                    await gemini_interaction_task
                except asyncio.CancelledError:
                    print("WebSocket endpoint: Gemini interaction task cancelled successfully.")
            except Exception as e_task:
                 print(f"WebSocket endpoint: Error waiting for/cancelling Gemini task: {e_task}")
        elif gemini_interaction_task:
             print("WebSocket endpoint: Gemini interaction task was already done.")

        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
                print("WebSocket connection closed gracefully.")
            except RuntimeError as e_close:
                print(f"RuntimeError during WebSocket close: {e_close}. Connection might have been already closed.")
        
        print("WebSocket endpoint cleanup complete.")


@app.get("/", response_class=FileResponse)
async def get_index_html(): 
    return FileResponse('index.html')

if __name__ == "__main__":
    if not gemini_api_key:
        print("CRITICAL: GEMINI_API_KEY is not set. The application might not function correctly.")
        print("Please ensure a .env file with GEMINI_API_KEY is present in the root directory or the variable is set in your environment.")
    
    print("Starting Uvicorn server for Hue Voice API...")
    print("Navigate to http://<your-ip-address>:8000 to access the voice assistant.")
    print("Connect WebSocket clients (internally) to ws://<your-ip-address>:8000/ws/voice")
    uvicorn.run(app, host="0.0.0.0", port=8000) 
"""
## Documentation
Quickstart: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py

## Setup

To install the dependencies for this script, run:

```
pip install google-genai opencv-python pyaudio pillow mss
```
"""

import os
import asyncio
import base64
import io
import traceback
import time

# import pyaudio # PyAudio removed as playback moves to client
# from google.generativeai import types # Corrected import path if needed, standard is google.genai
from google.genai import types # Standard import

import argparse

from google import genai
from dotenv import load_dotenv

# Import the new tools
from tools import turn_office_light_on, turn_office_light_off, control_light

load_dotenv()

# FORMAT = pyaudio.paInt16 # Removed, PyAudio not used here
CHANNELS = 1 # For mic config, and can inform client about Gemini output format
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000 # Gemini's output audio sample rate

MODEL = "models/gemini-2.0-flash-live-001" # Ensured correct model

client = genai.Client( # Ensured original client init
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# Define the new tools for the voice agent
tools_list = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="control_light",
                description="Controls a specific light by its name. Can turn it ON, OFF, or TOGGLE its state. Can optionally set brightness if turning ON or toggling to ON.",
                parameters=types.Schema(
                    type=types.Type.OBJECT, 
                    properties={
                        "light_name": types.Schema(type=types.Type.STRING, description="The user-friendly name of the light (e.g., 'Office Lamp', 'Living Boob 1')." ),
                        "action": types.Schema(type=types.Type.STRING, description="The desired action: 'ON', 'OFF', or 'TOGGLE'."),
                        "brightness_percent": types.Schema(type=types.Type.NUMBER, description="Optional: The desired brightness level (0-100). Applied if action results in the light being ON.", nullable=True),
                    },
                    required=["light_name", "action"]
                )
            ),
            # Retain old tools for now, can be removed if control_light covers all uses or if they are adapted
            # types.FunctionDeclaration(
            #     name="turn_office_light_on",
            #     description="Turns the office lamp on.",
            #     parameters=types.Schema(type=types.Type.OBJECT, properties={}),
            # ),
            # types.FunctionDeclaration(
            #     name="turn_office_light_off",
            #     description="Turns the office lamp off.",
            #     parameters=types.Schema(type=types.Type.OBJECT, properties={}),
            # ),
        ]
    )
]

CONFIG = types.LiveConnectConfig( # Ensured original config structure
    response_modalities=[
        "AUDIO",
    ],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
        )
    ),
    tools=tools_list, # Use renamed list
)

class AudioLoop:
    def __init__(self, user_text_input_queue=None, model_text_output_queue=None, webrtc_audio_input_queue=None, model_audio_output_queue=None): # Added model_audio_output_queue
        self.user_text_input_queue = user_text_input_queue
        self.model_text_output_queue = model_text_output_queue
        self.webrtc_audio_input_queue = webrtc_audio_input_queue
        self.model_audio_output_queue = model_audio_output_queue # Store new queue
        
        self.pya = None # PyAudio instance removed from direct init
        # self.audio_in_queue = None  # Replaced by model_audio_output_queue
        self.out_queue = None       # For audio from mic (WebRTC) to be sent to Gemini

        self.session = None
        self._shutdown_event = asyncio.Event()
        self._tasks = []
    
    # _initialize_pyaudio_if_needed method removed

    async def _shutdown_monitor(self):
        await self._shutdown_event.wait()  
        # print("AudioLoop: _shutdown_monitor: Shutdown signaled.") # Less verbose

        if self.audio_stream:  
            # print("AudioLoop: _shutdown_monitor: Attempting to stop microphone stream (self.audio_stream).") # Less verbose
            try:
                is_active = await asyncio.wait_for(asyncio.to_thread(self.audio_stream.is_active), timeout=1.0)
                if is_active:
                    # print("AudioLoop: _shutdown_monitor: Microphone stream is active, calling stop_stream().") # Less verbose
                    await asyncio.wait_for(asyncio.to_thread(self.audio_stream.stop_stream), timeout=1.0)
                    # print("AudioLoop: _shutdown_monitor: Microphone stream stop_stream() called.") # Less verbose
                # else:
                    # print("AudioLoop: _shutdown_monitor: Microphone stream was already inactive.") # Less verbose
            except asyncio.TimeoutError:
                print("AudioLoop: _shutdown_monitor: Timeout during microphone stream stop operation.")
            except Exception as e:
                print(f"AudioLoop: _shutdown_monitor: Error stopping microphone stream: {e}")
        else:
            print("AudioLoop: _shutdown_monitor: No microphone stream (self.audio_stream was None) to stop.")
            # No specific microphone stream to stop here if input is from WebRTC queue
            pass 

    async def send_text_from_queue(self):
        print("AudioLoop: send_text_from_queue started.")
        try:
            while not self._shutdown_event.is_set():
                if not self.user_text_input_queue:
                    await asyncio.sleep(0.1) 
                    continue
                
                try:
                    text = await asyncio.wait_for(self.user_text_input_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue 

                # print(f"AudioLoop: Got text from input queue: '{text}'") # Keep this for user input confirmation
                if text is None:  
                    print("AudioLoop: send_text_from_queue received None, signaling shutdown.")
                    self._shutdown_event.set()
                    break 
                
                text_to_send = text if text else "."
                user_content = types.Content(role="user", parts=[types.Part(text=text_to_send)])
                
                if self.session and not self._shutdown_event.is_set():
                    try:
                        await self.session.send_client_content(turns=user_content, turn_complete=True)
                        print(f"AudioLoop: Sent to Gemini: '{text_to_send}'") # Keep this for user input confirmation
                    except Exception as e:
                        if not self._shutdown_event.is_set(): 
                            print(f"AudioLoop: Error sending client content: {e}")
                elif self._shutdown_event.is_set():
                    # print("AudioLoop: Shutdown signaled in send_text, not sending.") # Less verbose
                    break
                else:
                    if not self._shutdown_event.is_set(): print("AudioLoop: Session not active, cannot send text.")
        finally:
            print("AudioLoop: send_text_from_queue finished.")

    async def send_realtime(self):
        # print("AudioLoop: send_realtime started.") # Less verbose
        try:
            while not self._shutdown_event.is_set():
                if not self.out_queue: # out_queue will be populated by listen_audio from webrtc_audio_input_queue
                    await asyncio.sleep(0.1)
                    continue
                try:
                    msg = await asyncio.wait_for(self.out_queue.get(), timeout=0.5)
                    if msg is None: 
                        # print("AudioLoop: send_realtime received None from out_queue, exiting.") # Less verbose
                        break
                    
                    if msg.inline_data and msg.inline_data.data:
                        # print(f"AudioLoop: send_realtime got part with {len(msg.inline_data.data)} bytes from out_queue.") # Too verbose
                        pass
                    else:
                        # print("AudioLoop: send_realtime got part from out_queue, but inline_data or data is missing/empty.") # Less verbose
                        continue 

                    if self.session and not self._shutdown_event.is_set():
                        try:
                            await self.session.send_realtime_input(media=msg.inline_data)
                        except Exception as e:
                            if not self._shutdown_event.is_set():
                                print(f"AudioLoop: Error in send_realtime while sending to session: {e}")
                    elif self._shutdown_event.is_set():
                        # print("AudioLoop: Shutdown signaled in send_realtime, not sending.") # Less verbose
                        break
                except asyncio.TimeoutError:
                    continue 
                except Exception as e:
                    if not self._shutdown_event.is_set(): 
                        print(f"AudioLoop: Error in send_realtime: {e}")
                    await asyncio.sleep(0.1)
        finally:
            # print("AudioLoop: send_realtime finished.") # Less verbose
            pass

    async def listen_audio(self):
        print("AudioLoop: listen_audio started (New Pause Detection Logic).")

        if not self.webrtc_audio_input_queue or not self.out_queue or not self.user_text_input_queue:
            print("AudioLoop: listen_audio: Essential queue(s) not provided. Cannot listen.")
            self._shutdown_event.set()
            return

        internal_audio_buffer = bytearray()
        last_audio_received_time = time.monotonic()
        
        MAX_UTTERANCE_DURATION_SECONDS = 2.5
        SILENCE_THRESHOLD_SECONDS = 0.4
        QUEUE_POLL_INTERVAL_SECONDS = 0.02 
        # MIN_BUFFER_FOR_SILENCE_SEND_BYTES = int(0.3 * 16000 * 2) # Removing this concept for now
        
        MAX_BUFFER_SIZE_BYTES = int(MAX_UTTERANCE_DURATION_SECONDS * 16000 * 2)

        try:
            while not self._shutdown_event.is_set():
                send_audio_now = False
                chunk_received_in_this_iteration = False

                try:
                    audio_chunk_bytes = await asyncio.wait_for(
                        self.webrtc_audio_input_queue.get(), 
                        timeout=QUEUE_POLL_INTERVAL_SECONDS 
                    )
                    
                    if audio_chunk_bytes is None: 
                        print("AudioLoop: listen_audio received None (shutdown) from webrtc_audio_input_queue.")
                        self._shutdown_event.set()
                        if internal_audio_buffer: 
                            send_audio_now = True
                        else:
                            break 

                    if not self._shutdown_event.is_set() and audio_chunk_bytes:
                        internal_audio_buffer.extend(audio_chunk_bytes)
                        last_audio_received_time = time.monotonic()
                        chunk_received_in_this_iteration = True

                        if len(internal_audio_buffer) >= MAX_BUFFER_SIZE_BYTES:
                            print(f"AudioLoop: listen_audio: Max buffer size ({MAX_BUFFER_SIZE_BYTES} bytes) reached.")
                            send_audio_now = True
                
                except asyncio.TimeoutError:
                    pass 
                
                except Exception as e:
                    if not self._shutdown_event.is_set():
                        print(f"AudioLoop: Unexpected error getting from webrtc_audio_input_queue: {e}")
                        self._shutdown_event.set()
                    break 

                if self._shutdown_event.is_set() and not internal_audio_buffer:
                     break

                if not chunk_received_in_this_iteration and internal_audio_buffer:
                    current_time = time.monotonic()
                    if (current_time - last_audio_received_time) > SILENCE_THRESHOLD_SECONDS:
                        # if len(internal_audio_buffer) >= MIN_BUFFER_FOR_SILENCE_SEND_BYTES: # Condition removed
                        # print(f"AudioLoop: listen_audio: Silence detected ({SILENCE_THRESHOLD_SECONDS}s) with SUFFICIENT audio ({len(internal_audio_buffer)} bytes vs min ...). Triggering send.")
                        print(f"AudioLoop: listen_audio: Silence detected ({SILENCE_THRESHOLD_SECONDS}s). Buffer has {len(internal_audio_buffer)} bytes. Triggering send.")
                        send_audio_now = True
                        # else: 
                            # print(f"AudioLoop: listen_audio: Silence detected ({SILENCE_THRESHOLD_SECONDS}s) but audio buffer ({len(internal_audio_buffer)} bytes) is BELOW MINIMUM (...). Not sending yet.")
                
                if send_audio_now and internal_audio_buffer:
                    # print(f"AudioLoop: listen_audio: Preparing to send {len(internal_audio_buffer)} accumulated bytes.") # Too verbose
                    part_to_send = types.Part(inline_data=types.Blob(data=bytes(internal_audio_buffer), mime_type="audio/pcm"))
                    
                    try:
                        await self.out_queue.put(part_to_send)
                        # print(f"AudioLoop: listen_audio: Sent {len(internal_audio_buffer)} bytes to out_queue.") # Too verbose
                    except asyncio.QueueFull:
                         print(f"AudioLoop: listen_audio: out_queue is full. Discarding {len(internal_audio_buffer)} bytes.")
                    
                    internal_audio_buffer.clear()
                    
                    last_audio_received_time = time.monotonic() 
                
                if self._shutdown_event.is_set() and not internal_audio_buffer:
                    break

        finally:
            print("AudioLoop: listen_audio (New Pause Detection Logic) finished.")

    async def receive_audio(self):
        # print("AudioLoop: receive_audio started.") # Less verbose
        try:
            while not self._shutdown_event.is_set():
                if not self.session:
                    await asyncio.sleep(0.1) 
                    continue
                try:
                    response_message_iterator = self.session.receive()
                    response_message = await asyncio.wait_for(response_message_iterator.__anext__(), timeout=0.5)
                    # print(f"AudioLoop: receive_audio got response_message: {type(response_message)}") # Less verbose
                    
                    if server_tool_call := response_message.tool_call:
                        # print(f"AudioLoop: receive_audio processing tool_call: {server_tool_call}") # Keep, important event
                        await handle_tool_call(self.session, server_tool_call)
                        continue

                    if server_content := response_message.server_content:
                        # print(f"AudioLoop: receive_audio processing server_content. Turn complete: {server_content.turn_complete}, Interrupted: {server_content.interrupted}") # Less verbose
                        if model_turn_content := server_content.model_turn:
                            # print(f"AudioLoop: receive_audio got model_turn_content with {len(model_turn_content.parts)} parts.") # Less verbose
                            for i, part in enumerate(model_turn_content.parts):
                                if self._shutdown_event.is_set(): break
                                if part.text:
                                    # print(f"AudioLoop: receive_audio part {i} is text: '{part.text[:100]}...'") # Less verbose
                                    if self.model_text_output_queue:
                                        self.model_text_output_queue.put_nowait(part.text)
                                    else: 
                                        if not self._shutdown_event.is_set(): print(part.text, end="", flush=True)
                                elif part.inline_data and part.inline_data.data:
                                    print(f"AudioLoop: receive_audio: Got audio data from Gemini ({len(part.inline_data.data)} bytes), queuing for browser.") # Key log added
                                    if self.model_audio_output_queue: # MODIFIED: Use new queue
                                        self.model_audio_output_queue.put_nowait(part.inline_data.data)
                                    else:
                                        print("AudioLoop: model_audio_output_queue not available to send audio data.")
                                elif exec_result := part.code_execution_result:
                                    outcome_str = exec_result.outcome if exec_result.outcome else "UNSPECIFIED"
                                    output_str = exec_result.output if exec_result.output else ""
                                    msg = f"\n[Code Execution Result From Server]: Outcome: {outcome_str}, Output: \"{output_str}\""
                                    # print(f"AudioLoop: receive_audio part {i} is code_execution_result: {msg}") # Keep, important event
                                    if self.model_text_output_queue:
                                        self.model_text_output_queue.put_nowait(msg)
                                    else:
                                        if not self._shutdown_event.is_set(): print(msg, flush=True)
                                else:
                                    # print(f"AudioLoop: receive_audio part {i} is of an unknown type or empty. Part details: {part}") # Less verbose
                                    pass 
                            if self.model_text_output_queue and server_content.turn_complete:
                                # print("AudioLoop: receive_audio model turn complete.") # Less verbose
                                pass 
                        
                        if server_content.interrupted:
                            msg = "\n[Playback Interrupted by Server Signal]"
                            print("AudioLoop: receive_audio server_content.interrupted is true.") # Keep, important event
                            if self.model_text_output_queue:
                                self.model_text_output_queue.put_nowait(msg)
                            else:
                                if not self._shutdown_event.is_set(): print(msg, flush=True)
                            if self.model_audio_output_queue: # MODIFIED: Clear new queue
                                while not self.model_audio_output_queue.empty():
                                    try: self.model_audio_output_queue.get_nowait()
                                    except asyncio.QueueEmpty: break
                except asyncio.TimeoutError:
                    continue 
                except StopAsyncIteration: 
                    if not self._shutdown_event.is_set(): 
                        print("AudioLoop: session.receive() iterator finished unexpectedly.")
                        self._shutdown_event.set() 
                    break
                except Exception as e:
                    if not self._shutdown_event.is_set():
                        print(f"AudioLoop: Error in receive_audio: {e}")
                        self._shutdown_event.set() 
                    break 
        finally:
            # print("AudioLoop: receive_audio finished.") # Less verbose
            pass

    async def play_audio(self):
        # print("AudioLoop: play_audio started (now a no-op for server-side PyAudio).")
        # This method no longer plays audio directly using PyAudio.
        # Audio is routed to model_audio_output_queue for client-side handling.
        try:
            while not self._shutdown_event.is_set():
                # Loop just sleeps to keep the task alive if it's still in the task group.
                # Or this task could be removed entirely if not needed for other coordination.
                await asyncio.sleep(0.5) 
                if self._shutdown_event.is_set():
                    break
        except asyncio.CancelledError:
            # print("AudioLoop: play_audio task cancelled.")
            pass
        except Exception as e:
            if not self._shutdown_event.is_set():
                print(f"AudioLoop: Error in play_audio (no-op) loop: {e}")
        finally:
            # print("AudioLoop: play_audio (no-op) finished.")
            pass

    async def run(self):
        print("AudioLoop: run method started.")
        self._shutdown_event.clear() 
        self._tasks = []
        self.audio_stream = None 

        try:
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                print("AudioLoop: Gemini session connected.")

                if self.user_text_input_queue is None: self.user_text_input_queue = asyncio.Queue()
                if self.model_text_output_queue is None: self.model_text_output_queue = asyncio.Queue()
                if self.webrtc_audio_input_queue is None:
                     print("AudioLoop: CRITICAL - webrtc_audio_input_queue not set up during run!")
                     # This should not happen if initialized correctly from webapp_streamlit.py
                     self._shutdown_event.set() # Force shutdown if this essential queue is missing
                     raise ValueError("webrtc_audio_input_queue not provided to AudioLoop")

                # self.audio_in_queue = asyncio.Queue() # MODIFIED: Removed, replaced by model_audio_output_queue
                self.out_queue = asyncio.Queue(maxsize=50)

                async with asyncio.TaskGroup() as tg:
                    print("AudioLoop: Creating tasks within TaskGroup...")
                    self._tasks.append(tg.create_task(self._shutdown_monitor(), name="shutdown_monitor"))
                    self._tasks.append(tg.create_task(self.send_text_from_queue(), name="send_text"))
                    self._tasks.append(tg.create_task(self.send_realtime(), name="send_realtime"))
                    self._tasks.append(tg.create_task(self.listen_audio(), name="listen_audio"))
                    self._tasks.append(tg.create_task(self.receive_audio(), name="receive_audio"))
                    self._tasks.append(tg.create_task(self.play_audio(), name="play_audio"))
                    print("AudioLoop: All tasks created.")
                
                print("AudioLoop: TaskGroup finished naturally (all tasks completed).")

        except asyncio.CancelledError:
            print("AudioLoop: Run method cancelled externally.")
            self._shutdown_event.set()
        except ExceptionGroup as eg: 
            print(f"AudioLoop: ExceptionGroup caught in run (likely TaskGroup failure): {eg}")
            self._shutdown_event.set() 
        except Exception as e:
            print(f"AudioLoop: General exception in run: {e}")
            traceback.print_exc()
            self._shutdown_event.set() 
        finally:
            print("AudioLoop: run method entering finally block, ensuring shutdown signal is set...")
            self._shutdown_event.set() 

            # print("AudioLoop: Placing sentinels on all queues to unblock any waiting tasks...") # Less verbose
            if self.user_text_input_queue: self.user_text_input_queue.put_nowait(None)
            # if self.audio_in_queue: self.audio_in_queue.put_nowait(None) # MODIFIED: Removed
            if self.model_audio_output_queue: self.model_audio_output_queue.put_nowait(None) # MODIFIED: Add sentinel for new queue
            if self.out_queue: self.out_queue.put_nowait(None)
            if self.model_text_output_queue: self.model_text_output_queue.put_nowait(None) 
            
            # print("AudioLoop: Waiting briefly (e.g., 1s) for tasks to finalize based on event and sentinels...") # Less verbose
            await asyncio.sleep(1.0) 

            # print("AudioLoop: Checking task states (for debugging, TaskGroup should handle joins/cancellations)...") # Less verbose
            for task in self._tasks:
                if not task.done():
                    # print(f"AudioLoop: Task {task.get_name()} is not done. Attempting cancellation.") # Less verbose
                    task.cancel() 
            if any(not task.done() for task in self._tasks):
                await asyncio.sleep(0.5) 
                for task in self._tasks:
                    if not task.done():
                        print(f"AudioLoop: WARNING - Task {task.get_name()} still not done after cancellation attempt.") # Keep, important warning
            
            if self.session:
                # print("AudioLoop: Gemini session should be closed by 'async with' context manager.") # Less verbose
                self.session = None 
            
            print("AudioLoop: Terminating self.pya (PyAudio instance)...") # Will be None now
            try:
                if self.pya: # pya will be None, so this block won't run
                    await asyncio.to_thread(self.pya.terminate) 
                    print("AudioLoop: self.pya (PyAudio instance) terminated successfully.")
            except Exception as e:
                print(f"AudioLoop: Error terminating self.pya (PyAudio instance): {e}")
            self.pya = None 
            print("AudioLoop: run method finished cleanup.")


async def handle_tool_call(session, server_tool_call: types.LiveServerToolCall):
    print(f"Server tool call received: {server_tool_call}") # Keep, important event
    
    function_responses = []
    
    if server_tool_call.function_calls:
        for fc in server_tool_call.function_calls:
            tool_id = fc.id
            tool_name = fc.name
            
            print(f"Executing tool: {tool_name} (ID: {tool_id})") # Keep, important event
            
            tool_result_dict = None 
            try:
                if tool_name == "turn_office_light_on":
                    # tool_result_dict = turn_office_light_on() # Old call
                    tool_result_dict = control_light(light_name="Office Lamp", action="ON")
                elif tool_name == "turn_office_light_off":
                    # tool_result_dict = turn_office_light_off() # Old call
                    tool_result_dict = control_light(light_name="Office Lamp", action="OFF")
                elif tool_name == "control_light":
                    light_name_arg = fc.args.get("light_name")
                    action_arg = fc.args.get("action")
                    brightness_arg = fc.args.get("brightness_percent") # This might be None

                    if not light_name_arg or not action_arg:
                        tool_result_dict = {"status": "error", "message": "Missing required arguments 'light_name' or 'action' for control_light."}
                        print(tool_result_dict["message"]) 
                    else:
                        print(f"Calling control_light with: name='{light_name_arg}', action='{action_arg}', brightness={brightness_arg}")
                        # Ensure brightness_arg is int if present, or None
                        if brightness_arg is not None:
                            try:
                                brightness_arg = int(brightness_arg)
                            except ValueError:
                                tool_result_dict = {"status": "error", "message": f"Invalid brightness_percent value: {brightness_arg}. Must be a number."}
                                print(tool_result_dict["message"]) 
                        
                        if not tool_result_dict: # Only call if brightness conversion was okay or not needed
                            tool_result_dict = control_light(
                                light_name=light_name_arg, 
                                action=action_arg, 
                                brightness_percent=brightness_arg
                            )
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
        print(f"Sending tool responses to Gemini: {function_responses}") # Keep, important event
        await session.send_tool_response(function_responses=function_responses)
    else:
        print("No function calls were processed or found in the server_tool_call message.") # Keep, useful info


if __name__ == "__main__":
    print("Starting AudioLoop directly for testing...")
    user_q = asyncio.Queue()
    model_q = asyncio.Queue()
    webrtc_q = asyncio.Queue() # Dummy queue for direct testing, WebRTCAudioRelay would normally populate this
    loop_task = None 
    
    main_loop = AudioLoop(user_text_input_queue=user_q, model_text_output_queue=model_q, webrtc_audio_input_queue=webrtc_q)
    
    async def console_input_sender(queue):
        while True:
            try:
                text = await asyncio.to_thread(input, "message (type 'q' to quit) > ")
                if text.lower() == 'q':
                    await queue.put(None) 
                    break
                await queue.put(text)
            except EOFError: 
                await queue.put(None)
                break
            except Exception as e:
                print(f"Console input error: {e}")
                await queue.put(None)
                break

    async def model_output_printer(queue):
        while True:
            item = await queue.get()
            if item is None: 
                print("MODEL_OUTPUT_PRINTER: Received None, exiting.")
                break
            print(f"MODEL: {item}", flush=True)
            queue.task_done() 

    async def run_test():
        global loop_task 
        loop_task = asyncio.create_task(main_loop.run())
        input_task = asyncio.create_task(console_input_sender(user_q))
        output_task = asyncio.create_task(model_output_printer(model_q))
        
        await input_task 
        print("Console input finished. Main loop should be shutting down.")
        
        try:
            await asyncio.wait_for(loop_task, timeout=15.0) # Increased timeout to allow for processing simulated audio
            print("Main loop task completed.")
        except asyncio.TimeoutError:
            print("Main loop task timed out during shutdown. Attempting cancellation.")
            if loop_task and not loop_task.done(): loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                print("Main loop task was cancelled.")
        except Exception as e:
            print(f"Error waiting for main loop task: {e}")
            
        if not model_q.empty(): 
             model_q.put_nowait(None) 
        try:
            await asyncio.wait_for(output_task, timeout=2.0)
        except asyncio.TimeoutError:
            print("Model output printer timed out during shutdown.")
        except Exception as e:
            print(f"Error waiting for model output printer: {e}")

        print("Direct test finished.")

    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        print("KeyboardInterrupt, attempting graceful shutdown...")
        if loop_task and not loop_task.done():
            user_q.put_nowait(None) 
    finally:
        print("Test script finished.")

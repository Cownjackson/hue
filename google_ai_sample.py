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

import pyaudio

import argparse

from google import genai
from google.genai import types
from dotenv import load_dotenv

# Import the new tools
from tools import turn_office_light_on, turn_office_light_off

load_dotenv()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001"

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# Define the new tools for the voice agent
tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="turn_office_light_on",
                description="Turns the office lamp on.",
                parameters=genai.types.Schema(type=genai.types.Type.OBJECT, properties={}),
            ),
        ]
    ),
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="turn_office_light_off",
                description="Turns the office lamp off.",
                parameters=genai.types.Schema(type=genai.types.Type.OBJECT, properties={}),
            ),
        ]
    )
]

CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
        )
    ),
    tools=tools,
)

class AudioLoop:
    def __init__(self, user_text_input_queue=None, model_text_output_queue=None):
        self.user_text_input_queue = user_text_input_queue
        self.model_text_output_queue = model_text_output_queue
        
        self.pya = pyaudio.PyAudio() # Instance-specific PyAudio
        self.audio_in_queue = None 
        self.out_queue = None      

        self.session = None
        self.audio_stream = None 
        self._shutdown_event = asyncio.Event()
        self._tasks = [] 

    async def _shutdown_monitor(self):
        await self._shutdown_event.wait()  
        print("AudioLoop: _shutdown_monitor: Shutdown signaled.")

        if self.audio_stream:  
            print("AudioLoop: _shutdown_monitor: Attempting to stop microphone stream (self.audio_stream).")
            try:
                is_active = await asyncio.wait_for(asyncio.to_thread(self.audio_stream.is_active), timeout=1.0)
                if is_active:
                    print("AudioLoop: _shutdown_monitor: Microphone stream is active, calling stop_stream().")
                    await asyncio.wait_for(asyncio.to_thread(self.audio_stream.stop_stream), timeout=1.0)
                    print("AudioLoop: _shutdown_monitor: Microphone stream stop_stream() called.")
                else:
                    print("AudioLoop: _shutdown_monitor: Microphone stream was already inactive.")
            except asyncio.TimeoutError:
                print("AudioLoop: _shutdown_monitor: Timeout during microphone stream stop operation.")
            except Exception as e:
                print(f"AudioLoop: _shutdown_monitor: Error stopping microphone stream: {e}")
        else:
            print("AudioLoop: _shutdown_monitor: No microphone stream (self.audio_stream was None) to stop.")

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

                print(f"AudioLoop: Got text from input queue: '{text}'")
                if text is None:  
                    print("AudioLoop: send_text_from_queue received None, signaling shutdown.")
                    self._shutdown_event.set()
                    break 
                
                text_to_send = text if text else "."
                user_content = types.Content(role="user", parts=[types.Part(text=text_to_send)])
                
                if self.session and not self._shutdown_event.is_set():
                    try:
                        await self.session.send_client_content(turns=user_content, turn_complete=True)
                        print(f"AudioLoop: Sent to Gemini: '{text_to_send}'")
                    except Exception as e:
                        if not self._shutdown_event.is_set(): 
                            print(f"AudioLoop: Error sending client content: {e}")
                elif self._shutdown_event.is_set():
                    print("AudioLoop: Shutdown signaled in send_text, not sending.")
                    break
                else:
                    if not self._shutdown_event.is_set(): print("AudioLoop: Session not active, cannot send text.")
        finally:
            print("AudioLoop: send_text_from_queue finished.")

    async def send_realtime(self):
        print("AudioLoop: send_realtime started.")
        try:
            while not self._shutdown_event.is_set():
                if not self.out_queue:
                    await asyncio.sleep(0.1)
                    continue
                try:
                    msg = await asyncio.wait_for(self.out_queue.get(), timeout=0.5)
                    if msg is None: 
                        print("AudioLoop: send_realtime received None from out_queue, exiting.")
                        break
                    if self.session and not self._shutdown_event.is_set():
                        await self.session.send_realtime_input(media=msg.inline_data)
                    elif self._shutdown_event.is_set():
                        print("AudioLoop: Shutdown signaled in send_realtime, not sending.")
                        break
                except asyncio.TimeoutError:
                    continue 
                except Exception as e:
                    if not self._shutdown_event.is_set(): 
                        print(f"AudioLoop: Error in send_realtime: {e}")
                    await asyncio.sleep(0.1)
        finally:
            print("AudioLoop: send_realtime finished.")

    async def listen_audio(self):
        print("AudioLoop: listen_audio started.")
        try:
            mic_info = await asyncio.to_thread(self.pya.get_default_input_device_info)
            self.audio_stream = await asyncio.to_thread(
                self.pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE,
            )
            print("AudioLoop: Microphone stream (self.audio_stream) opened.")
        except Exception as e:
            if not self._shutdown_event.is_set():
                print(f"AudioLoop: Error opening microphone stream: {e}")
                self._shutdown_event.set() 
            self.audio_stream = None 
            return 

        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        
        try:
            while not self._shutdown_event.is_set():
                if not self.out_queue or not self.audio_stream:
                    await asyncio.sleep(0.1)
                    if not self.audio_stream and not self._shutdown_event.is_set():
                        print("AudioLoop: listen_audio: audio_stream is None, cannot read.")
                        self._shutdown_event.set() 
                    if self._shutdown_event.is_set(): break
                    continue
                try:
                    data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                    if self._shutdown_event.is_set(): 
                        print("AudioLoop: Shutdown detected after audio_stream.read, breaking listen_audio loop.")
                        break
                    await self.out_queue.put(types.Part(inline_data=types.Blob(data=data, mime_type="audio/pcm")))
                except IOError as e: 
                    if self._shutdown_event.is_set():
                        print(f"AudioLoop: Expected IOError in listen_audio during shutdown (stream likely closed by monitor): {e}")
                    elif e.errno == pyaudio.paInputOverflowed:
                        if not self._shutdown_event.is_set(): print("AudioLoop: Microphone input overflowed.")
                    else:
                        if not self._shutdown_event.is_set(): 
                            print(f"AudioLoop: IOError in listen_audio: {e}")
                            self._shutdown_event.set() 
                    break 
                except Exception as e:
                    if not self._shutdown_event.is_set():
                        print(f"AudioLoop: Unexpected error in listen_audio: {e}")
                        self._shutdown_event.set() 
                    break 
        finally:
            if self.audio_stream:
                print("AudioLoop: listen_audio finally: Closing microphone stream (self.audio_stream)...")
                try:
                    is_stopped = await asyncio.to_thread(self.audio_stream.is_stopped)
                    if not is_stopped:
                        await asyncio.to_thread(self.audio_stream.stop_stream)
                    await asyncio.to_thread(self.audio_stream.close)
                    print("AudioLoop: listen_audio finally: Microphone stream (self.audio_stream) closed.")
                except Exception as e:
                    print(f"AudioLoop: listen_audio finally: Error closing mic stream (self.audio_stream): {e}")
                self.audio_stream = None 
            else:
                print("AudioLoop: listen_audio finally: self.audio_stream was already None.")
            print("AudioLoop: listen_audio finished.")

    async def receive_audio(self):
        print("AudioLoop: receive_audio started.")
        try:
            while not self._shutdown_event.is_set():
                if not self.session:
                    await asyncio.sleep(0.1) 
                    continue
                try:
                    response_message_iterator = self.session.receive()
                    response_message = await asyncio.wait_for(response_message_iterator.__anext__(), timeout=0.5)
                    
                    if server_tool_call := response_message.tool_call:
                        if not self._shutdown_event.is_set():
                             await handle_tool_call(self.session, server_tool_call)
                        continue

                    if server_content := response_message.server_content:
                        if model_turn_content := server_content.model_turn:
                            for part in model_turn_content.parts:
                                if self._shutdown_event.is_set(): break
                                if part.text:
                                    if self.model_text_output_queue:
                                        self.model_text_output_queue.put_nowait(part.text)
                                    else: 
                                        if not self._shutdown_event.is_set(): print(part.text, end="", flush=True)
                                elif part.inline_data and part.inline_data.data:
                                    if self.audio_in_queue:
                                        self.audio_in_queue.put_nowait(part.inline_data.data)
                                elif exec_result := part.code_execution_result:
                                    outcome_str = exec_result.outcome if exec_result.outcome else "UNSPECIFIED"
                                    output_str = exec_result.output if exec_result.output else ""
                                    msg = f"\n[Code Execution Result From Server]: Outcome: {outcome_str}, Output: \"{output_str}\""
                                    if self.model_text_output_queue:
                                        self.model_text_output_queue.put_nowait(msg)
                                    else:
                                        if not self._shutdown_event.is_set(): print(msg, flush=True)
                            if self.model_text_output_queue and server_content.turn_complete:
                                pass 
                        
                        if server_content.interrupted:
                            msg = "\n[Playback Interrupted by Server Signal]"
                            if self.model_text_output_queue:
                                self.model_text_output_queue.put_nowait(msg)
                            else:
                                if not self._shutdown_event.is_set(): print(msg, flush=True)
                            if self.audio_in_queue:
                                while not self.audio_in_queue.empty():
                                    try: self.audio_in_queue.get_nowait()
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
            print("AudioLoop: receive_audio finished.")

    async def play_audio(self):
        print("AudioLoop: play_audio started.")
        stream = None 
        try:
            stream = await asyncio.to_thread(
                self.pya.open, # Use self.pya
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )
            print("AudioLoop: Audio output stream opened.")
        except Exception as e:
            if not self._shutdown_event.is_set():
                print(f"AudioLoop: Error opening audio output stream: {e}")
                self._shutdown_event.set() 
            return 

        try:
            while not self._shutdown_event.is_set():
                if not self.audio_in_queue:
                    await asyncio.sleep(0.1)
                    continue
                try:
                    bytestream = await asyncio.wait_for(self.audio_in_queue.get(), timeout=0.5)
                    if bytestream is None: 
                        print("AudioLoop: play_audio received None from audio_in_queue, exiting.")
                        break 
                    if not self._shutdown_event.is_set(): 
                        await asyncio.to_thread(stream.write, bytestream)
                    else:
                        print("AudioLoop: Shutdown signaled in play_audio, not writing.")
                        break
                except asyncio.TimeoutError:
                    continue 
                except Exception as e:
                    if not self._shutdown_event.is_set():
                        print(f"AudioLoop: Error in play_audio: {e}")
                    await asyncio.sleep(0.1) 
        finally:
            if stream:
                print("AudioLoop: play_audio finally: Closing audio output stream...")
                try:
                    await asyncio.to_thread(stream.stop_stream)
                    await asyncio.to_thread(stream.close)
                    print("AudioLoop: play_audio finally: Audio output stream closed.")
                except Exception as e:
                     print(f"AudioLoop: play_audio finally: Error closing output stream: {e}")
            print("AudioLoop: play_audio finished.")

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
                self.audio_in_queue = asyncio.Queue()
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

            print("AudioLoop: Placing sentinels on all queues to unblock any waiting tasks...")
            if self.user_text_input_queue: self.user_text_input_queue.put_nowait(None)
            if self.audio_in_queue: self.audio_in_queue.put_nowait(None)
            if self.out_queue: self.out_queue.put_nowait(None)
            if self.model_text_output_queue: self.model_text_output_queue.put_nowait(None) 
            
            print("AudioLoop: Waiting briefly (e.g., 1s) for tasks to finalize based on event and sentinels...")
            await asyncio.sleep(1.0) 

            print("AudioLoop: Checking task states (for debugging, TaskGroup should handle joins/cancellations)...")
            for task in self._tasks:
                if not task.done():
                    print(f"AudioLoop: Task {task.get_name()} is not done. Attempting cancellation.")
                    task.cancel() 
            if any(not task.done() for task in self._tasks):
                await asyncio.sleep(0.5) 
                for task in self._tasks:
                    if not task.done():
                        print(f"AudioLoop: WARNING - Task {task.get_name()} still not done after cancellation attempt.")
            
            if self.session:
                print("AudioLoop: Gemini session should be closed by 'async with' context manager.")
                self.session = None 
            
            print("AudioLoop: Terminating self.pya (PyAudio instance)...") # Changed print message
            try:
                if self.pya: # Check if self.pya exists
                    self.pya.terminate()
                    print("AudioLoop: self.pya (PyAudio instance) terminated successfully.")
            except Exception as e:
                print(f"AudioLoop: Error terminating self.pya (PyAudio instance): {e}")
            self.pya = None # Clear the reference
            print("AudioLoop: run method finished cleanup.")


async def handle_tool_call(session, server_tool_call: types.LiveServerToolCall):
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


if __name__ == "__main__":
    print("Starting AudioLoop directly for testing...")
    user_q = asyncio.Queue()
    model_q = asyncio.Queue()
    loop_task = None 
    
    main_loop = AudioLoop(user_text_input_queue=user_q, model_text_output_queue=model_q)
    
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
            await asyncio.wait_for(loop_task, timeout=10.0) 
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

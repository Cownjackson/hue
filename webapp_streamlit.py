import streamlit as st
import sys
import asyncio
import os
import pyaudio # Keep for constants like paInt16, but not for stream opening if streamlit-webrtc handles it
import numpy as np # Added numpy import
import librosa # Added librosa
from google import genai
from google.genai import types
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, RTCConfiguration

# Import tools if they are in a separate file, e.g., from tools import turn_office_light_on, turn_office_light_off
# For now, let's define them inline for simplicity or assume they are globally available if running in the same context
# as google_ai_sample.py. If not, they need to be explicitly defined or imported.

# --- Mock/Placeholder Tool Functions ---
def turn_office_light_on():
    st.toast("Office light turned ON")
    print("Mock: Office light turned ON")
    return {"status": "success", "message": "Office light turned on."}

def turn_office_light_off():
    st.toast("Office light turned OFF")
    print("Mock: Office light turned OFF")
    return {"status": "success", "message": "Office light turned off."}
# --- End Mock/Placeholder Tool Functions ---


load_dotenv()

# Configuration from google_ai_sample.py
FORMAT = pyaudio.paInt16  # Used for interpreting audio data if necessary
CHANNELS = 1
SEND_SAMPLE_RATE = 16000  # Gemini expected sample rate for input
RECEIVE_SAMPLE_RATE = 24000 # Gemini output sample rate
# CHUNK_SIZE = 1024 # Might be dictated by streamlit-webrtc frame size

MODEL = "models/gemini-2.0-flash-live-001"

# Initialize Gemini Client
# Ensure API key is loaded via .env or environment variables
if not os.environ.get("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY not found. Please set it in your .env file or environment variables.")
    st.stop()

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# Define the tools for the voice agent (same as google_ai_sample.py)
gemini_tools = [
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

GEMINI_CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO", "TEXT"], # Request both audio and text
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
        )
    ),
    tools=gemini_tools,
)

# Session state variables
if "gemini_session" not in st.session_state:
    st.session_state.gemini_session = None
if "audio_loop_task" not in st.session_state:
    st.session_state.audio_loop_task = None
if "audio_loop_instance" not in st.session_state:
    st.session_state.audio_loop_instance = None
if "messages" not in st.session_state: # For chat display
    st.session_state.messages = []
if "user_text_input_queue" not in st.session_state:
    st.session_state.user_text_input_queue = asyncio.Queue()
if "model_text_output_queue" not in st.session_state: # For text from model to UI
    st.session_state.model_text_output_queue = asyncio.Queue()
if "gemini_audio_output_queue" not in st.session_state: # For audio from Gemini to be played
    st.session_state.gemini_audio_output_queue = asyncio.Queue()
if "webrtc_audio_input_queue" not in st.session_state:
    print("DEBUG: webrtc_audio_input_queue NOT IN SESSION STATE right before webrtc_streamer call! Initializing.", file=sys.stderr)
    st.session_state.webrtc_audio_input_queue = asyncio.Queue()
elif not isinstance(st.session_state.webrtc_audio_input_queue, asyncio.Queue):
    print("DEBUG: webrtc_audio_input_queue IS IN SESSION STATE but not a Queue! Re-initializing.", file=sys.stderr)
    st.session_state.webrtc_audio_input_queue = asyncio.Queue()
else:
    print("DEBUG: webrtc_audio_input_queue is present and is a Queue right before webrtc_streamer call.", file=sys.stderr)
# Initialize webrtc_desired_playing_state if not already in session state
if "webrtc_desired_playing_state" not in st.session_state:
    st.session_state.webrtc_desired_playing_state = False
if "ui_updater_task" not in st.session_state: # Added for completeness based on later usage
    st.session_state.ui_updater_task = None


async def handle_tool_call_streamlit(session, server_tool_call: types.LiveServerToolCall):
    print(f"Server tool call received: {server_tool_call}")
    tool_call_name = "Unknown tool"
    if server_tool_call.function_calls and server_tool_call.function_calls[0]:
        tool_call_name = server_tool_call.function_calls[0].name
    st.toast(f"🔧 Tool call: {tool_call_name}")

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
                function_responses.append(types.FunctionResponse(id=tool_id, name=tool_name, response=tool_result_dict))
            except Exception as e:
                print(f"An exception occurred while executing tool {tool_name} (ID: {tool_id}): {e}")
                st.error(f"Error executing tool {tool_name}: {e}")
                function_responses.append(types.FunctionResponse(id=tool_id, name=tool_name, response={"status": "error", "message": f"Exception during {tool_name} execution: {str(e)}"}))
    if function_responses:
        print(f"Sending tool responses to Gemini: {function_responses}")
        await session.send_tool_response(function_responses=function_responses)
    else:
        print("No function calls were processed or found in the server_tool_call message.")

class StreamlitAudioLoop:
    def __init__(self, user_text_q: asyncio.Queue, model_text_q: asyncio.Queue, gemini_audio_q: asyncio.Queue, webrtc_audio_q: asyncio.Queue):
        self.user_text_input_queue = user_text_q
        self.model_text_output_queue = model_text_q
        self.gemini_audio_output_queue = gemini_audio_q
        self.webrtc_audio_input_queue = webrtc_audio_q
        self.session: types.AsyncLiveSession | None = None
        self._shutdown_event = asyncio.Event()
        self._tasks = []
        self.pya = pyaudio.PyAudio()

    async def _shutdown_monitor(self):
        await self._shutdown_event.wait()
        print("StreamlitAudioLoop: Shutdown signaled.")

    async def send_text_from_queue(self):
        print("StreamlitAudioLoop: send_text_from_queue started.")
        try:
            while not self._shutdown_event.is_set():
                try:
                    text = await asyncio.wait_for(self.user_text_input_queue.get(), timeout=0.2)
                    print(f"StreamlitAudioLoop: Got text from input queue: '{text}'")
                    if text is None:
                        self._shutdown_event.set()
                        break
                    text_to_send = text if text else "."
                    user_content = types.Content(role="user", parts=[types.Part(text=text_to_send)])
                    if self.session and not self._shutdown_event.is_set():
                        await self.session.send_client_content(turns=user_content, turn_complete=True)
                        print(f"StreamlitAudioLoop: Sent to Gemini: '{text_to_send}'")
                        # UI update for user message is handled by chat_input callback
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    if not self._shutdown_event.is_set():
                        print(f"StreamlitAudioLoop: Error sending client content: {e}")
                        # Consider pushing error to a UI queue if needed
        finally:
            print("StreamlitAudioLoop: send_text_from_queue finished.")

    async def send_webrtc_audio(self):
        print("StreamlitAudioLoop: send_webrtc_audio started.")
        try:
            while not self._shutdown_event.is_set():
                try:
                    audio_data_bytes = await asyncio.wait_for(self.webrtc_audio_input_queue.get(), timeout=0.2)
                    if audio_data_bytes is None:
                        break 
                    if self.session and not self._shutdown_event.is_set():
                        part = types.Part(inline_data=types.Blob(data=audio_data_bytes, mime_type="audio/pcm"))
                        await self.session.send_realtime_input(media=part.inline_data)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    if not self._shutdown_event.is_set(): print(f"StreamlitAudioLoop: Error in send_webrtc_audio: {e}")
        finally:
            print("StreamlitAudioLoop: send_webrtc_audio finished.")

    async def receive_from_gemini(self):
        print("StreamlitAudioLoop: receive_from_gemini started.")
        try:
            while not self._shutdown_event.is_set():
                if not self.session: await asyncio.sleep(0.1); continue
                try:
                    response_message = await asyncio.wait_for(self.session.receive().__anext__(), timeout=0.2)
                    if server_tool_call := response_message.tool_call:
                        if not self._shutdown_event.is_set(): await handle_tool_call_streamlit(self.session, server_tool_call)
                        continue
                    if server_content := response_message.server_content:
                        if model_turn_content := server_content.model_turn:
                            full_text_response = []
                            for part in model_turn_content.parts:
                                if self._shutdown_event.is_set(): break
                                if part.text: full_text_response.append(part.text)
                                elif part.inline_data and part.inline_data.data:
                                    await self.gemini_audio_output_queue.put(part.inline_data.data)
                            if full_text_response and not self._shutdown_event.is_set():
                                await self.model_text_output_queue.put("".join(full_text_response))
                        if server_content.interrupted and not self._shutdown_event.is_set():
                            await self.model_text_output_queue.put("[Playback Interrupted]")
                            while not self.gemini_audio_output_queue.empty(): self.gemini_audio_output_queue.get_nowait()
                except asyncio.TimeoutError:
                    continue
                except StopAsyncIteration:
                    if not self._shutdown_event.is_set(): self._shutdown_event.set()
                    break
                except Exception as e:
                    if not self._shutdown_event.is_set(): print(f"StreamlitAudioLoop: Error in receive_from_gemini: {e}"); self._shutdown_event.set()
                    break
        finally:
            print("StreamlitAudioLoop: receive_from_gemini finished.")

    async def play_gemini_audio(self):
        print("StreamlitAudioLoop: play_gemini_audio started.")
        audio_stream = None
        try:
            audio_stream = await asyncio.to_thread(self.pya.open, format=FORMAT, channels=CHANNELS, rate=RECEIVE_SAMPLE_RATE, output=True)
            print("StreamlitAudioLoop: PyAudio output stream opened.")
            while not self._shutdown_event.is_set():
                try:
                    audio_chunk = await asyncio.wait_for(self.gemini_audio_output_queue.get(), timeout=0.2)
                    if audio_chunk is None: break
                    if not self._shutdown_event.is_set() and audio_stream:
                        await asyncio.to_thread(audio_stream.write, audio_chunk)
                except asyncio.TimeoutError:
                    continue
        except Exception as e:
            if not self._shutdown_event.is_set(): print(f"StreamlitAudioLoop: Error in play_gemini_audio: {e}")
        finally:
            if audio_stream:
                try: 
                    await asyncio.to_thread(audio_stream.stop_stream)
                    await asyncio.to_thread(audio_stream.close)
                except Exception as e: print(f"StreamlitAudioLoop: Error closing PyAudio output stream: {e}")
            print("StreamlitAudioLoop: play_gemini_audio finished.")

    async def run(self, webrtc_ctx): # webrtc_ctx might not be needed if queues are sufficient
        print("StreamlitAudioLoop: run method started.")
        self._shutdown_event.clear()
        self._tasks = []
        try:
            async with client.aio.live.connect(model=MODEL, config=GEMINI_CONFIG) as session:
                self.session = session
                st.session_state.gemini_session = session
                print("StreamlitAudioLoop: Gemini session connected.")
                st.toast("🎙️ Voice Assistant Connected!")
                async with asyncio.TaskGroup() as tg:
                    self._tasks.append(tg.create_task(self._shutdown_monitor(), name="shutdown_monitor"))
                    self._tasks.append(tg.create_task(self.send_text_from_queue(), name="send_text"))
                    self._tasks.append(tg.create_task(self.send_webrtc_audio(), name="send_webrtc_audio"))
                    self._tasks.append(tg.create_task(self.receive_from_gemini(), name="receive_gemini"))
                    self._tasks.append(tg.create_task(self.play_gemini_audio(), name="play_gemini_audio"))
                print("StreamlitAudioLoop: TaskGroup finished.")
        except asyncio.CancelledError: print("StreamlitAudioLoop: Run method cancelled."); self._shutdown_event.set()
        except Exception as e: print(f"StreamlitAudioLoop: General exception in run: {e}"); self._shutdown_event.set()
        finally:
            print("StreamlitAudioLoop: run method finally block...")
            self._shutdown_event.set()
            st.session_state.gemini_session = None
            # Send sentinels
            for q in [self.user_text_input_queue, self.webrtc_audio_input_queue, self.gemini_audio_output_queue, self.model_text_output_queue]:
                if q: await q.put(None)
            await asyncio.sleep(0.5) # Brief pause for tasks to process sentinels
            for task in self._tasks: 
                if not task.done(): task.cancel()
            if self.pya: self.pya.terminate()
            st.toast("🎙️ Voice Assistant Disconnected.")
            print("StreamlitAudioLoop: run method finished cleanup.")

class WebRTCAudioReceiver(AudioProcessorBase):
    def __init__(self, audio_input_queue: asyncio.Queue):
        self.audio_input_queue = audio_input_queue
        self.target_sr = SEND_SAMPLE_RATE
        self.target_channels = CHANNELS
        print(f"DEBUG: WebRTCAudioReceiver instance created with queue: {self.audio_input_queue}", file=sys.stderr)

    async def recv_queued(self, frames):
        for frame in frames:
            raw_audio_nd = frame.to_ndarray()
            current_sr = frame.sample_rate
            current_channels = frame.layout.nb_channels
            if current_channels > self.target_channels:
                if raw_audio_nd.ndim > 1 :
                    raw_audio_nd = librosa.to_mono(raw_audio_nd.T)
            if current_sr != self.target_sr:
                if not np.issubdtype(raw_audio_nd.dtype, np.floating):
                    raw_audio_nd = raw_audio_nd.astype(np.float32) / np.iinfo(raw_audio_nd.dtype).max
                resampled_audio_nd = librosa.resample(raw_audio_nd, orig_sr=current_sr, target_sr=self.target_sr)
            else:
                resampled_audio_nd = raw_audio_nd
            audio_int16 = (resampled_audio_nd * 32767).astype(np.int16)
            await self.audio_input_queue.put(audio_int16.tobytes())
        return frames

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Factory function that captures the queue
def create_audio_receiver_factory(queue_instance: asyncio.Queue):
    print(f"DEBUG: create_audio_receiver_factory called with queue: {queue_instance}", file=sys.stderr)
    def factory():
        print(f"DEBUG: Actual factory invoked, returning WebRTCAudioReceiver with queue: {queue_instance}", file=sys.stderr)
        return WebRTCAudioReceiver(queue_instance)
    return factory

# WebRTC component
webrtc_ctx = webrtc_streamer(
    key="gemini-voice-chat",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=create_audio_receiver_factory(st.session_state.webrtc_audio_input_queue),
    media_stream_constraints={"video": False, "audio": True},
    # rtc_configuration=RTC_CONFIGURATION, # Keep commented out for now
    desired_playing_state=st.session_state.webrtc_desired_playing_state
)

# Debug output for WebRTC state
st.write(f"WebRTC Debug: state={webrtc_ctx.state}, playing={getattr(webrtc_ctx.state, 'playing', None)}")

# Print to the Streamlit server console
import sys
print(f"WebRTC state (server): {webrtc_ctx.state}", file=sys.stderr)
if hasattr(webrtc_ctx, 'audio_receiver'):
    print("Audio receiver exists", file=sys.stderr)
else:
    print("No audio receiver", file=sys.stderr)

start_stop_col, status_col = st.columns([1,2])

with start_stop_col:
    if not st.session_state.audio_loop_task: # If our main audio processing loop is NOT running
        if st.button("▶️ Start Session", key="start_session_button"):
            # First, express our desire for WebRTC to be active.
            st.session_state.webrtc_desired_playing_state = True
            # Rerun to allow webrtc_streamer to react to desired_playing_state=True.
            # The user might need to interact with the webrtc_streamer UI (its own START, permissions)
            st.rerun()

        # This block will be evaluated on reruns, including the one after "Start Session" is clicked.
        if st.session_state.webrtc_desired_playing_state: # If we have expressed a desire for WebRTC to be active
            if webrtc_ctx.state.playing: # And WebRTC component confirms it IS playing
                # Proceed to start the main audio loop, only if it's not already started.
                # This check (if not st.session_state.audio_loop_task) is implicitly handled by the outer if.
                st.info("🚀 Voice Assistant Activating...")
                st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you?"}]
                
                # Reinitialize queues for a fresh session
                st.session_state.user_text_input_queue = asyncio.Queue()
                st.session_state.model_text_output_queue = asyncio.Queue()
                st.session_state.gemini_audio_output_queue = asyncio.Queue()
                # The webrtc_audio_input_queue is passed to WebRTCAudioReceiver,
                # which is recreated by webrtc_streamer if its key/factory changes,
                # or on a full rerun if the factory always provides a new queue.
                # For safety, let's ensure it's fresh if we are re-starting a session.
                st.session_state.webrtc_audio_input_queue = asyncio.Queue() 
                
                loop_instance = StreamlitAudioLoop(
                    st.session_state.user_text_input_queue,
                    st.session_state.model_text_output_queue,
                    st.session_state.gemini_audio_output_queue,
                    st.session_state.webrtc_audio_input_queue # Pass the potentially new queue
                )
                st.session_state.audio_loop_instance = loop_instance
                try:
                    st.session_state.audio_loop_task = asyncio.create_task(loop_instance.run(webrtc_ctx))
                    print("StreamlitAudioLoop task created.")
                    
                    if "ui_updater_task" not in st.session_state or st.session_state.ui_updater_task.done():
                        st.session_state.ui_updater_task = asyncio.create_task(ui_model_text_updater())
                        print("UI model text updater task started.")
                    
                    st.rerun() # Rerun to update UI status after tasks are created
                except Exception as e:
                    st.error(f"Failed to start session: {e}"); print(f"Error creating task: {e}")
                    st.session_state.webrtc_desired_playing_state = False # Reset desire if failed
            else: # We want WebRTC to play, but it's not.
                st.warning("Microphone not active. Please click the 'START' button within the microphone component above, select your microphone, and grant permissions. The component should then indicate it's playing (e.g., show 'Source playing' or similar).")

    else: # audio_loop_task IS running
        if st.button("⏹️ Stop Session", key="stop_session_button"):
            st.info("🛑 Stopping Voice Assistant...")
            st.session_state.webrtc_desired_playing_state = False # Indicate desire to stop WebRTC
            
            if st.session_state.audio_loop_instance:
                st.session_state.audio_loop_instance._shutdown_event.set()
            
            if "ui_updater_task" in st.session_state and st.session_state.ui_updater_task and not st.session_state.ui_updater_task.done():
                try:
                    st.session_state.ui_updater_task.cancel()
                    print("UI updater task cancellation requested.")
                except Exception as e: 
                    print(f"Error cancelling UI updater task: {e}")
            st.session_state.ui_updater_task = None
    
            st.session_state.audio_loop_task = None # Will be awaited/cleaned up by StreamlitAudioLoop's finally
            st.session_state.audio_loop_instance = None
            st.session_state.gemini_session = None
            st.rerun()

with status_col:
    status_text = "⚪ Idle"
    if st.session_state.audio_loop_task and not st.session_state.audio_loop_task.done():
        status_text = "🟢 Running"
    elif st.session_state.gemini_session:
        status_text = "🟡 Connecting..."
    st.markdown(f"**Status:** {status_text}")
    st.markdown(f"**WebRTC:** {'🎤 Active' if webrtc_ctx.state.playing else ' inactive/needs permission'}")

st.markdown("""---    
**Note:** Audio processing in `WebRTCAudioReceiver` currently does a basic data type conversion. For accurate voice recognition, it **must be updated to correctly resample audio from your microphone's native sample rate (often 48kHz) to the required 16kHz** for the Gemini model. This typically involves using a library like `librosa`.
""") 
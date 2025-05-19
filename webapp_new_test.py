import streamlit as st
import sys
import asyncio
import os
import pyaudio 
import numpy as np
import librosa 
from google import genai # Uncommented
from google.genai import types as genai_types # Uncommented, aliased to avoid conflict if any
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, RTCConfiguration

# Configuration
FORMAT = pyaudio.paInt16 # PyAudio format (still used for target format, though not directly by receiver for output)
CHANNELS = 1
SEND_SAMPLE_RATE = 16000 # Target sample rate for processing/Gemini
MODEL = "models/gemini-2.0-flash-live-001" # Reverted to user's model preference

load_dotenv()

# --- Session state variables initialization ---
if "webrtc_audio_input_queue" not in st.session_state:
    st.session_state.webrtc_audio_input_queue = asyncio.Queue()
# if "webrtc_desired_playing_state" not in st.session_state: # Will be handled by conversation_active
#    st.session_state.webrtc_desired_playing_state = False
if "audio_active" not in st.session_state: # Renamed to conversation_active later
    st.session_state.audio_active = False
if "messages" not in st.session_state: 
    st.session_state.messages = []

# New/Restored session state variables
if "audio_loop_task" not in st.session_state:
    st.session_state.audio_loop_task = None
if "ui_updater_task" not in st.session_state:
    st.session_state.ui_updater_task = None
if "gemini_client" not in st.session_state:
    st.session_state.gemini_client = None
if "audio_agent" not in st.session_state:
    st.session_state.audio_agent = None
if "text_input_content" not in st.session_state:
    st.session_state.text_input_content = ""
if "last_user_text" not in st.session_state:
    st.session_state.last_user_text = None
if "last_model_response" not in st.session_state:
    st.session_state.last_model_response = None
if "conversation_started_time" not in st.session_state:
    st.session_state.conversation_started_time = None
# if "run_id" not in st.session_state: # If StreamlitAudioLoop uses it
# st.session_state.run_id = None 
if "system_instruction_processed" not in st.session_state:
    st.session_state.system_instruction_processed = False
if "valid_event_loop" not in st.session_state:
    st.session_state.valid_event_loop = None
if "conversation_active" not in st.session_state: # More descriptive name
    st.session_state.conversation_active = False
if "webrtc_playing_state" not in st.session_state: # For desired playing state of webrtc
    st.session_state.webrtc_playing_state = False


class WebRTCAudioReceiver(AudioProcessorBase):
    def __init__(self, audio_input_queue: asyncio.Queue):
        super().__init__()
        self.audio_input_queue = audio_input_queue # Store the queue
        self.target_sr = SEND_SAMPLE_RATE 
        self.target_channels = CHANNELS
        self.loop_captured_in_recv = False # Flag to capture loop only once
        print(f"DEBUG: WebRTCAudioReceiver __init__ called. Loop will be captured in recv_queued if not already present.", file=sys.stderr)

        # Don't try to capture loop here as it might not be reliable during __init__
        # Instead, we will capture it in the first call to recv_queued

    async def recv_queued(self, frames): 
        # Capture event loop on first call if not already set and not yet attempted in recv_queued
        if not st.session_state.get("valid_event_loop") and not self.loop_captured_in_recv:
            try:
                loop = asyncio.get_running_loop()
                st.session_state.valid_event_loop = loop
                self.loop_captured_in_recv = True # Mark that we've tried/succeeded
                print(f"DEBUG (WebRTCAudioReceiver.recv_queued): Captured event loop: {loop}, running={loop.is_running()}", file=sys.stderr)
            except RuntimeError as e:
                self.loop_captured_in_recv = True # Mark that we've tried
                print(f"ERROR (WebRTCAudioReceiver.recv_queued): Failed to capture event loop: {e}", file=sys.stderr)
                # st.session_state.valid_event_loop remains None or as previously set

        all_audio_bytes_list = [] # Collect bytes if multiple frames are processed
        for frame in frames:
            try:
                # --- INSPECT AudioFrame and raw_audio_nd --- 
                # print(f"DEBUG: frame: format={frame.format.name}, layout={frame.layout.name}, samples={frame.samples}, rate={frame.sample_rate}, pts={frame.pts}, time_base={frame.time_base}", file=sys.stderr)
                raw_audio_nd = frame.to_ndarray() 
                # print(f"DEBUG: raw_audio_nd (from frame.to_ndarray()): shape={raw_audio_nd.shape}, dtype={raw_audio_nd.dtype}, min={np.min(raw_audio_nd) if raw_audio_nd.size > 0 else 'N/A'}, max={np.max(raw_audio_nd) if raw_audio_nd.size > 0 else 'N/A'}", file=sys.stderr)
                # --- END INSPECTION ---

                current_sr = frame.sample_rate
                current_channels = frame.layout.nb_channels
                samples_per_channel = frame.samples

                if raw_audio_nd.ndim == 2 and raw_audio_nd.shape[0] == 1:
                    interleaved_samples = raw_audio_nd[0] 
                elif raw_audio_nd.ndim == 1: 
                    interleaved_samples = raw_audio_nd
                else:
                    print(f"WARN: Unexpected shape for raw_audio_nd: {raw_audio_nd.shape}. Using zeros.", file=sys.stderr)
                    interleaved_samples = np.zeros(samples_per_channel * current_channels if current_channels > 0 else samples_per_channel, dtype=np.int16)

                if np.issubdtype(interleaved_samples.dtype, np.integer):
                    float_audio = interleaved_samples.astype(np.float32) / 32768.0
                elif np.issubdtype(interleaved_samples.dtype, np.floating):
                    float_audio = interleaved_samples 
                else:
                    print(f"WARN: Unexpected audio data type from frame: {interleaved_samples.dtype}. Attempting float conversion.", file=sys.stderr)
                    float_audio = interleaved_samples.astype(np.float32)
                
                mono_audio_for_resample = np.array([], dtype=np.float32)

                if current_channels == 2 and self.target_channels == 1:
                    expected_size = samples_per_channel * 2
                    if float_audio.size == expected_size:
                        stereo_reshaped = float_audio.reshape((-1, 2)).T
                        mono_audio_for_resample = librosa.to_mono(stereo_reshaped)
                    else:
                        print(f"WARN: Mismatch in expected stereo samples (expected {expected_size}, got {float_audio.size}). Using zeros.", file=sys.stderr)
                        mono_audio_for_resample = np.zeros(samples_per_channel, dtype=np.float32)
                elif current_channels == 1:
                    if float_audio.size == samples_per_channel:
                        mono_audio_for_resample = float_audio
                    else: 
                         print(f"WARN: Mismatch in expected mono samples (expected {samples_per_channel}, got {float_audio.size}). Using zeros.", file=sys.stderr)
                         mono_audio_for_resample = np.zeros(samples_per_channel, dtype=np.float32)
                else: 
                    print(f"WARN: Unsupported channel configuration: {current_channels} to {self.target_channels}. Using zeros.", file=sys.stderr)
                    mono_audio_for_resample = np.zeros(samples_per_channel, dtype=np.float32)
                
                if current_sr != self.target_sr:
                    if mono_audio_for_resample.size > 0: 
                        resampled_audio_nd = librosa.resample(mono_audio_for_resample, orig_sr=current_sr, target_sr=self.target_sr)
                    else: 
                        resampled_audio_nd = mono_audio_for_resample 
                else:
                    resampled_audio_nd = mono_audio_for_resample 
                
                audio_int16 = (np.clip(resampled_audio_nd, -1.0, 1.0) * 32767).astype(np.int16)
                
                # Instead of writing to PyAudio stream, put to queue
                if audio_int16.size > 0:
                    all_audio_bytes_list.append(audio_int16.tobytes())
                    # print(f"DEBUG: audio_int16 stats: min={np.min(audio_int16)}, max={np.max(audio_int16)}, mean={np.mean(audio_int16):.2f}, len={len(audio_int16)}", file=sys.stderr)
                # else:
                    # print("DEBUG: audio_int16 is empty after processing!", file=sys.stderr)

            except Exception as e:
                print(f"ERROR processing frame in recv_queued: {e}", file=sys.stderr)
        
        if all_audio_bytes_list:
            # Combine all byte strings and put into queue
            combined_bytes = b"".join(all_audio_bytes_list)
            try:
                # Using put_nowait for simplicity, assuming the queue won't be full often
                # For more robust handling, especially if StreamlitAudioLoop is slow,
                # this might need to be `await self.audio_input_queue.put(combined_bytes)`
                # which would require recv_queued to be an async method.
                # For now, let's try put_nowait and see. If queue full errors appear, we'll change.
                self.audio_input_queue.put_nowait(combined_bytes)
            except asyncio.QueueFull:
                print("ERROR: WebRTCAudioReceiver audio_input_queue is full. Discarding audio.", file=sys.stderr)
            except Exception as e:
                print(f"ERROR: WebRTCAudioReceiver queue put error: {e}", file=sys.stderr)
        return frames # Must return the frames

    def __del__(self):
        print("DEBUG: WebRTCAudioReceiver __del__ called.", file=sys.stderr)
        # No PyAudio stream to close or instance to terminate directly here anymore
        # The queue is managed by st.session_state

# --- Placeholder for GeminiAudioAgent and StreamlitAudioLoop ---
# (These will be uncommented/restored from previous versions or google_ai_sample.py)

class GeminiAudioAgent:
    def __init__(self, api_key, model_name, system_instruction=None):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API key not provided. Set the GOOGLE_API_KEY environment variable.")
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.system_instruction = system_instruction
        self._client = None
        self._conversation_history = [] # For text-based turns if needed

    def _initialize_client(self):
        if not self._client:
            self._client = genai.GenerativeModel(self.model_name)

    async def start_async_conversation(self):
        self._initialize_client()
        # For live models, the conversation is often managed through send_message_async
        # and doesn't have a formal 'start_chat_async' like some SDKs.
        # We'll manage the history implicitly or via the model's capabilities.
        print("DEBUG: GeminiAudioAgent async conversation started (client initialized).", file=sys.stderr)
        # If a system instruction is present, we might want to send it first
        # However, for streaming audio, the 'system_instruction' is often part of the initial Content.
        # For now, we assume it's handled by StreamlitAudioLoop's first message.
        return self # Or some representation of the conversation if the API provides it

    async def send_audio_data(self, audio_bytes, is_first_chunk=False):
        self._initialize_client()
        # This is a simplified mock. Actual implementation depends on Live API specifics.
        # Typically, you'd stream Parts of Content objects.
        # This method will likely be called by StreamlitAudioLoop.
        # For a true streaming API, you'd send chunks. Here, we simulate a response.
        
        print(f"DEBUG: GeminiAudioAgent received audio_bytes (len: {len(audio_bytes)}), is_first_chunk: {is_first_chunk}", file=sys.stderr)
        
        # This is where you'd use the actual Gemini streaming audio API.
        # For now, let's assume it processes and we'll get a response elsewhere or it's fire-and-forget for this part.
        # In a real scenario, this would be part of a larger streaming interaction.
        # This function might not directly return the model's speech/text but rather confirmation of send.
        # The response would come through another part of the API (e.g. iterating over chat.history for non-live, or stream object)
        
        # Placeholder: Simulate sending and that the model might respond.
        # Actual response handling will be in StreamlitAudioLoop or ui_model_text_updater
        pass

    async def send_text_data(self, text_query):
        self._initialize_client()
        print(f"DEBUG: GeminiAudioAgent sending text: {text_query}", file=sys.stderr)
        try:
            # This assumes a non-streaming text chat for simplicity of adding text input
            # For a unified audio/text live model, this might also be part of the streamed Content
            response = await self._client.generate_content_async(text_query) # Simplified
            self._conversation_history.append({"role": "user", "parts": [text_query]})
            self._conversation_history.append({"role": "model", "parts": [response.text]})
            print(f"DEBUG: Gemini Text Response: {response.text}", file=sys.stderr)
            return response.text
        except Exception as e:
            print(f"ERROR in GeminiAudioAgent send_text_data: {e}", file=sys.stderr)
            return f"Error: {e}"

    async def end_async_conversation(self):
        # Clean up resources if necessary
        print("DEBUG: GeminiAudioAgent async conversation ended.", file=sys.stderr)
        self._client = None # Or other cleanup
        self._conversation_history = []


class StreamlitAudioLoop:
    def __init__(self, audio_input_queue, gemini_agent, ui_updater_callback, system_instruction=None):
        self.audio_input_queue = audio_input_queue
        self.gemini_agent = gemini_agent
        self.ui_updater_callback = ui_updater_callback
        self.is_running = False
        self._audio_task = None
        self.system_instruction = system_instruction
        self.system_instruction_sent = False
        self.conversation_start_time = None

    async def _process_audio(self):
        print("DEBUG: StreamlitAudioLoop: Starting _process_audio loop.", file=sys.stderr)
        
        # Start the conversation with Gemini, potentially sending system instruction first
        # The Live API might handle system instructions differently (e.g., as part of first content)
        # For now, let's assume send_audio_data can handle a special "first chunk"
        
        is_first_audio_chunk = True
        if self.system_instruction and not self.system_instruction_sent:
            # How system instructions are sent in Live API needs to be confirmed.
            # It might be a special part in the first Content object, or a separate call.
            # For now, we'll assume it's prepended or part of the first audio send logic.
            # Let's defer this to the agent or make it part of the first audio message.
            print(f"DEBUG: StreamlitAudioLoop: System instruction to be sent: {self.system_instruction}", file=sys.stderr)
            # We could send it as a text message if the model supports multimodal Content
            # await self.gemini_agent.send_text_data(f"SYSTEM_INSTRUCTION: {self.system_instruction}")
            # self.system_instruction_sent = True 
            # Or assume it's handled by the agent when is_first_chunk is true for audio.
            pass

        while self.is_running:
            try:
                audio_bytes = await asyncio.wait_for(self.audio_input_queue.get(), timeout=0.1)
                if audio_bytes:
                    # print(f"DEBUG: StreamlitAudioLoop: Got audio bytes (len: {len(audio_bytes)}) from queue.", file=sys.stderr)
                    
                    # This is where we would send to Gemini.
                    # The Live API has a specific way to stream audio, likely using `glm.Content` and `glm.Part`.
                    # Example structure (conceptual):
                    # content_parts = []
                    # if is_first_audio_chunk and self.system_instruction:
                    #    content_parts.append(glm.Part(text=self.system_instruction)) # Or however it's specified
                    #    self.system_instruction_sent = True
                    # content_parts.append(glm.Part(inline_data=glm.Blob(mime_type="audio/wav", data=audio_bytes)))
                    # request_content = glm.Content(parts=content_parts, role="user")
                    
                    # For now, calling the simplified agent method:
                    await self.gemini_agent.send_audio_data(audio_bytes, is_first_chunk=is_first_audio_chunk)
                    is_first_audio_chunk = False

                    # Placeholder for getting response. In a real Live API, responses (text, tool calls)
                    # would stream back. We'd need to iterate over that stream.
                    # For this phase, ui_updater_callback will just show that audio is flowing.
                    # We'll add actual Gemini response handling in the next phase.
                    await self.ui_updater_callback(user_utterance="[Audio being processed...]", model_response=None, is_final=False)

                self.audio_input_queue.task_done()
            except asyncio.TimeoutError:
                continue # No audio, continue loop
            except Exception as e:
                print(f"ERROR in StreamlitAudioLoop _process_audio: {e}", file=sys.stderr)
                # Potentially stop or signal error
                break 
        print("DEBUG: StreamlitAudioLoop: Exited _process_audio loop.", file=sys.stderr)

    async def start(self):
        if not self.is_running:
            self.is_running = True
            self.conversation_start_time = asyncio.get_event_loop().time()
            await self.gemini_agent.start_async_conversation() # Initialize client etc.
            print("DEBUG: StreamlitAudioLoop started.", file=sys.stderr)
            # The task should be created on the valid_event_loop from session_state
            if st.session_state.valid_event_loop:
                self._audio_task = st.session_state.valid_event_loop.create_task(self._process_audio())
                print(f"DEBUG: StreamlitAudioLoop: _process_audio task created on loop {st.session_state.valid_event_loop}", file=sys.stderr)
            else:
                print("ERROR: StreamlitAudioLoop: No valid event loop to create _process_audio task.", file=sys.stderr)
                self.is_running = False # Cannot start
                return

    async def stop(self):
        if self.is_running:
            self.is_running = False
            print("DEBUG: StreamlitAudioLoop stopping...", file=sys.stderr)
            if self._audio_task:
                try:
                    # self._audio_task.cancel() # Cancel the task
                    await asyncio.wait_for(self._audio_task, timeout=2.0) # Wait for it to finish
                except asyncio.CancelledError:
                    print("DEBUG: StreamlitAudioLoop: _process_audio task cancelled.", file=sys.stderr)
                except asyncio.TimeoutError:
                    print("WARN: StreamlitAudioLoop: _process_audio task timed out during stop.", file=sys.stderr)
                except Exception as e:
                    print(f"ERROR stopping StreamlitAudioLoop _audio_task: {e}", file=sys.stderr)
                self._audio_task = None
            await self.gemini_agent.end_async_conversation()
            print("DEBUG: StreamlitAudioLoop stopped.", file=sys.stderr)

async def ui_model_text_updater():
    print("DEBUG: UI Model Text Updater task started.", file=sys.stderr)
    # This function would typically get updates from Gemini (text, tool calls)
    # and update st.session_state.messages for display.
    # For now, it's a placeholder. In a real Live API, this would listen to the response stream.
    while st.session_state.get("conversation_active", False) or \
          (st.session_state.audio_loop_task and not st.session_state.audio_loop_task.done()):
        
        # In a real scenario, this is where you'd check for new messages from Gemini
        # e.g. by iterating over `chat.history` or a response stream
        # For example (conceptual):
        # new_model_text = await get_latest_gemini_response()
        # if new_model_text and new_model_text != st.session_state.get("last_model_response"):
        # st.session_state.messages.append({"role": "assistant", "content": new_model_text})
        # st.session_state.last_model_response = new_model_text
        # st.rerun() # Trigger UI update
        
        await asyncio.sleep(0.2) # Check for updates periodically
    print("DEBUG: UI Model Text Updater task finished.", file=sys.stderr)


async def handle_tool_call_streamlit(tool_call, agent): # Placeholder
    # This function would handle tool calls from Gemini.
    print(f"DEBUG: Handling tool call: {tool_call.function_call.name}", file=sys.stderr)
    # ... logic to execute the tool and send back the result ...
    # response_part = genai_types.Part(
    # function_response=genai_types.FunctionResponse(name=tool_call.function_call.name, response={"key": "value"})
    # )
    # await agent.send_parts_async([response_part]) # Simplified
    pass


# --- Main UI and Logic (Phase 2 Integration) ---
st.set_page_config(layout="wide")
st.title("🎙️ Gemini Live Voice Chat")

# Ensure API key is loaded
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("Missing GOOGLE_API_KEY environment variable. Please set it in your .env file or environment.")
    st.stop()

# System instruction (optional, can be configured)
SYSTEM_INSTRUCTION = "You are a helpful and friendly AI assistant. Keep your responses concise unless asked for details."

async def ui_updater_callback_streamlit(user_utterance=None, model_response=None, tool_name=None, tool_input=None, is_final=True):
    """Callback to update Streamlit UI with new messages."""
    if user_utterance:
        # Check if the last message was also from user and same content to avoid dupes on quick reruns
        if not st.session_state.messages or \
           st.session_state.messages[-1].get("content") != user_utterance or \
           st.session_state.messages[-1].get("role") != "user":
            st.session_state.messages.append({"role": "user", "content": user_utterance})
    
    if model_response:
        # If the last message is from the assistant and this is not a final response, append to it (stream)
        if st.session_state.messages and st.session_state.messages[-1].get("role") == "assistant" and not is_final:
            st.session_state.messages[-1]["content"] += model_response
        else: # Else, new message or final part of a streamed one
            st.session_state.messages.append({"role": "assistant", "content": model_response})
        st.session_state.last_model_response = model_response # Keep track for UI updates

    if tool_name and tool_input:
        st.session_state.messages.append({"role": "assistant", "content": f"Calling tool: `{tool_name}` with input `({tool_input})`"})

    # Smart rerun: only if there are new messages or a significant state change that affects UI display directly
    # This is tricky; for now, let's rerun if there was any model activity or user input shown.
    if user_utterance or model_response or tool_name:
        st.rerun()

# Chat message display area
chat_container = st.container(height=500, border=False)
with chat_container:
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Text input area
def handle_text_input():
    # Value is present in st.session_state.text_input_content due to user submission via the keyed widget
    user_text_from_submit = st.session_state.text_input_content 

    if user_text_from_submit and st.session_state.conversation_active:
        print(f"DEBUG: User typed: {user_text_from_submit}", file=sys.stderr)
        
        async def send_text_and_update_ui_async_task(text_to_send): # Pass text as an argument
            await ui_updater_callback_streamlit(user_utterance=text_to_send)
            if st.session_state.audio_agent:
                model_text_response = await st.session_state.audio_agent.send_text_data(text_to_send)
                if model_text_response:
                     await ui_updater_callback_streamlit(model_response=model_text_response, is_final=True)
            else:
                await ui_updater_callback_streamlit(model_response="Error: Conversation not fully active.", is_final=True)
        
        if st.session_state.valid_event_loop and st.session_state.valid_event_loop.is_running():
            # Pass the captured text as an argument to the async task.
            # This is crucial because st.session_state.text_input_content will be cleared by Streamlit
            # after this on_submit callback finishes and before the async task might execute.
            st.session_state.valid_event_loop.create_task(send_text_and_update_ui_async_task(user_text_from_submit))
        else:
            # If the event loop isn't ready, we can't reliably run the async task.
            # Display the user's message with an error, or just log and show error.
            error_message = "Error: Cannot process text input due to event loop issue."
            print(f"ERROR: {error_message}", file=sys.stderr)
            st.error(error_message)
            # Optionally, add the user message to UI with an error note if desired, but this requires careful state management.
            # For simplicity now, just show error toast.
        
        # No explicit st.rerun() here. 
        # st.chat_input with on_submit handles the rerun and clearing of the input value 
        # in st.session_state.text_input_content automatically after this callback.

st.chat_input("Type your message...", key="text_input_content", on_submit=handle_text_input, disabled=not st.session_state.get("conversation_active"))

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def create_audio_receiver_factory(queue_instance: asyncio.Queue):
    def factory():
        return WebRTCAudioReceiver(queue_instance)
    return factory

webrtc_ctx = webrtc_streamer(
    key="gemini-audio-chat",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=create_audio_receiver_factory(st.session_state.webrtc_audio_input_queue),
    media_stream_constraints={"video": False, "audio": True},
    desired_playing_state=st.session_state.webrtc_playing_state, 
    rtc_configuration=RTC_CONFIGURATION
)

async def start_conversation_tasks():
    if not st.session_state.gemini_client:
        st.session_state.gemini_client = genai.GenerativeModel(MODEL) # Initialize client directly for now
    
    if not st.session_state.audio_agent:
        st.session_state.audio_agent = GeminiAudioAgent(
            api_key=API_KEY, 
            model_name=MODEL, 
            system_instruction=SYSTEM_INSTRUCTION
        )

    if not st.session_state.audio_loop_task or st.session_state.audio_loop_task.done():
        if st.session_state.valid_event_loop and st.session_state.valid_event_loop.is_running():
            audio_loop = StreamlitAudioLoop(
                st.session_state.webrtc_audio_input_queue,
                st.session_state.audio_agent,
                ui_updater_callback_streamlit, # Pass the new callback
                system_instruction=SYSTEM_INSTRUCTION
            )
            st.session_state.audio_loop_task = st.session_state.valid_event_loop.create_task(audio_loop.start())
            print("DEBUG: StreamlitAudioLoop task creation requested.", file=sys.stderr)
        else:
            st.error("Cannot start audio processing: Event loop not available or not running.")
            st.session_state.conversation_active = False # Revert state
            st.session_state.webrtc_playing_state = False
            return

    if not st.session_state.ui_updater_task or st.session_state.ui_updater_task.done():
        if st.session_state.valid_event_loop and st.session_state.valid_event_loop.is_running():
            st.session_state.ui_updater_task = st.session_state.valid_event_loop.create_task(ui_model_text_updater())
            print("DEBUG: UI Model Text Updater task creation requested.", file=sys.stderr)
        else:
            st.error("Cannot start UI updater: Event loop not available or not running.")
            # Decide if this is critical enough to stop conversation

async def stop_conversation_tasks():
    print("DEBUG: Stopping conversation tasks...", file=sys.stderr)
    if st.session_state.audio_loop_task and not st.session_state.audio_loop_task.done():
        # Access the StreamlitAudioLoop instance itself if it's stored, or send a signal
        # For now, we assume loop.stop() is part of the task cancellation sequence
        # This needs StreamlitAudioLoop instance to be stored if we want to call .stop() on it.
        # Let's refine this: audio_loop.stop() should be called by the task itself on cancellation or flag.
        # Here, we just cancel the task.
        st.session_state.audio_loop_task.cancel()
        try:
            await st.session_state.audio_loop_task
        except asyncio.CancelledError:
            print("DEBUG: Audio loop task successfully cancelled.", file=sys.stderr)
        except Exception as e:
            print(f"Error during audio loop task cancellation: {e}", file=sys.stderr)
    
    if st.session_state.ui_updater_task and not st.session_state.ui_updater_task.done():
        st.session_state.ui_updater_task.cancel()
        try:
            await st.session_state.ui_updater_task
        except asyncio.CancelledError:
            print("DEBUG: UI updater task successfully cancelled.", file=sys.stderr)
        except Exception as e:
            print(f"Error during UI updater task cancellation: {e}", file=sys.stderr)

    if st.session_state.audio_agent:
        await st.session_state.audio_agent.end_async_conversation()
        st.session_state.audio_agent = None

    # Reset states
    st.session_state.audio_loop_task = None
    st.session_state.ui_updater_task = None
    st.session_state.system_instruction_processed = False
    st.session_state.messages = [] # Clear messages on stop

# Control buttons and status display
controls_col, status_display_col = st.columns([1, 3])

with controls_col:
    if not st.session_state.conversation_active:
        if st.button("▶️ Start Conversation", key="start_conv_button"):
            st.session_state.webrtc_playing_state = True
            st.session_state.conversation_active = True
            st.session_state.messages = [] # Clear messages on new conversation
            if st.session_state.valid_event_loop and st.session_state.valid_event_loop.is_running():
                st.session_state.valid_event_loop.create_task(start_conversation_tasks())
            else:
                 # Fallback if loop not ready when button clicked (e.g. first run before WebRTC active)
                 # This situation should be less common now loop is captured in recv_queued
                asyncio.run(start_conversation_tasks()) # Risky, prefer loop.create_task
            st.rerun()
    else:
        if st.button("⏹️ Stop Conversation", key="stop_conv_button"):
            st.session_state.webrtc_playing_state = False # Stop WebRTC mic
            st.session_state.conversation_active = False
            if st.session_state.valid_event_loop and st.session_state.valid_event_loop.is_running():
                st.session_state.valid_event_loop.create_task(stop_conversation_tasks())
            else:
                asyncio.run(stop_conversation_tasks()) # Risky
            st.rerun()

with status_display_col:
    status_text = "⚪ Idle"
    webrtc_is_really_playing = hasattr(webrtc_ctx, 'state') and getattr(webrtc_ctx.state, 'playing', False)
    loop_status = "Unknown"
    if st.session_state.get("valid_event_loop"):
        loop_status = "✅ Captured" if st.session_state.valid_event_loop.is_running() else "⚠️ Captured but not running"
    else:
        loop_status = "❌ Not Captured"

    if st.session_state.conversation_active:
        if webrtc_is_really_playing and (st.session_state.audio_loop_task and not st.session_state.audio_loop_task.done()):
            status_text = "🟢 Conversation Active (Listening...)"
        elif webrtc_is_really_playing:
            status_text = "🟡 Conversation Starting (Mic OK, waiting for tasks...)"
        else:
            status_text = "🟠 Activating Mic & Tasks..."
            
    st.markdown(f"**Status:** {status_text}")
    st.markdown(f"**Event Loop:** {loop_status}")
    mic_icon = "🎤" if webrtc_is_really_playing else "🔇"
    st.markdown(f"**Microphone:** {mic_icon} {'Active' if webrtc_is_really_playing else 'Inactive / Needs Permission'}")

st.markdown("---    \n**Note:** Ensure your GOOGLE_API_KEY is set in a `.env` file or your environment variables.") 
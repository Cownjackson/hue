import streamlit as st
import asyncio
import threading
from queue import Queue as ThreadQueue
from google_ai_sample import AudioLoop, RECEIVE_SAMPLE_RATE, CHANNELS as GEMINI_OUTPUT_CHANNELS # Import rate and channels for Gemini output
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, RTCConfiguration
import numpy as np
import librosa
import av # For av.AudioFrame
# import pyaudio # No longer needed here for normal operation

# Ensure PyAudio is terminated when Streamlit shuts down (this is a bit tricky with Streamlit's execution model)
# A cleaner way would be to manage the PyAudio instance within AudioLoop and ensure its cleanup.
# For now, we rely on AudioLoop's cleanup.

def run_audioloop_in_thread(audioloop_instance, streamlit_user_q, streamlit_model_q, webrtc_audio_q, model_audio_output_q_from_st):
    """Target function for the AudioLoop thread."""
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Assign the Streamlit-managed asyncio.Queues to the audioloop_instance
    audioloop_instance.user_text_input_queue = streamlit_user_q
    audioloop_instance.model_text_output_queue = streamlit_model_q
    audioloop_instance.webrtc_audio_input_queue = webrtc_audio_q # Assign the WebRTC audio queue
    audioloop_instance.model_audio_output_queue = model_audio_output_q_from_st # Assign the new queue
    
    print("Starting AudioLoop.run() in a new thread with model_audio_output_queue...")
    try:
        loop.run_until_complete(audioloop_instance.run())
    except Exception as e:
        print(f"Exception in AudioLoop thread: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("AudioLoop thread finished its Python execution.")
        loop.close()

st.set_page_config(layout="wide")
st.title("Gemini Live Voice Agent - Text & Voice Interface")

# Initialize session state variables ONCE
if 'audioloop_running' not in st.session_state:
    st.session_state.audioloop_running = False
    st.session_state.audioloop_instance = None
    st.session_state.audioloop_thread = None
    st.session_state.user_text_input_queue = asyncio.Queue()
    st.session_state.model_text_output_queue = asyncio.Queue()
    st.session_state.webrtc_audio_queue = asyncio.Queue() # Initialize WebRTC audio queue here
    st.session_state.model_audio_output_queue = asyncio.Queue() # New queue for audio from Gemini to browser
    st.session_state.conversation_history = []
    st.session_state.stop_requested = False

TARGET_SAMPLE_RATE = 16000 # For Gemini
# PYAUDIO_FORMAT = pyaudio.paInt16 # No longer needed
# PYAUDIO_CHANNELS = 1 # No longer needed

TARGET_MIC_INPUT_SAMPLE_RATE = 16000

class WebRTCAudioRelay(AudioProcessorBase):
    def __init__(self, output_queue: asyncio.Queue): 
        self.output_queue = output_queue 
        self.last_sample_rate = None
        self.last_channels = None
        # print(f"WebRTCAudioRelay __init__ called. Output queue type: {type(self.output_queue)}") # Less verbose

    async def recv_queued(self, frames: list[av.AudioFrame], bundled_frames: list[av.AudioFrame] | None = None):
        all_processed_bytes_for_this_callback = bytearray()

        if not frames:
            return [] 

        # print(f"WebRTCAudioRelay: recv_queued called with {len(frames)} frames.") # Too verbose for normal operation

        for frame_idx, frame in enumerate(frames):
            if self.last_sample_rate != frame.sample_rate or self.last_channels != frame.layout.nb_channels:
                # print(f"WebRTCAudioRelay: Audio properties changed: SR={frame.sample_rate}, Channels={frame.layout.nb_channels}") # Informative but can be verbose
                self.last_sample_rate = frame.sample_rate
                self.last_channels = frame.layout.nb_channels

            raw_audio_nd = frame.to_ndarray()
            float_audio_nd = raw_audio_nd.astype(np.float32) / 32768.0
            
            mono_audio = None
            if frame.layout.nb_channels > 1:
                if float_audio_nd.ndim == 2 and float_audio_nd.shape[0] == 1:
                    interleaved_samples = float_audio_nd[0]
                    try:
                        reshaped_for_librosa = interleaved_samples.reshape((-1, frame.layout.nb_channels)).T
                        mono_audio = librosa.to_mono(reshaped_for_librosa)
                    except Exception as e:
                        # print(f"WebRTCAudioRelay: Error during stereo to mono conversion (reshape/to_mono): {e}...") # Original commented print
                        mono_audio = np.array([], dtype=np.float32)
                elif float_audio_nd.ndim == 2 and float_audio_nd.shape[0] == frame.layout.nb_channels:
                    try:
                        mono_audio = librosa.to_mono(float_audio_nd)
                    except Exception as e:
                        # print(f"WebRTCAudioRelay: Error during stereo to mono conversion (to_mono direct): {e}...") # Original commented print
                        mono_audio = np.array([], dtype=np.float32)
                else:
                    # print(f"WebRTCAudioRelay: Unexpected audio array shape for stereo: {float_audio_nd.shape}...") # Original commented print
                    mono_audio = np.array([], dtype=np.float32)
            elif frame.layout.nb_channels == 1:
                if float_audio_nd.ndim == 2 and float_audio_nd.shape[0] == 1:
                    mono_audio = float_audio_nd[0]
                elif float_audio_nd.ndim == 1:
                     mono_audio = float_audio_nd
                else:
                    # print(f"WebRTCAudioRelay: Unexpected mono audio array shape: {float_audio_nd.shape}...") # Original commented print
                    mono_audio = np.array([], dtype=np.float32)
            else:
                # print(f"WebRTCAudioRelay: Unknown channel layout: {frame.layout.nb_channels}...") # Original commented print
                mono_audio = np.array([], dtype=np.float32)

            if mono_audio.size == 0:
                continue

            if frame.sample_rate != TARGET_MIC_INPUT_SAMPLE_RATE:
                try:
                    resampled_audio = librosa.resample(mono_audio, orig_sr=frame.sample_rate, target_sr=TARGET_MIC_INPUT_SAMPLE_RATE)
                except Exception as e:
                    # print(f"WebRTCAudioRelay: Error during resampling: {e}...") # Original commented print
                    continue
            else:
                resampled_audio = mono_audio
            
            int16_audio = (np.clip(resampled_audio, -1.0, 1.0) * 32767).astype(np.int16)
            all_processed_bytes_for_this_callback.extend(int16_audio.tobytes())
        
        if all_processed_bytes_for_this_callback:
            try:
                self.output_queue.put_nowait(bytes(all_processed_bytes_for_this_callback))
                # print(f"WebRTCAudioRelay: Queued one consolidated block of {len(all_processed_bytes_for_this_callback)} bytes. Output queue size: {self.output_queue.qsize()}") # Too verbose
            except asyncio.QueueFull:
                print("WebRTCAudioRelay: Output queue is full. Discarding consolidated audio block.") # Keep this error
            except Exception as e:
                print(f"WebRTCAudioRelay: Error putting consolidated audio to queue: {e}") # Keep this error
        return [] 

    def __del__(self):
        # print("WebRTCAudioRelay __del__ called.") # Less verbose
        pass

class AudioPlayerProcessor(AudioProcessorBase):
    def __init__(self, model_audio_q: asyncio.Queue, target_sample_rate: int, target_channels: int):
        self.model_audio_q = model_audio_q
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.audio_buffer = bytearray() # Buffer for incomplete audio frames
        # Each audio frame for playback should ideally represent a fixed duration, e.g., 20ms.
        # For 24kHz, 16-bit mono, 20ms = 0.02s * 24000 samples/s * 1 channel * 2 bytes/sample = 960 bytes.
        self.bytes_per_20ms_frame = int(0.02 * self.target_sample_rate * self.target_channels * 2)
        self.phase = 0  # For tone generation
        self.frequency = 440  # Hz (A4 note)
        print(f"AudioPlayerProcessor initialized for TONE TEST. Target SR: {target_sample_rate}, Channels: {target_channels}, Bytes/Frame: {self.bytes_per_20ms_frame}")

    async def recv(self):
        # --- TONE GENERATION TEST --- 
        # print("AudioPlayerProcessor: recv() TONE TEST")
        samples_per_frame = self.bytes_per_20ms_frame // (self.target_channels * 2) # 2 bytes/sample (int16)
        
        t = (np.arange(samples_per_frame) + self.phase) / self.target_sample_rate
        tone = (0.3 * np.sin(2 * np.pi * self.frequency * t) * 32767).astype(np.int16) # 0.3 amplitude
        self.phase += samples_per_frame

        if self.target_channels == 1:
            audio_np_reshaped = tone.reshape(1, -1) # (1, num_samples)
        else: # Crude stereo, just duplicate mono
            audio_np_reshaped = np.vstack([tone, tone]).reshape(self.target_channels, -1, order='F')

        # print(f"AudioPlayerProcessor (TONE TEST): Sending tone frame {audio_np_reshaped.shape[1]} samples.")
        return av.AudioFrame.from_ndarray(audio_np_reshaped, format='s16', layout='mono' if self.target_channels == 1 else 'stereo', sample_rate=self.target_sample_rate)
        # --- END TONE GENERATION TEST ---

        # Original queue logic (commented out for tone test):
        # # print(\"AudioPlayerProcessor: recv() called\") 
        # # Try to get some data from the queue to fill the buffer
        # try:
        #     # Consume available data from the queue without blocking indefinitely
        #     # but wait a very short period if empty to catch data just arriving.
        #     chunk = await asyncio.wait_for(self.model_audio_q.get(), timeout=0.015) # Increased timeout
        #     if chunk is None: # Sentinel
        #         return av.AudioFrame(format=\'s16\', layout=\'mono\', samples=0, sample_rate=self.target_sample_rate)
        #     self.audio_buffer.extend(chunk)
        #     self.model_audio_q.task_done()
        #     while not self.model_audio_q.empty():
        #         try:
        #             chunk = self.model_audio_q.get_nowait()
        #             if chunk is None: break 
        #             self.audio_buffer.extend(chunk)
        #             self.model_audio_q.task_done()
        #         except asyncio.QueueEmpty:
        #             break
        # except asyncio.TimeoutError:
        #     pass 
        # except Exception as e:
        #     print(f\"AudioPlayerProcessor: Error getting from model_audio_q: {e}\")

        # if len(self.audio_buffer) >= self.bytes_per_20ms_frame:
        #     frame_data_bytes = self.audio_buffer[:self.bytes_per_20ms_frame]
        #     self.audio_buffer = self.audio_buffer[self.bytes_per_20ms_frame:]
        #     audio_np = np.frombuffer(frame_data_bytes, dtype=np.int16)
        #     if self.target_channels == 1:
        #         audio_np_reshaped = audio_np.reshape(1, -1) 
        #     else: 
        #         audio_np_reshaped = audio_np.reshape(self.target_channels, -1, order=\'F\')
        #     print(f\"AudioPlayerProcessor: Sending audio frame to browser ({audio_np_reshaped.shape[1]} samples per channel).\")
        #     return av.AudioFrame.from_ndarray(audio_np_reshaped, format=\'s16\', layout=\'mono\' if self.target_channels == 1 else \'stereo\', sample_rate=self.target_sample_rate)
        
        # silent_samples_per_channel = self.bytes_per_20ms_frame // (self.target_channels * 2)
        # return av.AudioFrame(format=\'s16\', layout=\'mono\' if self.target_channels == 1 else \'stereo\', samples=silent_samples_per_channel, sample_rate=self.target_sample_rate)

# Factory creator for WebRTCAudioRelay
def create_webrtc_audio_relay_factory(queue_instance: asyncio.Queue):
    def factory():
        return WebRTCAudioRelay(output_queue=queue_instance)
    return factory

def create_audio_player_factory(model_audio_q: asyncio.Queue):
    def factory():
        return AudioPlayerProcessor(model_audio_q=model_audio_q, target_sample_rate=RECEIVE_SAMPLE_RATE, target_channels=GEMINI_OUTPUT_CHANNELS)
    return factory

def start_audioloop():
    if not st.session_state.audioloop_running:
        # Re-initialize queues for a fresh start if they were used by a previous run
        # Ensure all queues are new for the new session
        st.session_state.user_text_input_queue = asyncio.Queue()
        st.session_state.model_text_output_queue = asyncio.Queue()
        st.session_state.webrtc_audio_queue = asyncio.Queue() # Also re-initialize here for a clean start
        st.session_state.model_audio_output_queue = asyncio.Queue() # Ensure it's fresh
        st.session_state.conversation_history = [] # Clear history for new session

        st.session_state.audioloop_instance = AudioLoop(
            user_text_input_queue=st.session_state.user_text_input_queue,
            model_text_output_queue=st.session_state.model_text_output_queue,
            webrtc_audio_input_queue=st.session_state.webrtc_audio_queue,
            model_audio_output_queue=st.session_state.model_audio_output_queue # Pass new queue
        )
        
        st.session_state.audioloop_thread = threading.Thread(
            target=run_audioloop_in_thread,
            args=(
                st.session_state.audioloop_instance,
                st.session_state.user_text_input_queue,
                st.session_state.model_text_output_queue,
                st.session_state.webrtc_audio_queue,
                st.session_state.model_audio_output_queue # Pass new queue to thread function
            ),
            daemon=True 
        )
        st.session_state.audioloop_thread.start()
        st.session_state.audioloop_running = True
        st.session_state.stop_requested = False
        st.success("AudioLoop service started!")
        print("Streamlit: AudioLoop thread started.")
        st.rerun() # Update UI immediately

def stop_audioloop():
    if st.session_state.audioloop_thread and st.session_state.audioloop_instance:
        print("Streamlit: Requesting AudioLoop stop...")
        st.session_state.stop_requested = True

        if st.session_state.user_text_input_queue:
            st.session_state.user_text_input_queue.put_nowait(None) # Signal AudioLoop to stop

        # Wait for the AudioLoop thread to finish its execution
        # The AudioLoop.run() method now has a robust finally block for cleanup.
        thread_to_join = st.session_state.audioloop_thread
        if thread_to_join.is_alive():
            print(f"Streamlit: Waiting for AudioLoop thread ({thread_to_join.name}) to join...")
            thread_to_join.join(timeout=5.0) # Wait up to 5 seconds
            if thread_to_join.is_alive():
                print("Streamlit: WARNING - AudioLoop thread did not join within timeout.")
            else:
                print("Streamlit: AudioLoop thread joined successfully.")
        else:
            print("Streamlit: AudioLoop thread was already finished.")

        st.session_state.audioloop_running = False
        st.session_state.audioloop_instance = None # Clear instance
        st.session_state.audioloop_thread = None   # Clear thread reference
        # Queues are re-created in start_audioloop for a clean slate
        st.session_state.stop_requested = False # Reset here before rerun
        st.warning("AudioLoop service stopped.")
        print("Streamlit: AudioLoop service stop process completed in Streamlit UI.")
        st.rerun() # Update UI to reflect stopped state
    else:
        st.info("AudioLoop service was not running or already stopped.")


# Layout
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Controls")
    if not st.session_state.audioloop_running and not st.session_state.stop_requested:
        if st.button("Start Voice Agent Service"):
            start_audioloop()
    elif st.session_state.audioloop_running:
        if st.button("Stop Voice Agent Service", type="primary"):
            stop_audioloop()
    elif st.session_state.stop_requested:
        # While stopping, show a disabled-like state or a message
        st.button("Processing Stop...", disabled=True)
    
    st.caption("Audio I/O uses browser microphone and speakers.")

    # Add WebRTC streamer
    st.subheader("Microphone Input (Browser)")
    if not st.session_state.audioloop_running:
        st.info("Start Voice Agent Service to enable microphone input.")
        # Disable WebRTC if audioloop is not running to prevent sending to a non-existent queue consumer
        webrtc_ctx = webrtc_streamer(
            key="audioloop-webrtc-input-disabled",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=None, 
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": False, "audio": True},
            sendback_audio=False, 
            desired_playing_state=False
        )
    else:
         webrtc_ctx = webrtc_streamer(
            key="audioloop-webrtc-input-enabled",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=create_webrtc_audio_relay_factory(st.session_state.webrtc_audio_queue),
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": False, "audio": True},
            sendback_audio=False,
            desired_playing_state=st.session_state.audioloop_running
        )
         if webrtc_ctx.state.playing:
            st.success("Microphone is active via browser.")
         else:
            st.warning("Microphone is not currently active or sending data. Click 'START' above if available.")

    st.subheader("Audio Output (Browser)")
    if not st.session_state.audioloop_running:
        st.info("Start Voice Agent Service to enable audio output.")
        # Optional: could render a disabled player cosmetic here if desired
    else:
        webrtc_player_ctx = webrtc_streamer(
            key="audioloop-webrtc-speaker-enabled",
            mode=WebRtcMode.RECVONLY,
            audio_processor_factory=create_audio_player_factory(st.session_state.model_audio_output_queue),
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": False, "audio": True}, # Audio is true, even for RECVONLY to enable playback path
            desired_playing_state=st.session_state.audioloop_running 
        )
        if webrtc_player_ctx.state.playing:
            st.success("Audio output is active.")
        else:
            st.warning("Audio output not active. Check browser or connection.")

with col1:
    st.subheader("Conversation")
    chat_placeholder = st.empty() 

    with st.form(key='text_input_form', clear_on_submit=True):
        user_input = st.text_input("Your message:", key="user_text_box", disabled=not st.session_state.audioloop_running)
        submit_button = st.form_submit_button(label='Send', disabled=not st.session_state.audioloop_running)

    if submit_button and user_input:
        if st.session_state.audioloop_running and st.session_state.user_text_input_queue:
            st.session_state.conversation_history.append(("You", user_input))
            st.session_state.user_text_input_queue.put_nowait(user_input)
            st.rerun() # Ensure UI updates to show user's message
        elif not st.session_state.audioloop_running:
            st.error("Voice Agent not running. Please start it.")

# Display conversation history and check for new model messages
conversation_html = ""
for speaker, text in st.session_state.conversation_history:
    text_cleaned = text.replace("\n", "<br>") # Ensure newlines are rendered
    if speaker == "You":
        conversation_html += f"<div style='text-align: right; background-color: #DCF8C6; color: black; padding: 8px; border-radius: 8px; margin-left: 20%; margin-bottom: 5px; margin-top: 5px;'><b>You:</b> {text_cleaned}</div>"
    else: # Model
        conversation_html += f"<div style='text-align: left; background-color: #E9E9EB; color: black; padding: 8px; border-radius: 8px; margin-right: 20%; margin-bottom: 5px; margin-top: 5px;'><b>Gemini:</b> {text_cleaned}</div>"
chat_placeholder.markdown(conversation_html, unsafe_allow_html=True)


# Polling for model responses
if st.session_state.audioloop_running:
    model_message_received = False
    try:
        while st.session_state.model_text_output_queue and not st.session_state.model_text_output_queue.empty():
            model_response = st.session_state.model_text_output_queue.get_nowait()
            if model_response is None: # Sentinel from AudioLoop's finally block
                print("Streamlit: Received None from model_text_output_queue, likely AudioLoop shutdown.")
                # Potentially trigger a final UI update or state change if needed
                if st.session_state.audioloop_running: # If still marked as running, process stop
                    # This could be a race condition if stop_audioloop hasn't fully completed
                    # stop_audioloop() # Careful with calling this again here
                    pass 
                break 
            st.session_state.conversation_history.append(("Gemini", model_response))
            print(f"Streamlit UI: Added Gemini response to history: '{model_response[:100]}...'")
            model_message_received = True
    except asyncio.QueueEmpty: 
        pass
    except Exception as e: 
        print(f"Streamlit: Error getting from model_text_output_queue: {e}")
    
    if model_message_received:
        st.rerun()

# Sidebar status
if st.session_state.stop_requested and st.session_state.audioloop_running:
    # This state means stop was clicked, but thread might not have joined yet
    st.sidebar.warning("Voice Agent is STOPPING...") 
elif st.session_state.audioloop_running:
    st.sidebar.success("Voice Agent is RUNNING")
else:
    st.sidebar.error("Voice Agent is STOPPED")
    # No longer need to reset stop_requested here, it's done in stop_audioloop()

# Add some footer or status
if st.session_state.audioloop_running:
    st.sidebar.success("Voice Agent is RUNNING")
elif st.session_state.stop_requested:
    st.sidebar.warning("Voice Agent is STOPPING...")
else:
    st.sidebar.error("Voice Agent is STOPPED")
# This is to try and force a refresh to check the queue periodically
# if st.session_state.audioloop_running:
#    time.sleep(0.5) # Be very careful with time.sleep in main Streamlit thread
#    st.rerun() # This will cause a continuous loop, not ideal. Better to rely on interaction-driven reruns and queue checks. 
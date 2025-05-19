import streamlit as st
import asyncio
import threading
from queue import Queue as ThreadQueue
from google_ai_sample import AudioLoop, RECEIVE_SAMPLE_RATE, CHANNELS as GEMINI_OUTPUT_CHANNELS # Import rate and channels for Gemini output
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, RTCConfiguration
import numpy as np
import librosa
import av # For av.AudioFrame
from aiortc.mediastreams import AudioStreamTrack # Import AudioStreamTrack
import time # For pts
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
    st.session_state.generated_audio_track = None # To store the track instance

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

class GeneratedAudioTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(self, model_audio_q: asyncio.Queue, target_sample_rate: int, target_channels: int):
        super().__init__()  # Initialize base class
        print(f"!!! GeneratedAudioTrack __init__ CALLED. Instance ID: {id(self)} !!!")
        self.model_audio_q = model_audio_q # This will be used later for Gemini's audio
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        
        # For tone generation (test)
        self.samples_per_frame = int(0.02 * self.target_sample_rate) # 20ms worth of samples
        self.phase = 0
        self.frequency = 440  # Hz (A4 note)
        self._start_time = time.time() # For pts calculation
        self.last_pts = 0

        print(f"GeneratedAudioTrack initialized for TONE TEST. Target SR: {self.target_sample_rate}, Channels: {self.target_channels}, Samples/Frame: {self.samples_per_frame}")

    async def recv(self):
        print(f"!!! GeneratedAudioTrack: recv() method ENTERED (TONE TEST) !!! Instance ID: {id(self)}")
        
        # --- TONE GENERATION ---
        # First, try to get actual model audio
        try:
            model_audio_bytes = self.model_audio_q.get_nowait()
            # print(f"GeneratedAudioTrack: Got {len(model_audio_bytes)} bytes from model_audio_q.")
            
            # Ensure it's a multiple of frame size (samples_per_frame * channels * 2 bytes/sample)
            # This part needs to be robust. For now, let's assume model_audio_bytes is correctly formatted.
            # If not, we might need buffering here.
            
            num_expected_bytes_per_frame = self.samples_per_frame * self.target_channels * 2 # 2 bytes for int16
            if len(model_audio_bytes) < num_expected_bytes_per_frame:
                # print(f"GeneratedAudioTrack: Model audio chunk {len(model_audio_bytes)} bytes is smaller than frame size {num_expected_bytes_per_frame}. Padding with silence for now.")
                # Simplistic padding - better to buffer and combine.
                padding_bytes = b'\\x00' * (num_expected_bytes_per_frame - len(model_audio_bytes))
                model_audio_bytes += padding_bytes
            elif len(model_audio_bytes) > num_expected_bytes_per_frame:
                # print(f"GeneratedAudioTrack: Model audio chunk {len(model_audio_bytes)} bytes is larger than frame size {num_expected_bytes_per_frame}. Truncating for now.")
                model_audio_bytes = model_audio_bytes[:num_expected_bytes_per_frame]


            audio_data_np = np.frombuffer(model_audio_bytes, dtype=np.int16)
            # print(f"GeneratedAudioTrack: Converted model audio bytes to np array shape: {audio_data_np.shape}")

            if self.target_channels == 1:
                # Expected shape for from_ndarray (mono): (1, num_samples)
                frame_data = audio_data_np.reshape(1, -1)
                layout = 'mono'
            elif self.target_channels == 2:
                # Expected shape for from_ndarray (stereo): (num_samples, 2) for packed, or (2, num_samples) for planar
                # Let's assume packed stereo: (samples_per_frame, 2)
                # If Gemini gives interleaved stereo, it would be (samples_per_frame * 2,). Reshape to (samples_per_frame, 2)
                frame_data = audio_data_np.reshape(-1, 2) # This assumes audio_data_np has samples_per_frame * 2 elements
                layout = 'stereo'
            else: # Fallback to mono
                frame_data = audio_data_np.reshape(1, -1)
                layout = 'mono'
            
            frame = av.AudioFrame.from_ndarray(frame_data, format='s16', layout=layout)
            # print(f"GeneratedAudioTrack: Created frame from MODEL audio. Samples: {frame.samples}")
            self.model_audio_q.task_done()

        except asyncio.QueueEmpty:
            # print("GeneratedAudioTrack: model_audio_q is empty. Generating TONE.") # Can be verbose
            # --- FALLBACK TO TONE GENERATION if queue is empty ---
            t = (np.arange(self.samples_per_frame) + self.phase) / self.target_sample_rate
            tone_data = (0.3 * np.sin(2 * np.pi * self.frequency * t) * 32767).astype(np.int16)
            self.phase += self.samples_per_frame

            if self.target_channels == 1:
                frame = av.AudioFrame.from_ndarray(tone_data.reshape(1, -1), format='s16', layout='mono')
            elif self.target_channels == 2:
                stereo_tone_data = np.vstack([tone_data, tone_data]).T
                frame = av.AudioFrame.from_ndarray(stereo_tone_data, format='s16', layout='stereo')
            else:
                frame = av.AudioFrame.from_ndarray(tone_data.reshape(1, -1), format='s16', layout='mono')
            # print(f"GeneratedAudioTrack: Created frame from TONE. Samples: {frame.samples}")


        pts = time.time() - self._start_time
        if pts <= self.last_pts:
            pts = self.last_pts + 0.001 
        self.last_pts = pts
        frame.pts = pts 
        frame.sample_rate = self.target_sample_rate

        await asyncio.sleep(0.018) # Simulate real-time generation, slightly less than 20ms
        return frame
        # --- END TONE GENERATION ---
        
        # Placeholder for future: Get audio from self.model_audio_q
        # For now, always generate tone. If queue is empty, could generate silence.


# Factory creator for WebRTCAudioRelay
def create_webrtc_audio_relay_factory(queue_instance: asyncio.Queue):
    def factory():
        return WebRTCAudioRelay(output_queue=queue_instance)
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

        # Create the GeneratedAudioTrack instance here
        st.session_state.generated_audio_track = GeneratedAudioTrack(
            model_audio_q=st.session_state.model_audio_output_queue,
            target_sample_rate=RECEIVE_SAMPLE_RATE, # Gemini's output rate
            target_channels=GEMINI_OUTPUT_CHANNELS  # Gemini's output channels
        )
        print(f"Streamlit: Created GeneratedAudioTrack instance: {id(st.session_state.generated_audio_track)}")


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
        st.session_state.generated_audio_track = None # Clear the track instance
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
        if st.session_state.generated_audio_track:
            print(f"Streamlit: Passing GeneratedAudioTrack instance {id(st.session_state.generated_audio_track)} to webrtc_streamer for speaker.")
            webrtc_player_ctx = webrtc_streamer(
                key="audioloop-webrtc-speaker-enabled",
                mode=WebRtcMode.RECVONLY,
                source_audio_track=st.session_state.generated_audio_track, # USE THIS
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": False, "audio": True}, 
                desired_playing_state=st.session_state.audioloop_running 
            )
            if webrtc_player_ctx.state.playing:
                st.success("Audio output is active.")
            else:
                st.warning("Audio output not active. Check browser or connection.")
        else:
            st.error("GeneratedAudioTrack not initialized. Cannot start speaker.")

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
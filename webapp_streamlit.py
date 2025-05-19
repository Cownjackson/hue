import streamlit as st
import asyncio
import threading
from queue import Queue as ThreadQueue
from google_ai_sample import AudioLoop 
from google.genai import types as genai_types # For creating audio parts

# Imports for WebRTC Audio Processing
from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer
from streamlit.runtime.scriptrunner import add_script_run_ctx # Import for context
import av # PyAV for audio frame manipulation
import numpy as np
from pydub import AudioSegment
import io
import time

# --- WebRTC Configuration & Constants ---
# Input to Gemini
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH_BYTES = 2 # 16-bit PCM

# Output from Gemini (for playback) - From google_ai_sample.py
PLAYBACK_SAMPLE_RATE = 24000
PLAYBACK_CHANNELS = 1
PLAYBACK_SAMPLE_WIDTH_BYTES = 2 # 16-bit PCM

class WebRtcAudioProcessor(AudioProcessorBase):
    def __init__(self, audioloop_out_q, target_sample_rate, target_channels, target_sample_width_bytes, audioloop_event_loop, processor_stop_event):
        self.audioloop_out_q = audioloop_out_q
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.target_sample_width_bytes = target_sample_width_bytes
        self.audioloop_event_loop = audioloop_event_loop
        self.processor_stop_event = processor_stop_event
        self.active = True
        self.frames_processed_count = 0
        self.capture_single_turn = False
        self.frames_for_single_turn = 0
        self.max_frames_for_single_turn = 50
        print("WebRtcAudioProcessor initialized.")

    def _process_audio_data(self, data: np.ndarray, original_sample_rate: int, original_channels: int, original_sample_width: int) -> bytes | None:
        try:
            sound = AudioSegment(
                data=data.tobytes(),
                sample_width=original_sample_width,
                frame_rate=original_sample_rate,
                channels=original_channels
            )
            if sound.frame_rate != self.target_sample_rate:
                sound = sound.set_frame_rate(self.target_sample_rate)
            if sound.channels != self.target_channels:
                sound = sound.set_channels(self.target_channels)
            if sound.sample_width != self.target_sample_width_bytes:
                sound = sound.set_sample_width(self.target_sample_width_bytes)
            return sound.raw_data
        except Exception as e:
            print(f"WebRtcAudioProcessor: Error in _process_audio_data: {e}")
            return None

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        if not self.active or (self.processor_stop_event and self.processor_stop_event.is_set()):
            if self.active:
                print("WebRtcAudioProcessor: recv called while inactive or stop event set. Stopping.")
                self.active = False
            return frame

        if self.capture_single_turn:
            if self.frames_for_single_turn >= self.max_frames_for_single_turn:
                if self.active:
                    print(f"WebRtcAudioProcessor: Single turn capture reached max frames ({self.max_frames_for_single_turn}). Signaling end.")
                    self.active = False
                    self.capture_single_turn = False
                    self.frames_for_single_turn = 0
                    if self.audioloop_out_q and self.audioloop_event_loop and self.audioloop_event_loop.is_running():
                        print("WebRtcAudioProcessor (single_turn): Queuing None to AudioLoop out_queue.")
                        try:
                            self.audioloop_event_loop.call_soon_threadsafe(self.audioloop_out_q.put_nowait, None)
                        except Exception as e: print(f"WebRtcAudioProcessor (single_turn): Error queuing None: {e}")
                return frame

        try:
            raw_samples_float32 = frame.to_ndarray()
            if raw_samples_float32.ndim == 1:
                raw_samples_float32 = np.expand_dims(raw_samples_float32, axis=0)
            raw_samples_int16 = (raw_samples_float32 * 32767).astype(np.int16)
            data_to_process = raw_samples_int16[0, :]
            
            processed_audio_bytes = self._process_audio_data(
                data_to_process,
                original_sample_rate=frame.sample_rate,
                original_channels=1,
                original_sample_width=raw_samples_int16.dtype.itemsize
            )

            if processed_audio_bytes and self.audioloop_out_q and self.audioloop_event_loop and self.audioloop_event_loop.is_running() and self.active:
                gemini_audio_part = genai_types.Part(inline_data=genai_types.Blob(data=processed_audio_bytes, mime_type="audio/pcm"))
                try:
                    self.audioloop_event_loop.call_soon_threadsafe(self.audioloop_out_q.put_nowait, gemini_audio_part)
                    self.frames_processed_count += 1
                    if self.capture_single_turn:
                        self.frames_for_single_turn +=1
                        print(f"WebRtcAudioProcessor (single_turn): Frame {self.frames_for_single_turn}/{self.max_frames_for_single_turn} queued.")
                    elif self.frames_processed_count % 100 == 0: 
                        print(f"WebRtcAudioProcessor: Queued {self.frames_processed_count} audio frames to AudioLoop.")
                except asyncio.QueueFull:
                    print("WebRtcAudioProcessor: AudioLoop.out_queue is full. Dropping input frame.")
                except RuntimeError as e:
                    if "Event loop is closed" in str(e):
                        print("WebRtcAudioProcessor: Event loop closed, cannot queue audio frame.")
                        self.active = False
                    else: raise
                except Exception as e:
                    print(f"WebRtcAudioProcessor: Error putting to AudioLoop.out_queue: {e}")
            elif not (self.audioloop_event_loop and self.audioloop_event_loop.is_running()) and self.active:
                print("WebRtcAudioProcessor: audioloop_event_loop is not running. Cannot queue audio.")
                self.active = False
        except Exception as e:
            print(f"WebRtcAudioProcessor: Error in recv: {e}")
        return frame

    def on_ended(self):
        print("WebRtcAudioProcessor: on_ended called (input stream stopped from client/browser side).")
        if self.active:
            self.active = False
            if self.audioloop_out_q and self.audioloop_event_loop and self.audioloop_event_loop.is_running():
                print("WebRtcAudioProcessor (on_ended): Queuing None to AudioLoop out_queue.")
                try:
                    self.audioloop_event_loop.call_soon_threadsafe(self.audioloop_out_q.put_nowait, None)
                except asyncio.QueueFull:
                    print("WebRtcAudioProcessor (on_ended): AudioLoop.out_queue full, couldn't queue None sentinel.")
                except RuntimeError as e:
                    if "Event loop is closed" in str(e): print("WebRtcAudioProcessor (on_ended): Event loop closed, cannot queue None.")
                    else: raise
                except Exception as e: print(f"WebRtcAudioProcessor (on_ended): Error queuing None: {e}")
            elif self.audioloop_out_q:
                print("WebRtcAudioProcessor (on_ended): Event loop not running. Cannot queue None.")

# --- WebRTC Audio Playback Processor ---
class AudioPlaybackProcessor(AudioProcessorBase):
    def __init__(self, audioloop_in_q, playback_sample_rate, playback_channels, playback_sample_width_bytes, audioloop_event_loop, processor_stop_event):
        self.audioloop_in_q = audioloop_in_q
        self.sample_rate = playback_sample_rate
        self.channels = playback_channels
        self.sample_width_bytes = playback_sample_width_bytes
        self.audioloop_event_loop = audioloop_event_loop
        self.processor_stop_event = processor_stop_event
        self.active = True
        self.audio_buffer = bytearray()
        self.bytes_per_frame = int(self.sample_rate * self.channels * self.sample_width_bytes * 0.020) # 20ms frame duration
        self.frames_sent_count = 0
        print(f"AudioPlaybackProcessor initialized. Expecting {self.bytes_per_frame} bytes per 20ms output frame.")

    async def _get_audio_data_from_queue(self):
        if not self.active or not self.audioloop_in_q or not self.audioloop_event_loop or not self.audioloop_event_loop.is_running():
            return None
        try:
            data = await asyncio.wait_for(self.audioloop_in_q.get(), timeout=0.005) # Short timeout
            if data is None:
                print("AudioPlaybackProcessor: Received None (sentinel) from AudioLoop.audio_in_queue.")
                self.active = False
            return data
        except asyncio.TimeoutError:
            return None # No data currently
        except Exception as e:
            print(f"AudioPlaybackProcessor: Error getting data from audio_in_queue: {e}")
            self.active = False
            return None

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame: # frame is a dummy from webrtc for RECVONLY mode
        if not self.active or (self.processor_stop_event and self.processor_stop_event.is_set()):
            if self.active:
                print("AudioPlaybackProcessor: recv called while inactive or stop event set.")
                self.active = False
            return av.AudioFrame(format='s16', layout='mono' if self.channels == 1 else 'stereo', samples=self.bytes_per_frame // self.sample_width_bytes, sample_rate=self.sample_rate) # Send silence

        # Accumulate data
        while len(self.audio_buffer) < self.bytes_per_frame and self.active:
            if not self.audioloop_event_loop or not self.audioloop_event_loop.is_running():
                self.active = False; break
            
            future = asyncio.run_coroutine_threadsafe(self._get_audio_data_from_queue(), self.audioloop_event_loop)
            try:
                chunk = future.result(timeout=0.010) # Wait a bit for data
                if chunk: self.audio_buffer.extend(chunk)
                elif not self.active: break # Sentinel received
                else: break # No data this cycle
            except asyncio.TimeoutError: break
            except Exception as e:
                print(f"AudioPlaybackProcessor: Error in recv getting chunk: {e}"); self.active = False; break
        
        if len(self.audio_buffer) >= self.bytes_per_frame:
            frame_bytes = self.audio_buffer[:self.bytes_per_frame]
            del self.audio_buffer[:self.bytes_per_frame]
            samples_int16 = np.frombuffer(frame_bytes, dtype=np.int16)
            new_frame = av.AudioFrame.from_ndarray(samples_int16, format='s16', layout='mono', rate=self.sample_rate)
            self.frames_sent_count += 1
            if self.frames_sent_count % 100 == 0: # Log less frequently
                 print(f"AudioPlaybackProcessor: Sent {self.frames_sent_count} audio frames to browser.")
            return new_frame
        else: # Not enough data, send silence
            return av.AudioFrame(format='s16', layout='mono' if self.channels == 1 else 'stereo', samples=self.bytes_per_frame // self.sample_width_bytes, sample_rate=self.sample_rate)

    def on_ended(self):
        print("AudioPlaybackProcessor: on_ended called (output stream stopped from client/browser side).")
        self.active = False

class WebRtcAudioProcessorFactory:
    def __init__(self, audioloop_instance, audioloop_event_loop, processor_stop_event):
        self.audioloop_instance = audioloop_instance
        self.audioloop_event_loop = audioloop_event_loop
        self.processor_stop_event = processor_stop_event
        self.current_processor = None
        print("WebRtcAudioProcessorFactory initialized with direct object references.")

    def __call__(self):
        if not self.audioloop_instance or not hasattr(self.audioloop_instance, 'out_queue'):
            print("WebRtcAudioProcessorFactory: AudioLoop instance or its out_queue is invalid!")
            raise RuntimeError("AudioLoop instance not ready for WebRTC processor.")
        if not self.audioloop_event_loop:
            print("WebRtcAudioProcessorFactory: AudioLoop event loop is invalid!")
            raise RuntimeError("AudioLoop event loop not ready for WebRTC processor.")
        if not self.processor_stop_event:
            print("WebRtcAudioProcessorFactory: Processor stop event is invalid!")
            raise RuntimeError("Processor stop event not ready for WebRTC processor.")

        print("WebRtcAudioProcessorFactory: Creating new WebRtcAudioProcessor instance using direct references.")
        self.current_processor = WebRtcAudioProcessor(
            audioloop_out_q=self.audioloop_instance.out_queue,
            target_sample_rate=TARGET_SAMPLE_RATE,
            target_channels=TARGET_CHANNELS,
            target_sample_width_bytes=TARGET_SAMPLE_WIDTH_BYTES,
            audioloop_event_loop=self.audioloop_event_loop,
            processor_stop_event=self.processor_stop_event
        )
        return self.current_processor

    def start_single_turn_capture(self):
        if self.current_processor:
            print("WebRtcAudioProcessorFactory: Requesting single turn capture from current processor.")
            self.current_processor.capture_single_turn = True
            self.current_processor.frames_for_single_turn = 0
            self.current_processor.active = True
        else:
            print("WebRtcAudioProcessorFactory: No current processor to start single turn capture.")

class AudioPlaybackProcessorFactory:
    def __init__(self, audioloop_instance, audioloop_event_loop, processor_stop_event):
        self.audioloop_instance = audioloop_instance
        self.audioloop_event_loop = audioloop_event_loop
        self.processor_stop_event = processor_stop_event
        print("AudioPlaybackProcessorFactory initialized.")

    def __call__(self):
        if not self.audioloop_instance or not hasattr(self.audioloop_instance, 'audio_in_queue'):
            raise RuntimeError("AudioLoop instance or audio_in_queue invalid for PlaybackProcessor.")
        if not self.audioloop_event_loop or not self.processor_stop_event:
            raise RuntimeError("Event loop or stop event not ready for PlaybackProcessor.")
        
        print("AudioPlaybackProcessorFactory: Creating new AudioPlaybackProcessor.")
        return AudioPlaybackProcessor(
            audioloop_in_q=self.audioloop_instance.audio_in_queue,
            playback_sample_rate=PLAYBACK_SAMPLE_RATE,
            playback_channels=PLAYBACK_CHANNELS,
            playback_sample_width_bytes=PLAYBACK_SAMPLE_WIDTH_BYTES,
            audioloop_event_loop=self.audioloop_event_loop,
            processor_stop_event=self.processor_stop_event
        )

# --- End WebRTC Audio Processor ---

def run_audioloop_in_thread(audioloop_instance, streamlit_user_q, streamlit_model_q):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Store loop and a stop event for WebRTC processor to use
    st.session_state.audioloop_event_loop = loop
    st.session_state.webrtc_processor_stop_event = threading.Event()
    time.sleep(0.1) # Small delay to help ensure session_state write is visible
    print("Streamlit BG THREAD: Stored AudioLoop event loop and WebRTC processor stop event in session_state.")
    print(f"Streamlit BG THREAD: audioloop_event_loop value: {st.session_state.get('audioloop_event_loop')}")
    print(f"Streamlit BG THREAD: webrtc_processor_stop_event value: {st.session_state.get('webrtc_processor_stop_event')}")

    audioloop_instance.user_text_input_queue = streamlit_user_q
    audioloop_instance.model_text_output_queue = streamlit_model_q
    
    print("Streamlit BG THREAD: Starting AudioLoop.run()...")
    try:
        loop.run_until_complete(audioloop_instance.run())
    except Exception as e:
        print(f"Exception in AudioLoop thread: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("AudioLoop thread finished its Python execution.")
        # Signal the WebRTC processor that this loop is stopping
        if "webrtc_processor_stop_event" in st.session_state and st.session_state.webrtc_processor_stop_event:
            print("Streamlit: Signaling WebRTC processor to stop.")
            st.session_state.webrtc_processor_stop_event.set()
        
        loop.close()
        # Clean up session state related to this thread
        if "audioloop_event_loop" in st.session_state:
            del st.session_state.audioloop_event_loop
            print("Streamlit: Cleared AudioLoop event loop from session_state.")
        if "webrtc_processor_stop_event" in st.session_state:
            del st.session_state.webrtc_processor_stop_event
            print("Streamlit: Cleared WebRTC processor stop event from session_state.")


st.set_page_config(layout="wide")
st.title("Gemini Live Voice Agent - Text & Voice Interface")

if 'audioloop_running' not in st.session_state:
    st.session_state.audioloop_running = False
    st.session_state.audioloop_instance = None
    st.session_state.audioloop_thread = None
    st.session_state.user_text_input_queue = asyncio.Queue()
    st.session_state.model_text_output_queue = asyncio.Queue()
    st.session_state.conversation_history = []
    st.session_state.stop_requested = False
    st.session_state.webrtc_input_processor_factory = None

def start_audioloop():
    if not st.session_state.audioloop_running:
        st.session_state.user_text_input_queue = asyncio.Queue()
        st.session_state.model_text_output_queue = asyncio.Queue()
        st.session_state.conversation_history = []

        # Instantiate AudioLoop with use_pyaudio=False as WebRTC will handle audio input
        st.session_state.audioloop_instance = AudioLoop(
            user_text_input_queue=st.session_state.user_text_input_queue,
            model_text_output_queue=st.session_state.model_text_output_queue,
            use_pyaudio=False # IMPORTANT: Disable PyAudio handling in AudioLoop
        )
        
        # The factory is now created dynamically when prerequisites are met.
        # st.session_state.webrtc_audio_processor_factory = WebRtcAudioProcessorFactory(...)
        # print("Streamlit: WebRtcAudioProcessorFactory created and stored in session_state.")

        st.session_state.audioloop_thread = threading.Thread(
            target=run_audioloop_in_thread,
            args=(
                st.session_state.audioloop_instance,
                st.session_state.user_text_input_queue,
                st.session_state.model_text_output_queue
            ),
            daemon=True
        )
        add_script_run_ctx(st.session_state.audioloop_thread) # Add context before starting
        st.session_state.audioloop_thread.start()
        st.session_state.audioloop_running = True
        st.session_state.stop_requested = False
        st.success("AudioLoop service started! (WebRTC audio input mode)")
        print("Streamlit: AudioLoop thread started (use_pyaudio=False).")
        st.rerun()

def stop_audioloop():
    if st.session_state.audioloop_thread and st.session_state.audioloop_instance:
        print("Streamlit: Requesting AudioLoop stop...")
        st.session_state.stop_requested = True

        # Signal WebRTC processor to stop by setting its event (if it exists and is listening)
        # The AudioLoop thread's finally block also does this.
        if "webrtc_processor_stop_event" in st.session_state and st.session_state.webrtc_processor_stop_event:
            print("Streamlit: Signaling WebRTC processor to stop via its dedicated event during service stop.")
            st.session_state.webrtc_processor_stop_event.set()

        if st.session_state.user_text_input_queue:
            # This signals send_text_from_queue to stop
            st.session_state.user_text_input_queue.put_nowait(None) 
        
        # The WebRTC audio processor's on_ended (if client disconnects) 
        # or its recv loop (if processor_stop_event is set)
        # should put None on audioloop_instance.out_queue to stop send_realtime.

        thread_to_join = st.session_state.audioloop_thread
        if thread_to_join.is_alive():
            print(f"Streamlit: Waiting for AudioLoop thread ({thread_to_join.name}) to join...")
            thread_to_join.join(timeout=7.0) # Increased timeout slightly
            if thread_to_join.is_alive():
                print("Streamlit: WARNING - AudioLoop thread did not join within timeout.")
            else:
                print("Streamlit: AudioLoop thread joined successfully.")
        else:
            print("Streamlit: AudioLoop thread was already finished.")

        st.session_state.audioloop_running = False
        st.session_state.audioloop_instance = None 
        st.session_state.audioloop_thread = None
        # st.session_state.webrtc_audio_processor_factory = None # No longer stored globally this way
        # Event loop and stop event are cleared by run_audioloop_in_thread's finally block.
        
        st.session_state.stop_requested = False 
        st.warning("AudioLoop service stopped.")
        print("Streamlit: AudioLoop service stop process completed in Streamlit UI.")
        st.rerun()
    else:
        st.info("AudioLoop service was not running or already stopped.")

# --- UI Layout ---
col1, col2 = st.columns([3, 1]) # Text input and conversation on left, controls on right

with col2: # Controls Column
    st.subheader("Controls")
    if not st.session_state.audioloop_running and not st.session_state.stop_requested:
        if st.button("Start Voice Agent Service"):
            start_audioloop()
    elif st.session_state.audioloop_running:
        if st.button("Stop Voice Agent Service", type="primary"):
            stop_audioloop()
        
        if st.button("Send Single Audio Burst & Finish Turn"):
            if st.session_state.get('webrtc_input_processor_factory') and \
               hasattr(st.session_state.webrtc_input_processor_factory, 'start_single_turn_capture'):
                st.session_state.webrtc_input_processor_factory.start_single_turn_capture()
                st.info("Attempting to send a short audio burst...")
            else:
                st.warning("WebRTC input processor not ready for single burst yet.")

    elif st.session_state.stop_requested:
        st.button("Processing Stop...", disabled=True)
    
    st.caption("Text I/O below. Microphone via browser if service is running.")

    # WebRTC Microphone Input - only show if service is running and components are ready
    if st.session_state.audioloop_running:
        # Detailed prerequisite checks for debugging
        s = st.session_state
        event_loop_ok = "audioloop_event_loop" in s and s.get("audioloop_event_loop") is not None
        stop_event_ok = "webrtc_processor_stop_event" in s and s.get("webrtc_processor_stop_event") is not None
        instance_ok = s.audioloop_instance is not None
        out_queue_ok = instance_ok and hasattr(s.audioloop_instance, 'out_queue') and s.audioloop_instance.out_queue is not None
        # factory_ok is no longer checked here as it's created on demand

        are_webrtc_prerequisites_met = (
            event_loop_ok and stop_event_ok and instance_ok and out_queue_ok
        )
        current_poll_attempt = s.get('audioloop_init_refresh_count', 0) + 1
        # Print details on every poll attempt for clarity during debugging
        print(f"DEBUG MAIN THREAD: Poll Attempt {current_poll_attempt} for WebRTC Prerequisites:")
        print(f"  - audioloop_event_loop exists: {'audioloop_event_loop' in s}, value: {s.get('audioloop_event_loop')}")
        print(f"  - webrtc_processor_stop_event exists: {'webrtc_processor_stop_event' in s}, value: {s.get('webrtc_processor_stop_event')}")
        print(f"  - audioloop_instance exists and not None: {instance_ok}")
        print(f"  - audioloop_instance.out_queue exists and not None: {out_queue_ok}")
        # print(f"  - webrtc_audio_processor_factory exists and not None: {factory_ok}") # No longer checking factory_ok here
        print(f"  Overall met: {are_webrtc_prerequisites_met}")

        if are_webrtc_prerequisites_met:
            # Prerequisites met, clear any refresh counter and show WebRTC
            if "audioloop_init_refresh_count" in s:
                del s.audioloop_init_refresh_count
                print("DEBUG: WebRTC prerequisites met, cleared refresh counter.")

            # Create the factory now that all components are confirmed to be available
            current_audioloop_instance = s.audioloop_instance
            current_event_loop = s.audioloop_event_loop
            current_stop_event = s.webrtc_processor_stop_event
            
            actual_factory_to_use = WebRtcAudioProcessorFactory(
                audioloop_instance=current_audioloop_instance,
                audioloop_event_loop=current_event_loop,
                processor_stop_event=current_stop_event
            )

            st.markdown("---") # Separator
            st.subheader("Microphone (Browser)")
            try:
                webrtc_ctx = webrtc_streamer(
                    key="audioloop_mic_input", # Unique key
                    mode=WebRtcMode.SENDONLY, # Send audio from browser to server
                    audio_processor_factory=actual_factory_to_use, # Pass the primed factory
                    media_stream_constraints={"video": False, "audio": True},
                    # Desired audio constraints for input:
                    # desired_playing_state=True, # Try to start automatically
                    # rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]} # Optional
                )
                if webrtc_ctx.state.playing:
                    st.success("🎤 Microphone is active and sending audio.")
                    # Now, add the Playback WebRTC component if all is well
                    st.markdown("---")
                    st.subheader("Audio Output (Gemini)")
                    try:
                        # Reuse the same prerequisites check for simplicity, 
                        # as playback also depends on the audioloop_instance and its event loop.
                        # The factory will use audioloop_instance.audio_in_queue
                        playback_processor_factory = AudioPlaybackProcessorFactory(
                            audioloop_instance=s.audioloop_instance,
                            audioloop_event_loop=s.audioloop_event_loop,
                            processor_stop_event=s.webrtc_processor_stop_event
                        )
                        webrtc_streamer(
                            key="audioloop_speaker_output",
                            mode=WebRtcMode.RECVONLY, # We send audio to the client
                            audio_processor_factory=playback_processor_factory,
                            media_stream_constraints={"video": False, "audio": False}, # We are sending audio, client receives
                            # desired_playing_state=True # Optional: try to start playing immediately
                        )
                        st.info("🔊 Audio output stream active. Waiting for Gemini's voice...")
                    except Exception as e:
                        st.error(f"Failed to initialize WebRTC audio output: {e}")
                        # import traceback; traceback.print_exc()
                else:
                    st.info("🎤 Microphone is idle. Click 'Start' in the component above if available, or check browser permissions.")
            except Exception as e:
                st.error(f"Failed to initialize WebRTC microphone: {e}")
                # import traceback; traceback.print_exc()
        else:
            # Prerequisites not met, show initializing message and attempt refresh
            st.info("AudioLoop components initializing, WebRTC microphone will be available shortly...")

            if "audioloop_init_refresh_count" not in s:
                s.audioloop_init_refresh_count = 0
            
            if s.audioloop_init_refresh_count < 5: # Still limit to 5 attempts total
                s.audioloop_init_refresh_count += 1
                time.sleep(0.25) 
                # The print for poll attempt is now at the start of the check block
                st.rerun()
            # else: After 5 attempts, it will just display the "initializing..." message without further reruns.
            # Consider adding a message here if it consistently fails to initialize after N retries.

    # Clean up refresh counter if audioloop is stopped (also covers successful init then stop)
    if not st.session_state.audioloop_running:
        if "audioloop_init_refresh_count" in st.session_state: # Use st.session_state directly
            del st.session_state.audioloop_init_refresh_count    # Use st.session_state directly
            print("DEBUG: AudioLoop not running, cleared refresh counter.")


with col1: # Main Conversation and Text Input Column
    st.subheader("Conversation")
    chat_placeholder = st.empty() 

    with st.form(key='text_input_form', clear_on_submit=True):
        user_input = st.text_input("Your message (text input):", key="user_text_box", disabled=not st.session_state.audioloop_running)
        submit_button = st.form_submit_button(label='Send Text', disabled=not st.session_state.audioloop_running)

    if submit_button and user_input:
        if st.session_state.audioloop_running and st.session_state.user_text_input_queue:
            st.session_state.conversation_history.append(("You (text)", user_input))
            st.session_state.user_text_input_queue.put_nowait(user_input)
        elif not st.session_state.audioloop_running:
            st.error("Voice Agent not running. Please start it.")

# Display conversation history and check for new model messages
conversation_html = ""
for speaker, text in st.session_state.conversation_history:
    text_cleaned = text.replace("\n", "<br>")
    if speaker.startswith("You"):
        conversation_html += f"<div style='text-align: right; background-color: #DCF8C6; color: black; padding: 8px; border-radius: 8px; margin-left: 20%; margin-bottom: 5px; margin-top: 5px;'><b>{speaker}:</b> {text_cleaned}</div>"
    else: 
        conversation_html += f"<div style='text-align: left; background-color: #E9E9EB; color: black; padding: 8px; border-radius: 8px; margin-right: 20%; margin-bottom: 5px; margin-top: 5px;'><b>{speaker}:</b> {text_cleaned}</div>"
chat_placeholder.markdown(conversation_html, unsafe_allow_html=True)

if st.session_state.audioloop_running:
    model_message_received = False
    try:
        while st.session_state.model_text_output_queue and not st.session_state.model_text_output_queue.empty():
            model_response = st.session_state.model_text_output_queue.get_nowait()
            if model_response is None: 
                print("Streamlit: Received None from model_text_output_queue (AudioLoop shutdown).")
                break 
            st.session_state.conversation_history.append(("Gemini (voice output - text part)", model_response)) # Label clearly for now
            model_message_received = True
    except asyncio.QueueEmpty: 
        pass
    except Exception as e: 
        print(f"Streamlit: Error getting from model_text_output_queue: {e}")
    
    if model_message_received:
        st.rerun()

# Sidebar status
st.sidebar.title("Status")
if st.session_state.stop_requested and st.session_state.audioloop_running:
    st.sidebar.warning("Voice Agent is STOPPING...") 
elif st.session_state.audioloop_running:
    st.sidebar.success("Voice Agent is RUNNING")
    if "audioloop_event_loop" not in st.session_state: # Check if the specific prerequisite is missing
        st.sidebar.info("Initializing audio components...")
else:
    st.sidebar.error("Voice Agent is STOPPED")

# Periodic refresh to check queues, but be cautious
# This is usually handled by interactions or callbacks in Streamlit
# If st.session_state.audioloop_running:
#    time.sleep(0.2) # Avoid aggressive sleeping
#    st.rerun()

# Add some footer or status
# No, this is redundant with the above sidebar status and the refresh counter cleanup logic
# if st.session_state.audioloop_running:
#    st.sidebar.success("Voice Agent is RUNNING")
# elif st.session_state.stop_requested:
#    st.sidebar.warning("Voice Agent is STOPPING...")
# else:
#    st.sidebar.error("Voice Agent is STOPPED")

# This is to try and force a refresh to check the queue periodically
# if st.session_state.audioloop_running:
#    time.sleep(0.5) # Be very careful with time.sleep in main Streamlit thread
#    st.rerun() # This will cause a continuous loop, not ideal. Better to rely on interaction-driven reruns and queue checks. 
import streamlit as st
import asyncio
import threading
from queue import Queue as ThreadQueue # Retained for compatibility, though asyncio.Queue is mostly used
from google_ai_sample import AudioLoop, RECEIVE_SAMPLE_RATE, CHANNELS as GEMINI_OUTPUT_CHANNELS
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, RTCConfiguration
import numpy as np
import librosa
import av # For av.AudioFrame
import traceback

TARGET_MIC_INPUT_SAMPLE_RATE = 16000

# --- AudioLoop Thread Management ---
def run_audioloop_in_thread(audioloop_instance, user_q, model_q, webrtc_q, model_audio_q):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    audioloop_instance.user_text_input_queue = user_q
    audioloop_instance.model_text_output_queue = model_q
    audioloop_instance.webrtc_audio_input_queue = webrtc_q
    audioloop_instance.model_audio_output_queue = model_audio_q
    
    print("Simplified WebApp: Starting AudioLoop.run() in a new thread.")
    try:
        loop.run_until_complete(audioloop_instance.run())
    except Exception as e:
        print(f"Simplified WebApp: Exception in AudioLoop thread: {e}")
        traceback.print_exc()
    finally:
        print("Simplified WebApp: AudioLoop thread finished.")
        loop.close()

# --- WebRTC Audio Processors ---
class WebRTCAudioRelay(AudioProcessorBase):
    def __init__(self, output_queue: asyncio.Queue):
        self.output_queue = output_queue
        self.last_sample_rate = None
        self.last_channels = None

    async def recv_queued(self, frames: list[av.AudioFrame], bundled_frames: list[av.AudioFrame] | None = None):
        all_processed_bytes = bytearray()
        if not frames:
            return []

        for frame in frames:
            if self.last_sample_rate != frame.sample_rate or self.last_channels != frame.layout.nb_channels:
                # Minimal log for property changes
                # print(f"Relay: Audio props changed: SR={frame.sample_rate}, Ch={frame.layout.nb_channels}")
                self.last_sample_rate = frame.sample_rate
                self.last_channels = frame.layout.nb_channels

            raw_audio_nd = frame.to_ndarray()
            float_audio_nd = raw_audio_nd.astype(np.float32) / 32768.0
            
            mono_audio = np.array([], dtype=np.float32)
            if frame.layout.nb_channels > 1: # Stereo or more
                if float_audio_nd.ndim == 2 and float_audio_nd.shape[0] == 1: # (1, N)
                    interleaved_samples = float_audio_nd[0]
                    try:
                        reshaped = interleaved_samples.reshape((-1, frame.layout.nb_channels)).T
                        mono_audio = librosa.to_mono(reshaped)
                    except Exception: # Simplified error handling
                        pass 
                elif float_audio_nd.ndim == 2 and float_audio_nd.shape[0] == frame.layout.nb_channels: # (C, N)
                    try:
                        mono_audio = librosa.to_mono(float_audio_nd)
                    except Exception:
                        pass
            elif frame.layout.nb_channels == 1: # Mono
                if float_audio_nd.ndim == 2 and float_audio_nd.shape[0] == 1: # (1, N)
                    mono_audio = float_audio_nd[0]
                elif float_audio_nd.ndim == 1: # (N,)
                     mono_audio = float_audio_nd
            
            if mono_audio.size == 0:
                continue

            resampled_audio = mono_audio
            if frame.sample_rate != TARGET_MIC_INPUT_SAMPLE_RATE:
                try:
                    resampled_audio = librosa.resample(mono_audio, orig_sr=frame.sample_rate, target_sr=TARGET_MIC_INPUT_SAMPLE_RATE)
                except Exception:
                    continue
            
            int16_audio = (np.clip(resampled_audio, -1.0, 1.0) * 32767).astype(np.int16)
            all_processed_bytes.extend(int16_audio.tobytes())
        
        if all_processed_bytes:
            try:
                self.output_queue.put_nowait(bytes(all_processed_bytes))
            except asyncio.QueueFull:
                print("WebRTCAudioRelay: Output queue full. Discarding audio.")
            except Exception as e:
                print(f"WebRTCAudioRelay: Error queueing audio: {e}")
        return []

class AudioPlayerProcessor(AudioProcessorBase):
    # Using TONE GENERATOR for now, as per original webapp_streamlit.py
    def __init__(self, model_audio_q: asyncio.Queue, target_sample_rate: int, target_channels: int):
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.bytes_per_20ms_frame = int(0.02 * target_sample_rate * target_channels * 2)
        self.phase = 0
        self.frequency = 440  # A4 note
        print(f"AudioPlayerProcessor (Tone Test) initialized: SR={target_sample_rate}, Ch={target_channels}")

    async def recv(self):
        samples_per_frame = self.bytes_per_20ms_frame // (self.target_channels * 2)
        t = (np.arange(samples_per_frame) + self.phase) / self.target_sample_rate
        tone = (0.3 * np.sin(2 * np.pi * self.frequency * t) * 32767).astype(np.int16)
        self.phase += samples_per_frame

        layout = 'mono'
        if self.target_channels == 1:
            audio_np_reshaped = tone.reshape(1, -1)
        else: # Crude stereo
            audio_np_reshaped = np.vstack([tone, tone]).reshape(self.target_channels, -1, order='F')
            layout = 'stereo'
        return av.AudioFrame.from_ndarray(audio_np_reshaped, format='s16', layout=layout, sample_rate=self.target_sample_rate)

# --- Factories for Processors ---
def create_webrtc_audio_relay_factory(queue_instance: asyncio.Queue):
    def factory():
        return WebRTCAudioRelay(output_queue=queue_instance)
    return factory

def create_audio_player_factory(model_audio_q: asyncio.Queue):
    def factory():
        return AudioPlayerProcessor(model_audio_q=model_audio_q, target_sample_rate=RECEIVE_SAMPLE_RATE, target_channels=GEMINI_OUTPUT_CHANNELS)
    return factory

# --- Main Streamlit App Logic ---
def main():
    st.set_page_config(layout="wide")
    st.title("Simplified Gemini Voice Agent")

    # Initialize session state variables
    if 'audioloop_running' not in st.session_state:
        st.session_state.audioloop_running = False
        st.session_state.audioloop_instance = None
        st.session_state.audioloop_thread = None
        st.session_state.user_text_input_queue = asyncio.Queue()
        st.session_state.model_text_output_queue = asyncio.Queue()
        st.session_state.webrtc_audio_queue = asyncio.Queue()
        st.session_state.model_audio_output_queue = asyncio.Queue()
        st.session_state.conversation_history = []
        st.session_state.stop_requested = False

    def start_service():
        if not st.session_state.audioloop_running:
            # Fresh queues for a new session
            st.session_state.user_text_input_queue = asyncio.Queue()
            st.session_state.model_text_output_queue = asyncio.Queue()
            st.session_state.webrtc_audio_queue = asyncio.Queue()
            st.session_state.model_audio_output_queue = asyncio.Queue()
            st.session_state.conversation_history = []

            st.session_state.audioloop_instance = AudioLoop(
                user_text_input_queue=st.session_state.user_text_input_queue,
                model_text_output_queue=st.session_state.model_text_output_queue,
                webrtc_audio_input_queue=st.session_state.webrtc_audio_queue,
                model_audio_output_queue=st.session_state.model_audio_output_queue
            )
            
            st.session_state.audioloop_thread = threading.Thread(
                target=run_audioloop_in_thread,
                args=(
                    st.session_state.audioloop_instance,
                    st.session_state.user_text_input_queue,
                    st.session_state.model_text_output_queue,
                    st.session_state.webrtc_audio_queue,
                    st.session_state.model_audio_output_queue
                ),
                daemon=True 
            )
            st.session_state.audioloop_thread.start()
            st.session_state.audioloop_running = True
            st.session_state.stop_requested = False
            print("Simplified WebApp: AudioLoop service started.")
            st.rerun()

    def stop_service():
        if st.session_state.audioloop_thread and st.session_state.audioloop_instance:
            print("Simplified WebApp: Requesting AudioLoop stop...")
            st.session_state.stop_requested = True
            if st.session_state.user_text_input_queue:
                st.session_state.user_text_input_queue.put_nowait(None)

            thread_to_join = st.session_state.audioloop_thread
            if thread_to_join.is_alive():
                thread_to_join.join(timeout=5.0)
                if thread_to_join.is_alive():
                    print("Simplified WebApp: WARNING - AudioLoop thread did not join.")
            
            st.session_state.audioloop_running = False
            st.session_state.audioloop_instance = None
            st.session_state.audioloop_thread = None
            st.session_state.stop_requested = False
            print("Simplified WebApp: AudioLoop service stopped.")
            st.rerun()

    # --- UI Layout ---
    col1, col2 = st.columns([3, 1])

    with col2: # Controls Column
        st.subheader("Controls")
        if not st.session_state.audioloop_running and not st.session_state.stop_requested:
            if st.button("Start Service"):
                start_service()
        elif st.session_state.audioloop_running:
            if st.button("Stop Service", type="primary"):
                stop_service()
        elif st.session_state.stop_requested:
            st.button("Stopping...", disabled=True)
        
        st.caption("Browser mic/speakers for audio I/O.")

        st.subheader("Microphone (Input)")
        webrtc_ctx_send = webrtc_streamer(
            key="webrtc-send",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=create_webrtc_audio_relay_factory(st.session_state.webrtc_audio_queue) if st.session_state.audioloop_running else None,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": False, "audio": True},
            desired_playing_state=st.session_state.audioloop_running
        )
        if st.session_state.audioloop_running:
            st.text(f"Mic Status: {'Active' if webrtc_ctx_send.state.playing else 'Inactive'}")
        else:
            st.info("Start service to enable mic.")

        st.subheader("Speaker (Output)")
        webrtc_ctx_recv = webrtc_streamer(
            key="webrtc-recv",
            mode=WebRtcMode.RECVONLY,
            audio_processor_factory=create_audio_player_factory(st.session_state.model_audio_output_queue) if st.session_state.audioloop_running else None,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": False, "audio": True},
            desired_playing_state=st.session_state.audioloop_running
        )
        if st.session_state.audioloop_running:
            st.text(f"Speaker Status: {'Active' if webrtc_ctx_recv.state.playing else 'Inactive'}")
        else:
            st.info("Start service to enable speakers.")


    with col1: # Conversation Column
        st.subheader("Conversation")
        chat_placeholder = st.container() # Use a container for dynamic updates

        with st.form(key='text_input_form', clear_on_submit=True):
            user_input = st.text_input("Your message:", disabled=not st.session_state.audioloop_running)
            submit_button = st.form_submit_button(label='Send', disabled=not st.session_state.audioloop_running)

        if submit_button and user_input:
            if st.session_state.audioloop_running and st.session_state.user_text_input_queue:
                st.session_state.conversation_history.append(("You", user_input))
                st.session_state.user_text_input_queue.put_nowait(user_input)
                st.rerun() 
            elif not st.session_state.audioloop_running:
                st.error("Service not running.")

        # Display conversation history
        with chat_placeholder: # Draw inside the container
            for speaker, text in st.session_state.conversation_history:
                text_cleaned = text.replace("\n", "<br>")
                if speaker == "You":
                    st.markdown(f"<div style='text-align: right; background-color: #DCF8C6; color: black; padding: 8px; border-radius: 8px; margin-left: 20%; margin-bottom: 5px; margin-top: 5px;'><b>You:</b> {text_cleaned}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align: left; background-color: #E9E9EB; color: black; padding: 8px; border-radius: 8px; margin-right: 20%; margin-bottom: 5px; margin-top: 5px;'><b>Gemini:</b> {text_cleaned}</div>", unsafe_allow_html=True)

    # Polling for model responses and updating UI
    if st.session_state.audioloop_running:
        model_message_received = False
        try:
            while st.session_state.model_text_output_queue and not st.session_state.model_text_output_queue.empty():
                model_response = st.session_state.model_text_output_queue.get_nowait()
                if model_response is None:
                    break 
                st.session_state.conversation_history.append(("Gemini", model_response))
                # print(f"Simplified WebApp: Gemini response added to history: '{model_response[:50]}...'") # Optional: minimal log
                model_message_received = True
        except asyncio.QueueEmpty:
            pass
        except Exception as e:
            print(f"Simplified WebApp: Error getting from model_text_output_queue: {e}")
        
        if model_message_received:
            st.rerun() # Rerun to update the chat display

    # Sidebar status
    if st.session_state.stop_requested:
        st.sidebar.warning("Service STOPPING...")
    elif st.session_state.audioloop_running:
        st.sidebar.success("Service RUNNING")
    else:
        st.sidebar.error("Service STOPPED")

if __name__ == "__main__":
    main() 
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
import time
from aiortc.mediastreams import AudioStreamTrack # Added for custom audio track

TARGET_MIC_INPUT_SAMPLE_RATE = 16000

# --- AudioLoop Thread Management ---
def run_audioloop_in_thread(audioloop_instance, user_q, model_q, webrtc_q, model_audio_q):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    audioloop_instance.user_text_input_queue = user_q
    audioloop_instance.model_text_output_queue = model_q
    audioloop_instance.webrtc_audio_input_queue = webrtc_q
    audioloop_instance.model_audio_output_queue = model_audio_q
    
    print("WebApp: Starting AudioLoop.run() in a new thread.")
    try:
        loop.run_until_complete(audioloop_instance.run())
    except Exception as e:
        print(f"WebApp: Exception in AudioLoop thread: {e}")
        traceback.print_exc()
    finally:
        print("WebApp: AudioLoop thread finished.")
        loop.close()

# --- WebRTC Audio Input Processor ---
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
                self.last_sample_rate = frame.sample_rate
                self.last_channels = frame.layout.nb_channels

            raw_audio_nd = frame.to_ndarray()
            float_audio_nd = raw_audio_nd.astype(np.float32) / 32768.0
            
            mono_audio = np.array([], dtype=np.float32)
            if frame.layout.nb_channels > 1:
                if float_audio_nd.ndim == 2 and float_audio_nd.shape[0] == 1:
                    interleaved_samples = float_audio_nd[0]
                    try:
                        reshaped = interleaved_samples.reshape((-1, frame.layout.nb_channels)).T
                        mono_audio = librosa.to_mono(reshaped)
                    except Exception: 
                        pass 
                elif float_audio_nd.ndim == 2 and float_audio_nd.shape[0] == frame.layout.nb_channels:
                    try:
                        mono_audio = librosa.to_mono(float_audio_nd)
                    except Exception:
                        pass
            elif frame.layout.nb_channels == 1: 
                if float_audio_nd.ndim == 2 and float_audio_nd.shape[0] == 1:
                    mono_audio = float_audio_nd[0]
                elif float_audio_nd.ndim == 1:
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

# --- WebRTC Audio Output Source Track ---
class WebAppAudioSourceTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(self, model_audio_q: asyncio.Queue, target_sample_rate: int, target_channels: int):
        super().__init__()
        print(f"!!! WebAppAudioSourceTrack __init__ CALLED. SR: {target_sample_rate}, CH: {target_channels} !!!")
        self.model_audio_q = model_audio_q
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        
        self.samples_per_20ms_frame = int(0.02 * self.target_sample_rate)

        self.phase = 0
        self.frequency = 440  
        self._start_time = time.time()
        self.last_pts = 0
        self._source_ended = False # New state variable

    async def recv(self):
        frame_to_send = None

        if self._source_ended:
            # If source (AudioLoop) signaled end, keep sending silence
            # print("WebAppAudioSourceTrack: Source ended, sending silent frame.") # Can be verbose
            silent_samples = np.zeros(self.samples_per_20ms_frame, dtype=np.int16)
            layout_silent = 'mono' if self.target_channels == 1 else 'stereo'
            shape_silent = (1, self.samples_per_20ms_frame) if self.target_channels == 1 else (self.target_channels, self.samples_per_20ms_frame)
            silent_data_np = np.zeros(shape_silent, dtype=np.int16)
            frame_to_send = av.AudioFrame.from_ndarray(silent_data_np, format='s16', layout=layout_silent)
        else:
            try:
                model_audio_bytes = self.model_audio_q.get_nowait()

                if model_audio_bytes is None: # Check for shutdown sentinel
                    print("WebAppAudioSourceTrack: Received None (shutdown sentinel) from model_audio_q. Switching to silent frames.")
                    self._source_ended = True
                    # Fall through to generate a silent frame for this call via the general error/silent path
                    # or directly create one here. For clarity, let current logic make it silent via exception or next call.
                    # For this immediate call, let's force a silent frame to be robust.
                    silent_samples = np.zeros(self.samples_per_20ms_frame, dtype=np.int16)
                    layout_silent = 'mono' if self.target_channels == 1 else 'stereo'
                    shape_silent = (1, self.samples_per_20ms_frame) if self.target_channels == 1 else (self.target_channels, self.samples_per_20ms_frame)
                    silent_data_np = np.zeros(shape_silent, dtype=np.int16)
                    frame_to_send = av.AudioFrame.from_ndarray(silent_data_np, format='s16', layout=layout_silent)
                else:
                    num_expected_bytes_per_frame = self.samples_per_20ms_frame * self.target_channels * 2
                    
                    if len(model_audio_bytes) < num_expected_bytes_per_frame:
                        padding_bytes = b'\\x00' * (num_expected_bytes_per_frame - len(model_audio_bytes))
                        model_audio_bytes += padding_bytes
                    elif len(model_audio_bytes) > num_expected_bytes_per_frame:
                        model_audio_bytes = model_audio_bytes[:num_expected_bytes_per_frame]

                    audio_data_np = np.frombuffer(model_audio_bytes, dtype=np.int16)
                    layout = 'mono'
                    if self.target_channels == 1:
                        frame_data_np = audio_data_np.reshape(1, self.samples_per_20ms_frame)
                    elif self.target_channels == 2:
                        layout = 'stereo'
                        frame_data_np = audio_data_np.reshape(self.samples_per_20ms_frame, 2).T 
                    else:
                        frame_data_np = audio_data_np.reshape(1, self.samples_per_20ms_frame)
                    frame_to_send = av.AudioFrame.from_ndarray(frame_data_np, format='s16', layout=layout)
                    self.model_audio_q.task_done()

            except asyncio.QueueEmpty:
                # Queue empty, generate TONE (if not _source_ended)
                # print("WebAppAudioSourceTrack: model_audio_q empty. Generating TONE.") # Can be verbose
                t = (np.arange(self.samples_per_20ms_frame) + self.phase) / self.target_sample_rate
                tone_data = (0.3 * np.sin(2 * np.pi * self.frequency * t) * 32767).astype(np.int16)
                self.phase += self.samples_per_20ms_frame
                layout_tone = 'mono'
                if self.target_channels == 1:
                    frame_data_tone_np = tone_data.reshape(1, self.samples_per_20ms_frame)
                elif self.target_channels == 2:
                    layout_tone = 'stereo'
                    stereo_tone_data_np = np.vstack([tone_data, tone_data]).T
                    frame_data_tone_np = stereo_tone_data_np.T
                else:
                    frame_data_tone_np = tone_data.reshape(1, self.samples_per_20ms_frame)
                frame_to_send = av.AudioFrame.from_ndarray(frame_data_tone_np, format='s16', layout=layout_tone)
            
            except Exception as e: # Catch other exceptions including the fixed TypeError potential
                print(f"WebAppAudioSourceTrack: Error in recv (after None check or during processing): {e}")
                traceback.print_exc()
                silent_samples = np.zeros(self.samples_per_20ms_frame, dtype=np.int16)
                layout_silent = 'mono' if self.target_channels == 1 else 'stereo'
                shape_silent = (1, self.samples_per_20ms_frame) if self.target_channels == 1 else (self.target_channels, self.samples_per_20ms_frame)
                silent_data_np = np.zeros(shape_silent, dtype=np.int16)
                frame_to_send = av.AudioFrame.from_ndarray(silent_data_np, format='s16', layout=layout_silent)

        # Common PTS and sample rate setting for any frame generated
        if frame_to_send:
            frame_to_send.sample_rate = self.target_sample_rate
            current_time_pts = time.time() - self._start_time
            if current_time_pts <= self.last_pts:
                current_time_pts = self.last_pts + 0.001 
            self.last_pts = current_time_pts
            frame_to_send.pts = current_time_pts
        else: 
            # This case should be rarer now with explicit silent frame on None from queue
            print("WebAppAudioSourceTrack: CRITICAL - frame_to_send is None post processing. Emergency silent frame.")
            silent_samples = np.zeros(self.samples_per_20ms_frame, dtype=np.int16)
            layout_silent = 'mono' if self.target_channels == 1 else 'stereo'
            shape_silent = (1, self.samples_per_20ms_frame) if self.target_channels == 1 else (self.target_channels, self.samples_per_20ms_frame)
            silent_data_np = np.zeros(shape_silent, dtype=np.int16)
            frame_to_send = av.AudioFrame.from_ndarray(silent_data_np, format='s16', layout=layout_silent)
            frame_to_send.sample_rate = self.target_sample_rate
            frame_to_send.pts = self.last_pts + 0.02 # Crude PTS update to keep it moving

        await asyncio.sleep(0.018)
        return frame_to_send

# --- Factories for Processors/Tracks ---
def create_webrtc_audio_relay_factory(queue_instance: asyncio.Queue):
    def factory():
        return WebRTCAudioRelay(output_queue=queue_instance)
    return factory

# Removed create_audio_player_factory

# --- Main Streamlit App Logic ---
def main():
    st.set_page_config(layout="wide")
    st.title("Simplified Gemini Voice Agent (WebApp)")

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
        st.session_state.webapp_audio_source_track = None # For the new track instance

    def start_service():
        if not st.session_state.audioloop_running:
            st.session_state.user_text_input_queue = asyncio.Queue()
            st.session_state.model_text_output_queue = asyncio.Queue()
            st.session_state.webrtc_audio_queue = asyncio.Queue()
            st.session_state.model_audio_output_queue = asyncio.Queue()
            st.session_state.conversation_history = []

            # Create and store the WebAppAudioSourceTrack instance
            st.session_state.webapp_audio_source_track = WebAppAudioSourceTrack(
                model_audio_q=st.session_state.model_audio_output_queue,
                target_sample_rate=RECEIVE_SAMPLE_RATE,
                target_channels=GEMINI_OUTPUT_CHANNELS
            )
            print(f"WebApp: Created WebAppAudioSourceTrack with model_audio_q ID: {id(st.session_state.model_audio_output_queue)}")
            print(f"WebApp: WebAppAudioSourceTrack instance ID: {id(st.session_state.webapp_audio_source_track)}")

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
            print("WebApp: AudioLoop service started.")
            st.rerun()

    def stop_service():
        if st.session_state.audioloop_thread and st.session_state.audioloop_instance:
            print("WebApp: Requesting AudioLoop stop...")
            st.session_state.stop_requested = True
            if st.session_state.user_text_input_queue:
                st.session_state.user_text_input_queue.put_nowait(None)

            thread_to_join = st.session_state.audioloop_thread
            if thread_to_join.is_alive():
                thread_to_join.join(timeout=5.0)
                if thread_to_join.is_alive():
                    print("WebApp: WARNING - AudioLoop thread did not join.")
            
            st.session_state.audioloop_running = False
            st.session_state.audioloop_instance = None
            st.session_state.audioloop_thread = None
            st.session_state.webapp_audio_source_track = None # Clear track instance
            st.session_state.stop_requested = False
            print("WebApp: AudioLoop service stopped.")
            st.rerun()

    col1, col2 = st.columns([3, 1])

    with col2: 
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
        if st.session_state.audioloop_running and st.session_state.webapp_audio_source_track:
            print(f"WebApp: Passing WebAppAudioSourceTrack instance {id(st.session_state.webapp_audio_source_track)} to webrtc_streamer for speaker.")
            webrtc_ctx_recv = webrtc_streamer(
                key="webrtc-recv",
                mode=WebRtcMode.RECVONLY,
                source_audio_track=st.session_state.webapp_audio_source_track, # Use source_audio_track
                # audio_processor_factory=None, # Explicitly None if using source_audio_track
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": False, "audio": True},
                desired_playing_state=st.session_state.audioloop_running
            )
            st.text(f"Speaker Status: {'Active' if webrtc_ctx_recv.state.playing else 'Inactive'}")
        elif st.session_state.audioloop_running and not st.session_state.webapp_audio_source_track:
             st.error("WebAppAudioSourceTrack not initialized. Cannot start speaker.")
        else:
            st.info("Start service to enable speakers.")

    with col1: 
        st.subheader("Conversation")
        chat_placeholder = st.container()

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

        with chat_placeholder:
            for speaker, text in st.session_state.conversation_history:
                text_cleaned = text.replace("\n", "<br>")
                if speaker == "You":
                    st.markdown(f"<div style='text-align: right; background-color: #DCF8C6; color: black; padding: 8px; border-radius: 8px; margin-left: 20%; margin-bottom: 5px; margin-top: 5px;'><b>You:</b> {text_cleaned}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align: left; background-color: #E9E9EB; color: black; padding: 8px; border-radius: 8px; margin-right: 20%; margin-bottom: 5px; margin-top: 5px;'><b>Gemini:</b> {text_cleaned}</div>", unsafe_allow_html=True)

    if st.session_state.audioloop_running:
        model_message_received = False
        try:
            while st.session_state.model_text_output_queue and not st.session_state.model_text_output_queue.empty():
                model_response = st.session_state.model_text_output_queue.get_nowait()
                if model_response is None:
                    break 
                st.session_state.conversation_history.append(("Gemini", model_response))
                model_message_received = True
        except asyncio.QueueEmpty:
            pass
        except Exception as e:
            print(f"WebApp: Error getting from model_text_output_queue: {e}")
        
        if model_message_received:
            st.rerun()

    if st.session_state.stop_requested:
        st.sidebar.warning("Service STOPPING...")
    elif st.session_state.audioloop_running:
        st.sidebar.success("Service RUNNING")
    else:
        st.sidebar.error("Service STOPPED")

if __name__ == "__main__":
    main() 
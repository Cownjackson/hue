import streamlit as st
import sys
import asyncio
import os
import pyaudio 
import numpy as np
import librosa 
# from google import genai # Not used in this test
# from google.genai import types # Not used in this test
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, RTCConfiguration

# Configuration ( 일부는 WebRTCAudioReceiver에서 직접 사용됨 )
FORMAT = pyaudio.paInt16 # PyAudio format
CHANNELS = 1
SEND_SAMPLE_RATE = 16000 # Target sample rate for processing/playback
# RECEIVE_SAMPLE_RATE = 24000 # Not used in this simple test
# MODEL = "models/gemini-2.0-flash-live-001" # Not used

load_dotenv()

# --- Session state variables initialization ---
if "webrtc_audio_input_queue" not in st.session_state: # Still needed for the factory
    st.session_state.webrtc_audio_input_queue = asyncio.Queue()
if "webrtc_desired_playing_state" not in st.session_state:
    st.session_state.webrtc_desired_playing_state = False
if "audio_active" not in st.session_state: # For simple start/stop UI
    st.session_state.audio_active = False
if "messages" not in st.session_state: # For a simple info message
    st.session_state.messages = []


class WebRTCAudioReceiver(AudioProcessorBase):
    def __init__(self, audio_input_queue: asyncio.Queue): # audio_input_queue is not actively used
        super().__init__()
        self.target_sr = SEND_SAMPLE_RATE 
        self.target_channels = CHANNELS
        self._pyaudio_instance = None
        self._output_stream = None
        print(f"DEBUG: WebRTCAudioReceiver __init__ called.", file=sys.stderr)
        try:
            self._pyaudio_instance = pyaudio.PyAudio()
            
            # --- PYAUDIO DEVICE LISTING --- 
            print("DEBUG: Available PyAudio output devices:", file=sys.stderr)
            for i in range(self._pyaudio_instance.get_device_count()):
                dev_info = self._pyaudio_instance.get_device_info_by_index(i)
                if dev_info.get('maxOutputChannels') > 0:
                    print(f"  Device {i}: {dev_info.get('name')} (Output Channels: {dev_info.get('maxOutputChannels')}, Default SR: {dev_info.get('defaultSampleRate')})", file=sys.stderr)
            # --- END PYAUDIO DEVICE LISTING ---
            
            self._output_stream = self._pyaudio_instance.open(
                format=FORMAT, # Use constant
                channels=self.target_channels,
                rate=self.target_sr,
                output=True
            )
            print("DEBUG: WebRTCAudioReceiver direct PyAudio output stream opened.", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: WebRTCAudioReceiver failed to open PyAudio output stream: {e}", file=sys.stderr)
            if self._pyaudio_instance: # Terminate if open failed after instance creation
                self._pyaudio_instance.terminate()
            self._pyaudio_instance = None
            self._output_stream = None

        # Diagnostic for _event_loop from superclass
        if hasattr(self, '_event_loop') and self._event_loop:
            print(f"DEBUG (WebRTCAudioReceiver): self._event_loop from super: {self._event_loop}, running={self._event_loop.is_running()}", file=sys.stderr)
        else:
            print(f"WARN (WebRTCAudioReceiver): self._event_loop not set by super or None.", file=sys.stderr)

    async def recv_queued(self, frames):
        print(f"DEBUG: WebRTCAudioReceiver.recv_queued CALLED with {len(frames)} frames. id(self): {id(self)}, self._output_stream is None: {self._output_stream is None}", file=sys.stderr)
        if not self._output_stream:
            print(f"DEBUG: recv_queued exiting early because self._output_stream is None. id(self): {id(self)}", file=sys.stderr)
            return frames

        all_audio_bytes = bytearray()
        for frame in frames:
            try:
                # --- INSPECT AudioFrame and raw_audio_nd --- 
                print(f"DEBUG: frame: format={frame.format.name}, layout={frame.layout.name}, samples={frame.samples}, rate={frame.sample_rate}, pts={frame.pts}, time_base={frame.time_base}", file=sys.stderr)
                raw_audio_nd = frame.to_ndarray() 
                print(f"DEBUG: raw_audio_nd (from frame.to_ndarray()): shape={raw_audio_nd.shape}, dtype={raw_audio_nd.dtype}, min={np.min(raw_audio_nd) if raw_audio_nd.size > 0 else 'N/A'}, max={np.max(raw_audio_nd) if raw_audio_nd.size > 0 else 'N/A'}", file=sys.stderr)
                # --- END INSPECTION ---

                # --- De-interleave, convert to float, then to mono ---
                current_sr = frame.sample_rate
                current_channels = frame.layout.nb_channels
                samples_per_channel = frame.samples

                # raw_audio_nd is typically shape (1, total_samples_interleaved) for packed stereo
                if raw_audio_nd.ndim == 2 and raw_audio_nd.shape[0] == 1:
                    interleaved_samples = raw_audio_nd[0] # Extract 1D array
                elif raw_audio_nd.ndim == 1: # Already 1D
                    interleaved_samples = raw_audio_nd
                else:
                    print(f"WARN: Unexpected shape for raw_audio_nd: {raw_audio_nd.shape}. Using zeros.", file=sys.stderr)
                    interleaved_samples = np.zeros(samples_per_channel * current_channels if current_channels > 0 else samples_per_channel, dtype=np.int16)


                if np.issubdtype(interleaved_samples.dtype, np.integer):
                    # For int16, dividing by 32768.0 scales to approx [-1.0, 1.0]
                    float_audio = interleaved_samples.astype(np.float32) / 32768.0
                elif np.issubdtype(interleaved_samples.dtype, np.floating):
                    float_audio = interleaved_samples # Assume it's already in a good range if float
                else:
                    print(f"WARN: Unexpected audio data type from frame: {interleaved_samples.dtype}. Attempting float conversion.", file=sys.stderr)
                    float_audio = interleaved_samples.astype(np.float32)


                mono_audio_for_resample = np.array([], dtype=np.float32) # Initialize to empty

                if current_channels == 2 and self.target_channels == 1:
                    expected_size = samples_per_channel * 2
                    if float_audio.size == expected_size:
                        # Reshape (L,R,L,R,...) to (num_frames, 2) then transpose to (2, num_frames)
                        stereo_reshaped = float_audio.reshape((-1, 2)).T
                        mono_audio_for_resample = librosa.to_mono(stereo_reshaped)
                    else:
                        print(f"WARN: Mismatch in expected stereo samples (expected {expected_size}, got {float_audio.size}). Using zeros.", file=sys.stderr)
                        mono_audio_for_resample = np.zeros(samples_per_channel, dtype=np.float32)
                elif current_channels == 1:
                    if float_audio.size == samples_per_channel:
                        mono_audio_for_resample = float_audio
                    else: # Fallback if size mismatch for mono
                         print(f"WARN: Mismatch in expected mono samples (expected {samples_per_channel}, got {float_audio.size}). Using zeros.", file=sys.stderr)
                         mono_audio_for_resample = np.zeros(samples_per_channel, dtype=np.float32)
                else: # More than 2 channels or other unsupported configurations
                    print(f"WARN: Unsupported channel configuration: {current_channels} to {self.target_channels}. Using zeros.", file=sys.stderr)
                    mono_audio_for_resample = np.zeros(samples_per_channel, dtype=np.float32)
                
                # At this point, mono_audio_for_resample should be a 1D float NumPy array

                if current_sr != self.target_sr:
                    if mono_audio_for_resample.size > 0: # Only resample if there's data
                        resampled_audio_nd = librosa.resample(mono_audio_for_resample, orig_sr=current_sr, target_sr=self.target_sr)
                    else: # If mono_audio_for_resample is empty (e.g. from a warning path), keep it empty
                        resampled_audio_nd = mono_audio_for_resample # which is np.array([])
                else:
                    resampled_audio_nd = mono_audio_for_resample # Already float and 1D
                
                # Clip and convert to int16 for PyAudio output stream
                # resampled_audio_nd is now expected to be 1D
                audio_int16 = (np.clip(resampled_audio_nd, -1.0, 1.0) * 32767).astype(np.int16)
                
                all_audio_bytes.extend(audio_int16.tobytes())
                if len(audio_int16) > 0:
                    print(f"DEBUG: audio_int16 stats: min={np.min(audio_int16)}, max={np.max(audio_int16)}, mean={np.mean(audio_int16):.2f}, len={len(audio_int16)}", file=sys.stderr)
                else:
                    print("DEBUG: audio_int16 is empty!", file=sys.stderr)
            except Exception as e:
                print(f"ERROR processing frame in recv_queued: {e}", file=sys.stderr)
        
        if all_audio_bytes:
            try:
                self._output_stream.write(bytes(all_audio_bytes))
            except Exception as e:
                print(f"ERROR: WebRTCAudioReceiver PyAudio stream write error: {e}", file=sys.stderr)
        return frames

    def __del__(self):
        print("DEBUG: WebRTCAudioReceiver __del__ called.", file=sys.stderr)
        if self._output_stream:
            try: 
                self._output_stream.stop_stream()
                self._output_stream.close()
                print("DEBUG: WebRTCAudioReceiver PyAudio output stream closed.", file=sys.stderr)
            except Exception as e: print(f"ERROR closing output stream in __del__: {e}", file=sys.stderr)
        if self._pyaudio_instance:
            try: 
                self._pyaudio_instance.terminate()
                print("DEBUG: WebRTCAudioReceiver PyAudio instance terminated.", file=sys.stderr)
            except Exception as e: print(f"ERROR terminating PyAudio in __del__: {e}", file=sys.stderr)

# --- Main UI and Simplified Logic ---
st.set_page_config(layout="wide")
st.title("🎙️ Audio Passthrough Test")

chat_container = st.container() # Just for the info message
with chat_container:
    if not st.session_state.get("audio_active", False): # Show initial message
         st.info("Click 'Start Audio Test' and speak. If working, you should hear your voice played back (echo).")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def create_audio_receiver_factory(queue_instance: asyncio.Queue):
    print(f"DEBUG: create_audio_receiver_factory (simplified test) called.", file=sys.stderr)
    def factory():
        print(f"DEBUG (factory simplified test): Instantiating WebRTCAudioReceiver.", file=sys.stderr)
        return WebRTCAudioReceiver(queue_instance)
    return factory

webrtc_ctx = webrtc_streamer(
    key="audio-passthrough-test",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=create_audio_receiver_factory(st.session_state.webrtc_audio_input_queue),
    media_stream_constraints={"video": False, "audio": True},
    desired_playing_state=st.session_state.webrtc_desired_playing_state,
    rtc_configuration=RTC_CONFIGURATION
)

start_stop_col, status_col = st.columns([1,2])
with start_stop_col:
    if not st.session_state.audio_active:
        if st.button("▶️ Start Audio Test", key="start_audio_test_button"):
            st.session_state.webrtc_desired_playing_state = True
            st.session_state.audio_active = True
            st.rerun()
    else:
        if st.button("⏹️ Stop Audio Test", key="stop_audio_test_button"):
            st.session_state.webrtc_desired_playing_state = False
            st.session_state.audio_active = False
            st.rerun()

with status_col:
    status_text = "⚪ Idle"
    webrtc_playing = hasattr(webrtc_ctx, 'state') and getattr(webrtc_ctx.state, 'playing', False)
    if st.session_state.audio_active and webrtc_playing:
        status_text = "🟢 Audio Test Running (Speak for Echo)"
    elif st.session_state.audio_active: # Button clicked, but WebRTC might not be fully active yet
        status_text = "🟡 Audio Test Activating..."
    st.markdown(f"**Status:** {status_text}")
    st.markdown(f"**WebRTC:** {'🎤 Active' if webrtc_playing else ' inactive/needs permission'}")

st.markdown("---    \n**Note:** This is a simplified audio passthrough test. Audio from your microphone should be played back directly. If you hear an echo, it's working.") 
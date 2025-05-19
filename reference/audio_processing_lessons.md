Key Learnings from WebRTC Audio Passthrough Experiment (webapp_new_test.py)

This document summarizes the main findings and debugging steps taken to successfully capture audio from the microphone via WebRTC, process it, and play it back using PyAudio.

1.  WebRTC Audio Frame Reception:
    *   The `WebRTCAudioReceiver.recv_queued(self, frames)` method, inheriting from `AudioProcessorBase` in `streamlit-webrtc`, is correctly called when the WebRTC connection is active and the microphone is sending data.
    *   It receives a list of `av.AudioFrame` objects.

2.  Audio Data Extraction and Initial Format:
    *   `frame.to_ndarray()` is used to convert an `av.AudioFrame` into a NumPy array.
    *   The raw audio data from the microphone typically arrived as:
        *   Data Type (`dtype`): `int16` (signed 16-bit integers).
        *   Layout: Stereo.
        *   Shape: Often `(1, N)`, where N is the total number of interleaved samples (e.g., `(1, 1920)` for 960 samples per channel at 48kHz for a 20ms frame, meaning `samples_per_channel * num_channels`).
    *   `frame.samples` attribute indicates samples *per channel*.
    *   `frame.sample_rate` indicates the original sample rate from WebRTC (e.g., 48000 Hz).
    *   `frame.layout.nb_channels` gives the number of channels.

3.  Librosa Data Requirements:
    *   Librosa functions (e.g., `librosa.resample`, `librosa.to_mono`) strictly require audio data to be in floating-point format (e.g., `np.float32`).
    *   Passing integer arrays directly results in a `librosa.util.exceptions.ParameterError: Audio data must be floating-point`.

4.  Data Type Conversion and Normalization for Librosa:
    *   Integer audio data (e.g., `int16`) must be converted to `float32`.
    *   It should also be normalized to a range of `[-1.0, 1.0]`. For `int16` data, this is typically done by dividing the `int16` array by `32768.0`.

5.  Stereo to Mono Conversion:
    *   The target format for playback and further processing (like sending to Gemini) was mono.
    *   If the input is stereo (e.g., `shape=(1, total_interleaved_samples)` from `frame.to_ndarray()`), it needs to be correctly converted to mono.
    *   The successful process was:
        1.  Extract the 1D array of interleaved samples (e.g., `raw_audio_nd[0]`).
        2.  Convert this 1D array to `float32` and normalize it.
        3.  If stereo (2 channels), reshape the 1D interleaved float array into a 2D array of shape `(num_samples_per_channel, 2)` (L/R pairs) and then transpose it to `(2, num_samples_per_channel)`. This is the format `librosa.to_mono` expects.
            *   Example: `stereo_reshaped = float_audio.reshape((-1, 2)).T`
        4.  Pass this `(2, N)` array to `librosa.to_mono()`. The result is a 1D mono array.
        5.  If the input was already mono, this step is skipped, and the 1D float array is used directly.

6.  Resampling:
    *   `librosa.resample(y, orig_sr, target_sr)` was used to change the sample rate.
    *   `y` must be a 1D NumPy array (mono, float).
    *   `orig_sr` is the original sample rate (from `frame.sample_rate`).
    *   `target_sr` is the desired sample rate (e.g., `16000 Hz`).

7.  Preparing Audio for PyAudio Playback:
    *   PyAudio was configured to play `paInt16` mono audio at the `target_sr`.
    *   The resampled float audio (1D mono array, range `[-1.0, 1.0]`) was:
        1.  Clipped to ensure values are strictly within `[-1.0, 1.0]` using `np.clip()`.
        2.  Scaled back to `int16` range by multiplying by `32767`.
        3.  Converted to `np.int16` dtype.
        4.  Converted to bytes using `audio_int16.tobytes()` for `PyAudio.Stream.write()`.

8.  PyAudio Stream Management:
    *   An instance of `pyaudio.PyAudio()` is needed.
    *   An output stream is opened using `pa_instance.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)`.
    *   Proper cleanup is essential: `stream.stop_stream()`, `stream.close()`, and `pa_instance.terminate()`. This was handled in the `__del__` method of `WebRTCAudioReceiver` for the passthrough test.

9.  Streamlit Reruns and Object Instances:
    *   `streamlit-webrtc` can create new instances of the `AudioProcessorBase` subclass (our `WebRTCAudioReceiver`) on Streamlit app reruns.
    *   This means that resources initialized in `__init__` (like the PyAudio stream) could potentially be orphaned or lead to unexpected behavior if not managed carefully, especially if multiple instances are briefly active or if `recv_queued` is called on an instance different from the one whose `__init__` set up the intended resources.
    *   In the successful test, the PyAudio stream was initialized in `__init__`, and the `recv_queued` method of the *same instance* correctly used that stream. Debugging with `id(self)` was helpful to observe instance creation.

10. Debugging Silent or Incorrect Audio:
    *   Printing the `shape`, `dtype`, `min`, `max`, and `len` of NumPy arrays at each stage of processing was crucial for identifying where the audio data was being lost or corrupted.
    *   The "all zeros with `len=1`" issue was due to incorrect array shapes being passed to Librosa functions, leading them to return empty or zeroed arrays that, when further processed, appeared as a 1-element array (often a 2D array `[[0]]` or similar, whose `len()` is 1). Correctly preparing 1D mono arrays for resampling and `int16` conversion fixed this.

This covers the main technical hurdles and solutions encountered during that specific debugging phase.
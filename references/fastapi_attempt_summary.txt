Summary of the FastAPI & WebSocket Attempt:

1.  **Initial Goal:** Make the functionalities of `google_ai_sample.py` accessible via a web browser on your phone.
2.  **Chosen Architecture:**
    *   **Backend:** A Python FastAPI server (`main_web.py`) to handle WebSocket connections and interact with the Gemini Live API.
    *   **Frontend:** An HTML page (`index.html`) with JavaScript to manage microphone input, audio playback, text input/output, and WebSocket communication with the backend.
3.  **Key Implementation Steps & Challenges:**
    *   **Basic Setup:**
        *   Added `fastapi` and `uvicorn` to `requirements.txt`.
        *   Created `main_web.py` with a WebSocket endpoint (`/ws/voice`) and a basic `WebAudioHandler` class.
        *   Created `index.html` with UI elements and JavaScript for WebSocket connection, media recording, and message display.
    *   **Gemini Integration:**
        *   Adapted the Gemini Live API logic (session creation, audio/text sending and receiving, tool handling) from `google_ai_sample.py` into `WebAudioHandler`.
    *   **Audio Format Issues (Client to Server to Gemini):**
        *   Initial attempts involved transcoding browser audio (WebM/Opus) to PCM using `pydub`, leading to `ffmpeg` path issues and `CouldntDecodeError`.
        *   Realized the Gemini Live API (`v1beta`) likely accepts formats like `audio/webm` directly, so `pydub` was removed.
    *   **Client-Side Audio Handling (JavaScript in `index.html`):**
        *   Configured `MediaRecorder` for `audio/webm;codecs=opus`.
        *   Changed audio data transmission to use `ArrayBuffer`s over WebSockets.
        *   Implemented Web Audio API for playing raw PCM audio (16-bit, 1ch, 24kHz) from Gemini.
        *   Fixed various JavaScript errors (e.g., `wsReconnectTimeout`, inconsistent variables, JSON parsing for text messages).
    *   **WebSocket Connection Errors (Python Server to Gemini API):**
        *   After client-to-server audio seemed stable, encountered `ConnectionClosedError: received 1007 (invalid frame payload data)` from the Google Gemini server. This occurred when our Python backend tried to interact with Gemini, even before user audio was sent.
        *   Suspected the `MODEL = "models/gemini-2.0-flash-live-001"`.
        *   Changing `MODEL` to `"models/gemini-1.5-flash-latest"` resulted in a clearer error from Google: `ConnectionClosedError: received 1008 (policy violation) models/gemini-1.5-flash-latest is not found for API version v1beta, or is not supported for bidiGenerateContent`, confirming this model was unsuitable for the Live API and that the original model ID was closer but still problematic for an unknown reason (possibly an "invalid argument" in the initial request to Google not related to audio data itself).
    *   **Model Identification:** The final step before deciding to revert was to add code to list available models from the `v1beta` API to find one explicitly supporting `bidiGenerateContent`. 
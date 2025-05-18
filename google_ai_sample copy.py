"""
## Documentation
Quickstart: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py

## Setup

To install the dependencies for this script, run:

```
pip install google-genai opencv-python pyaudio pillow mss
```
"""

import os
import asyncio
import base64
import io
import traceback

import pyaudio

import argparse

from google import genai
from google.genai import types
from dotenv import load_dotenv

# Import the new light control functions
from test_utils import turn_office_light_on, turn_office_light_off

load_dotenv()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001"

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)

tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="getWeather",
                description="gets the weather for a requested city",
                parameters=genai.types.Schema(
                    type = genai.types.Type.OBJECT,
                    properties = {
                        "city": genai.types.Schema(
                            type = genai.types.Type.STRING,
                        ),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="turn_office_light_on",
                description="Turns the office light on.",
                # No parameters needed for this function
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={}
                )
            ),
            types.FunctionDeclaration(
                name="turn_office_light_off",
                description="Turns the office light off.",
                # No parameters needed for this function
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={}
                )
            ),
        ]
    ),
]

CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
    tools=tools,
)

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self):

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            text_to_send = text if text else "."
            user_content = types.Content(role="user", parts=[types.Part(text=text_to_send)])
            await self.session.send_client_content(turns=user_content, turn_complete=True)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send_realtime_input(media=msg.inline_data)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put(types.Part(inline_data=types.Blob(data=data, mime_type="audio/pcm")))

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive() # turn is an async iterable of LiveServerMessage
            async for response_msg in turn: # response_msg is a LiveServerMessage
                function_call_found_and_handled = False

                # Check for model_turn and function calls within its parts
                model_turn_content = getattr(response_msg, 'model_turn', None)
                if model_turn_content: # If model_turn exists and is not None
                    for part in model_turn_content.parts:
                        fc = getattr(part, 'function_call', None)
                        if fc: # If function_call exists on this part
                            result = None
                            error = None
                            print(f"Executing function call: {fc.name} with args: {fc.args}")

                            if fc.name == "turn_office_light_on":
                                try:
                                    success = turn_office_light_on()
                                    result = {"success": success}
                                except Exception as e:
                                    print(f"Error executing {fc.name}: {e}")
                                    error = str(e)
                            elif fc.name == "turn_office_light_off":
                                try:
                                    success = turn_office_light_off()
                                    result = {"success": success}
                                except Exception as e:
                                    print(f"Error executing {fc.name}: {e}")
                                    error = str(e)
                            elif fc.name == "getWeather":
                                # Placeholder for actual weather function implementation
                                city_arg = fc.args.get("city", "unknown city") if fc.args else "unknown city"
                                print(f"Function call: getWeather, Args: {fc.args}")
                                result = {"weather": "sunny", "location": city_arg} # Example response
                            else:
                                print(f"Unknown function call: {fc.name}")
                                error = f"Unknown function call: {fc.name}"
                            
                            if result is not None:
                                function_response_payload = {'result': result}
                            else:
                                function_response_payload = {'error': error or "Unknown error occurred"}
                            
                            print(f"Sending function response for {fc.name}: {function_response_payload}")
                            response_part_to_send = types.Part.from_function_response(
                                name=fc.name,
                                response=function_response_payload
                            )
                            
                            await self.session.send_client_content(
                                turns=[types.Content(parts=[response_part_to_send])],
                                turn_complete=False # Keep the turn open for the model to respond
                            )
                            function_call_found_and_handled = True
                            break # Processed a function call in this model_turn, break from parts loop
                
                if function_call_found_and_handled:
                    continue # Move to the next message from the server

                # If no function call was handled in this message, process for audio data or text
                audio_data = getattr(response_msg, 'data', None)
                if audio_data:
                    self.audio_in_queue.put_nowait(audio_data)
                    continue # If there's audio data, assume this message is primarily for audio
                
                text_data = getattr(response_msg, 'text', None)
                if text_data:
                    print(text_data, end="")
                    # Continue to process next message, as this one's text has been printed
                    continue

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    main = AudioLoop()
    asyncio.run(main.run())

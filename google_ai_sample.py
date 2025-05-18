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

# Import the new tools
from tools import turn_office_light_on, turn_office_light_off

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

# Define the new tools for the voice agent
tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="turn_office_light_on",
                description="Turns the office lamp on.",
                # No parameters needed for this simple version
                parameters=genai.types.Schema(type=genai.types.Type.OBJECT, properties={}),
            ),
        ]
    ),
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="turn_office_light_off",
                description="Turns the office lamp off.",
                # No parameters needed for this simple version
                parameters=genai.types.Schema(type=genai.types.Type.OBJECT, properties={}),
            ),
        ]
    )
]

CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
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
        "Background task to read from the websocket, handle server messages, and write PCM audio chunks to the output queue."
        while True:
            async for response_message in self.session.receive(): # response_message is LiveServerMessage
                # Priority 1: Handle Tool Call requests from the server
                if server_tool_call := response_message.tool_call:
                    await handle_tool_call(self.session, server_tool_call)
                    continue  # Tool call messages are typically standalone

                # Priority 2: Process Server Content (text, audio, execution results, status updates)
                if server_content := response_message.server_content:
                    if model_turn_content := server_content.model_turn:
                        for part in model_turn_content.parts:
                            if part.text:
                                print(part.text, end="", flush=True)
                            elif part.inline_data and part.inline_data.data: # inline_data is a Blob
                                self.audio_in_queue.put_nowait(part.inline_data.data)
                            elif exec_result := part.code_execution_result:
                                outcome_str = exec_result.outcome if exec_result.outcome else "UNSPECIFIED"
                                output_str = exec_result.output if exec_result.output else ""
                                print(f"\n[Code Execution Result From Server]: Outcome: {outcome_str}, Output: \"{output_str}\"", flush=True)
                            # else:
                                # You can uncomment this to log any other unexpected part types during development
                                # print(f"\n[Unhandled Part in Model Turn]: {part}", flush=True)
                    
                    # Handle interruption: If the server indicates its output generation was interrupted by the client
                    if server_content.interrupted:
                        print("\n[Playback Interrupted by Server Signal]", flush=True)
                        # Empty the audio queue to stop playback of any buffered audio
                        while not self.audio_in_queue.empty():
                            try:
                                self.audio_in_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break 
                    
                    # server_content.turn_complete indicates the model has finished its current segment of output.
                    # No specific action is taken here just for turn_complete unless it's part of an interruption.
                    # The original interruption logic was: "If you interrupt the model, it sends a turn_complete."
                    # The `interrupted` flag is a more direct signal for this.

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

async def handle_tool_call(session, server_tool_call: types.LiveServerToolCall):
    print(f"Server tool call received: {server_tool_call}")
    
    function_responses = []
    
    if server_tool_call.function_calls:
        for fc in server_tool_call.function_calls:
            tool_id = fc.id
            tool_name = fc.name
            # tool_args = fc.args # Not strictly needed for these parameter-less tools, but good to have if args were present
            
            print(f"Executing tool: {tool_name} (ID: {tool_id})")
            
            tool_result_dict = None # This will store the dict like {"status": "...", "message": "..."}

            try:
                if tool_name == "turn_office_light_on":
                    tool_result_dict = turn_office_light_on()
                elif tool_name == "turn_office_light_off":
                    tool_result_dict = turn_office_light_off()
                else:
                    tool_result_dict = {"status": "error", "message": f"Tool '{tool_name}' not found."}
                    print(tool_result_dict["message"])

                # The FunctionResponse expects a dictionary for its 'response' field.
                # We will pass the entire tool_result_dict which includes status and message.
                function_responses.append(types.FunctionResponse(
                    id=tool_id,
                    name=tool_name,
                    response=tool_result_dict 
                ))
            
            except Exception as e:
                print(f"An exception occurred while executing tool {tool_name} (ID: {tool_id}): {e}")
                # Ensure a response is sent even if an unexpected error occurs during tool execution
                function_responses.append(types.FunctionResponse(
                    id=tool_id,
                    name=tool_name,
                    response={"status": "error", "message": f"Exception during {tool_name} execution: {str(e)}"}
                ))

    if function_responses:
        # The send_tool_response method expects a list of FunctionResponse objects directly.
        print(f"Sending tool responses to Gemini: {function_responses}")
        await session.send_tool_response(function_responses=function_responses)
    else:
        print("No function calls were processed or found in the server_tool_call message.")


if __name__ == "__main__":
    main = AudioLoop()
    asyncio.run(main.run())

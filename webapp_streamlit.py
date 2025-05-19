import streamlit as st
import asyncio
import threading
from queue import Queue as ThreadQueue
from google_ai_sample import AudioLoop # Assuming types and other necessary components are accessible or re-imported if needed

# Ensure PyAudio is terminated when Streamlit shuts down (this is a bit tricky with Streamlit's execution model)
# A cleaner way would be to manage the PyAudio instance within AudioLoop and ensure its cleanup.
# For now, we rely on AudioLoop's cleanup.

def run_audioloop_in_thread(audioloop_instance, streamlit_user_q, streamlit_model_q):
    """Target function for the AudioLoop thread."""
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Assign the Streamlit-managed asyncio.Queues to the audioloop_instance
    audioloop_instance.user_text_input_queue = streamlit_user_q
    audioloop_instance.model_text_output_queue = streamlit_model_q
    
    print("Starting AudioLoop.run() in a new thread...")
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
st.title("Gemini Live Voice Agent - Text Interface")

# Initialize session state variables
if 'audioloop_running' not in st.session_state:
    st.session_state.audioloop_running = False
    st.session_state.audioloop_instance = None
    st.session_state.audioloop_thread = None
    # Use asyncio.Queue for interaction with the asyncio-based AudioLoop
    st.session_state.user_text_input_queue = asyncio.Queue()
    st.session_state.model_text_output_queue = asyncio.Queue()
    st.session_state.conversation_history = []
    st.session_state.stop_requested = False


def start_audioloop():
    if not st.session_state.audioloop_running:
        # Re-initialize queues for a fresh start if they were used by a previous run
        st.session_state.user_text_input_queue = asyncio.Queue()
        st.session_state.model_text_output_queue = asyncio.Queue()
        st.session_state.conversation_history = [] # Clear history for new session

        st.session_state.audioloop_instance = AudioLoop(
            user_text_input_queue=st.session_state.user_text_input_queue,
            model_text_output_queue=st.session_state.model_text_output_queue
        )
        
        st.session_state.audioloop_thread = threading.Thread(
            target=run_audioloop_in_thread,
            args=(
                st.session_state.audioloop_instance,
                st.session_state.user_text_input_queue,
                st.session_state.model_text_output_queue
            ),
            daemon=True # Daemon allows Streamlit to exit even if thread is stuck, but we try to join.
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
    
    st.caption("Audio I/O uses server's default devices.")

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
            # No st.rerun() needed here immediately, update will happen via polling or next interaction
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
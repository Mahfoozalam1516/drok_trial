from flask import Flask, render_template, request, jsonify
import json
import os
from deepgram.utils import verboselogs
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    AgentWebSocketEvents,
    SettingsConfigurationOptions,
)
import threading
import queue
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Global variables
active_connections = {}
message_queues = {}

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/start_session', methods=['POST'])
def start_session():
    """Start a new Deepgram voice session"""
    session_id = f"session_{len(active_connections) + 1}"
    
    # Create message queue for this session
    message_queues[session_id] = queue.Queue()
    
    # Start Deepgram in a separate thread
    thread = threading.Thread(target=start_deepgram_session, args=(session_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({"session_id": session_id, "status": "started"})

@app.route('/api/get_messages', methods=['GET'])
def get_messages():
    """Get any new messages for a session"""
    session_id = request.args.get('session_id')
    if not session_id or session_id not in message_queues:
        return jsonify({"error": "Invalid session"}), 400
    
    messages = []
    try:
        while not message_queues[session_id].empty():
            messages.append(message_queues[session_id].get_nowait())
    except queue.Empty:
        pass
    
    return jsonify({"messages": messages})

@app.route('/api/end_session', methods=['POST'])
def end_session():
    """End an active session"""
    session_id = request.json.get('session_id')
    if not session_id or session_id not in active_connections:
        return jsonify({"error": "Invalid session"}), 400
    
    # Close the Deepgram connection
    if active_connections[session_id]:
        active_connections[session_id].finish()
        del active_connections[session_id]
    
    if session_id in message_queues:
        del message_queues[session_id]
    
    return jsonify({"status": "session ended"})

def start_deepgram_session(session_id):
    """Start a Deepgram websocket session"""
    try:
        # Get Deepgram API key from environment variable
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY not found in environment variables")

        # Client configuration
        config = DeepgramClientOptions(
            options={
                "keepalive": "true",
                "microphone_record": "true",
                "speaker_playback": "true",
                "speaker_channels": "2",
                "echo_cancellation": "true",
                "noise_suppression": "true",
                "auto_gain_control": "true",
                "vad_level": "2",
                "barge_in_enabled": "true",
                "microphone_always_on": "true",
            },
        )
        
        # Create Deepgram client with API key from .env
        deepgram = DeepgramClient(api_key, config)
        dg_connection = deepgram.agent.websocket.v("1")
        
        active_connections[session_id] = dg_connection

        def on_open(self, open, **kwargs):
            message_queues[session_id].put({
                "type": "system",
                "message": "Connection established. Speak to begin conversation."
            })

        def on_welcome(self, welcome, **kwargs):
            message_queues[session_id].put({
                "type": "system",
                "message": "Welcome to Deepgram Voice Conversation"
            })

        def on_settings_applied(self, settings_applied, **kwargs):
            message_queues[session_id].put({
                "type": "system",
                "message": "Settings applied. Ready for conversation."
            })

        def on_conversation_text(self, conversation_text, **kwargs):
            try:
                if hasattr(conversation_text, 'role') and hasattr(conversation_text, 'content'):
                    role = conversation_text.role
                    content = conversation_text.content
                    
                    message_queues[session_id].put({
                        "type": role,
                        "message": content
                    })
            except Exception as e:
                message_queues[session_id].put({
                    "type": "error",
                    "message": f"Error handling conversation: {str(e)}"
                })

        def on_user_started_speaking(self, user_started_speaking, **kwargs):
            message_queues[session_id].put({
                "type": "status",
                "message": "listening"
            })

        def on_agent_thinking(self, agent_thinking, **kwargs):
            message_queues[session_id].put({
                "type": "status",
                "message": "thinking"
            })

        def on_agent_started_speaking(self, agent_started_speaking, **kwargs):
            message_queues[session_id].put({
                "type": "status",
                "message": "speaking"
            })

        def on_agent_audio_done(self, agent_audio_done, **kwargs):
            message_queues[session_id].put({
                "type": "status",
                "message": "done_speaking"
            })

        def on_close(self, close, **kwargs):
            message_queues[session_id].put({
                "type": "system",
                "message": "Connection closed"
            })

        def on_error(self, error, **kwargs):
            message_queues[session_id].put({
                "type": "error",
                "message": f"Error: {str(error)}"
            })

        def on_end_of_thought(self, end_of_thought, **kwargs):
            message_queues[session_id].put({
                "type": "status",
                "message": "preparing_response"
            })

        def on_unhandled(self, unhandled, **kwargs):
            pass

        dg_connection.on(AgentWebSocketEvents.Open, on_open)
        dg_connection.on(AgentWebSocketEvents.Welcome, on_welcome)
        dg_connection.on(AgentWebSocketEvents.SettingsApplied, on_settings_applied)
        dg_connection.on(AgentWebSocketEvents.ConversationText, on_conversation_text)
        dg_connection.on(AgentWebSocketEvents.UserStartedSpeaking, on_user_started_speaking)
        dg_connection.on(AgentWebSocketEvents.AgentThinking, on_agent_thinking)
        dg_connection.on(AgentWebSocketEvents.AgentStartedSpeaking, on_agent_started_speaking)
        dg_connection.on(AgentWebSocketEvents.AgentAudioDone, on_agent_audio_done)
        dg_connection.on(AgentWebSocketEvents.Close, on_close)
        dg_connection.on(AgentWebSocketEvents.Error, on_error)
        dg_connection.on(AgentWebSocketEvents.Unhandled, on_unhandled)
        
        try:
            dg_connection.on("EndOfThought", on_end_of_thought)
        except Exception:
            pass

        options = SettingsConfigurationOptions()
        
        options.agent.think.provider.type = "open_ai"
        options.agent.think.model = "gpt-4o-mini"
        options.agent.think.instructions = """
        You are a sales executive at DigiRocket Technologies, specializing in website optimization and digital marketing. You aim to collect the customer's email by offering a free website audit report and scheduling a consultation. The conversation should not end until the user provides their email or explicitly refuses multiple times. Answers should not be more than 30 words. Use more filler words in between the conversation to make the conversation more realistic and natural, and have context awareness. And remember to get the customer email id. Take the location and time zone of Dover, United States.
        """
        
        options.agent.speak.voice = "nova"
        options.agent.speak.rate = 0.95
        
        options.agent.barge_in = True
        
        options.transcription = {
            "model": "nova-2",
            "smart_format": True,
            "diarize": True,
            "language": "en",
            "punctuate": True,
        }

        if dg_connection.start(options) is False:
            message_queues[session_id].put({
                "type": "error",
                "message": "Failed to start connection"
            })
            return

    except Exception as e:
        message_queues[session_id].put({
            "type": "error",
            "message": f"An error occurred: {str(e)}"
        })

if __name__ == "__main__":
    app.run(debug=True)
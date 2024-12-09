from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import subprocess
from logger import debug_logger, info_logger, warning_logger, error_logger, critical_logger
from collections import defaultdict
from agent_registry import AgentRegistry
from tools_registry import ToolRegistry
from werkzeug.utils import secure_filename
import requests
from urllib.parse import urlparse

# Register your agents
from tools.ai_runners.live_caption.main import LiveCaptionAgent
from tools.ai_runners.transcribe_audio.main import TranscribeAudioAgent
from tools.ai_runners.speaker_diarization.main import SpeakerDiarizationAgent

AgentRegistry.register(LiveCaptionAgent)
AgentRegistry.register(TranscribeAudioAgent)
AgentRegistry.register(SpeakerDiarizationAgent)  # Add this line

app = Flask(__name__)
app.config['DEBUG'] = True

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def discover_scripts():
    script_tree = defaultdict(list)
    debug_logger.debug("Starting script discovery")
    
    for root, dirs, files in os.walk('tools'):
        for file in files:
            if not file.endswith('.py') or file == '__init__.py':
                continue
            script_path = os.path.relpath(os.path.join(root, file)).replace('\\', '/')
            category = script_path.split('/')[1]
            script_tree[category].append(script_path)
            debug_logger.debug(f"Found script: {script_path}")
    return script_tree

def get_agent_metadata():
    agents = {}
    for agent_name, agent_class in AgentRegistry.get_all_agents().items():
        agent_instance = agent_class()
        agents[agent_name] = agent_instance.get_metadata()
    return agents

def get_tool_metadata():
    tools = {}
    for tool_name, tool_class in ToolRegistry.get_all_tools().items():
        tool_instance = tool_class()
        tools[tool_name] = tool_instance.get_metadata()
    return tools

@app.route('/')
def index():
    script_tree = discover_scripts()
    if not script_tree:
        warning_logger.warning("No scripts found")
        return "<pre>No scripts found</pre>"

    agents = get_agent_metadata()
    tools = get_tool_metadata()
    try:
        return render_template('index.html', 
                             script_tree=script_tree, 
                             agents=agents,
                             tools=tools)
    except Exception as e:
        error_logger.error(f"Template error: {str(e)}")
        return f"<pre>Error: {str(e)}</pre>"

def validate_script(script):
    script = script.replace('\\', '/').replace('//', '/')
    if not script.endswith('.py'):
        return False, "Not a Python script"
    if not os.path.exists(script):
        return False, "Script not found"
    return True, ""

@app.route('/run/<path:script>')
def run_script(script):
    is_valid, error = validate_script(script)
    if not is_valid:
        return error

    try:
        result = subprocess.run(['python', script], capture_output=True, text=True)
        return f"<pre>{result.stdout}</pre>"
    except Exception as e:
        error_logger.error(f"Script error: {str(e)}")
        return f"<pre>Error: {str(e)}</pre>"

@app.route('/run_agent/<agent_name>', methods=['POST'])
def run_agent(agent_name):
    debug_logger.debug(f"Attempting to run agent: {agent_name}")
    
    agent_class = AgentRegistry.get_agent(agent_name)
    if not agent_class:
        warning_logger.warning(f"Agent not found: {agent_name}")
        return "Agent not found", 404
        
    try:
        agent = agent_class()
        result = agent.run()
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        error_logger.error(f"Failed to run agent {agent_name}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/run_tool/<tool_name>', methods=['POST'])
def run_tool(tool_name):
    debug_logger.debug(f"Attempting to run tool: {tool_name}")
    
    tool_class = ToolRegistry.get_tool(tool_name)
    if not tool_class:
        warning_logger.warning(f"Tool not found: {tool_name}")
        return "Tool not found", 404
        
    try:
        tool = tool_class()
        result = tool.execute(request.json)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        error_logger.error(f"Failed to run tool {tool_name}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/diarize', methods=['POST'])
def diarize_audio():
    debug_logger.debug("Received diarization request")
    
    try:
        if 'audio' not in request.files:
            debug_logger.error("No file in request")
            return jsonify({"status": "error", "message": "No file uploaded"}), 400
        
        file = request.files['audio']
        if file.filename == '':
            debug_logger.error("Empty filename")
            return jsonify({"status": "error", "message": "No file selected"}), 400
            
        if not allowed_file(file.filename):
            debug_logger.error(f"Invalid file type: {file.filename}")
            return jsonify({
                "status": "error", 
                "message": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
            
        try:
            # Use absolute paths
            abs_upload_folder = os.path.abspath(app.config['UPLOAD_FOLDER'])
            filename = secure_filename(file.filename)
            filepath = os.path.join(abs_upload_folder, filename)
            
            debug_logger.debug(f"Saving uploaded file to {filepath}")
            
            # Ensure upload directory exists
            os.makedirs(abs_upload_folder, exist_ok=True)
            
            # Save file
            file.save(filepath)
            
            if not os.path.exists(filepath):
                return jsonify({"status": "error", "message": "Failed to save file"}), 500
            
            debug_logger.debug(f"File saved successfully at {filepath}")
            
            # Process with absolute path
            agent = SpeakerDiarizationAgent()
            debug_logger.debug(f"Starting diarization process on {filepath}")
            result = agent.run({"audio_path": filepath})
            
            # Cleanup
            if os.path.exists(filepath):
                os.remove(filepath)
                debug_logger.debug("Cleaned up temporary file")
            
            return jsonify(result)
            
        except Exception as e:
            if 'filepath' in locals() and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            error_logger.error(f"File handling error: {str(e)}")
            return jsonify({"status": "error", "message": f"File handling error: {str(e)}"}), 500
            
    except Exception as e:
        error_logger.error(f"Diarization error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Simplified view_agent route - only returns the form
@app.route('/view/agent/<agent_name>')
def view_agent(agent_name):
    agent_class = AgentRegistry.get_agent(agent_name)
    if not agent_class:
        return "Agent not found", 404
        
    agent = agent_class()
    metadata = agent.get_metadata()
    
    template = "diarization_tool.html" if agent_name == "SpeakerDiarizationAgent" else "base_tool.html"
    return render_template(template, title=metadata['name'], description=metadata['description'])

@app.route('/view/script/<path:script>')
def view_script(script):
    is_valid, error = validate_script(script)
    if not is_valid:
        return error
        
    script_name = script.split('/')[-2].replace('_', ' ').title()
    return render_template("base_tool.html",
                         title=script_name,
                         description=f"Script: {script}")

if __name__ == '__main__':
    info_logger.info("Starting Flask application")
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import subprocess
from logger import debug_logger, info_logger, warning_logger, error_logger, critical_logger
from collections import defaultdict
from agent_registry import AgentRegistry
from tools_registry import ToolRegistry

# Register your agents
from tools.ai_runners.live_caption.main import LiveCaptionAgent
from tools.ai_runners.transcribe_audio.main import TranscribeAudioAgent

AgentRegistry.register(LiveCaptionAgent)
AgentRegistry.register(TranscribeAudioAgent)

app = Flask(__name__)
app.config['DEBUG'] = True

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

if __name__ == '__main__':
    info_logger.info("Starting Flask application")
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for
import os
import subprocess
from logger import debug_logger, info_logger, warning_logger, error_logger, critical_logger
from collections import defaultdict

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def index():
    script_tree = defaultdict(list)
    debug_logger.debug("Starting script discovery")
    
    for root, dirs, files in os.walk('tools'):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                script_path = os.path.relpath(os.path.join(root, file))
                # Normalize path separators to forward slashes for URLs
                script_path = script_path.replace('\\', '/')
                category = script_path.split('/')[1]  # Get category (file_management/ai_runners)
                script_tree[category].append(script_path)
                debug_logger.debug(f"Found script: {script_path}")
                info_logger.info(f"Added script {script_path} to category {category}")
    
    if not script_tree:
        warning_logger.warning("No scripts found in tools directory")
    
    debug_logger.debug(f"Final script tree: {dict(script_tree)}")
    
    try:
        info_logger.info("Rendering template 'index.html'")
        return render_template('index.html', script_tree=script_tree)
    except Exception as e:
        error_logger.error(f"Failed to render template: {str(e)}")
        critical_logger.critical(f"Application may be unstable: {str(e)}")
        return f"<pre>Error rendering template: {str(e)}</pre>"

@app.route('/run/<path:script>')
def run_script(script):
    debug_logger.debug(f"Attempting to run script: {script}")
    
    # Normalize path separators
    script = script.replace('\\', '/').replace('//', '/')
    
    if not script.endswith('.py'):
        warning_logger.warning(f"Attempted to run non-Python file: {script}")
        return "Not a Python script"
        
    if not os.path.exists(script):
        warning_logger.warning(f"Attempted to run non-existent script: {script}")
        return "Script not found"
        
    try:
        info_logger.info(f"Running script: {script}")
        result = subprocess.run(['python', script], capture_output=True, text=True)
        
        if result.stderr:
            warning_logger.warning(f"Script {script} produced errors: {result.stderr}")
            
        debug_logger.debug(f"Script output: {result.stdout}")
        info_logger.info(f"Successfully ran script: {script}")
        return f"<pre>{result.stdout}</pre>"
    except Exception as e:
        error_logger.error(f"Failed to run script {script}: {str(e)}")
        return f"<pre>Error running script: {str(e)}</pre>"

if __name__ == '__main__':
    info_logger.info("Starting Flask application")
    app.run(debug=True)

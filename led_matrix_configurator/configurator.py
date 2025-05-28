from flask import Flask, render_template, request, jsonify
import json
import os
import subprocess
from threading import Thread

# Flask app setup
app = Flask(__name__)

# Determine the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths compatible with both Linux and Windows
PARENT_DIR = os.path.join(BASE_DIR, '..')  # Move up one directory
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
LOG_PATH = os.path.join(PARENT_DIR, "app.log")
SCRIPT_PATH = os.path.join(PARENT_DIR, "GlucoseMatrixDisplay.py")

# Function to run the Python script
def run_script():
    with open(LOG_PATH, "a") as log_file:
        process = subprocess.Popen(
            ["python", SCRIPT_PATH],
            stdout=log_file,
            stderr=log_file,
        )
        process.wait()

@app.route("/run", methods=["POST"])
def run_script():
    with open(LOG_PATH, "a") as log_file:
        commands = [
            "sudo git stash",
            "sudo git pull",
            "sudo git stash pop",
            "sudo systemctl restart glucose_matrix.service"
        ]
        for command in commands:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=log_file,
                stderr=log_file,
            )
            process.wait()



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/config", methods=["GET"])
def get_config():
    try:
        with open(CONFIG_PATH, "r") as file:
            config_data = json.load(file)
        return jsonify({"config": config_data})
    except FileNotFoundError:
        return jsonify({"error": f"Config file not found at {CONFIG_PATH}"}), 404

# Save the configuration and trigger the script
@app.route("/save", methods=["POST"])
def save_config():
    try:
        config_data = request.get_json()
        # Convert numeric strings to actual numbers
        for key, value in config_data.items():
            if value.isdigit():
                config_data[key] = int(value)
            else:
                try:
                    config_data[key] = float(value)
                except ValueError:
                    pass
        with open(CONFIG_PATH, "w") as config_file:
            json.dump(config_data, config_file, indent=4)
        return jsonify({"message": "Config saved successfully!"})
    except Exception as e:
        return jsonify({"message": str(e)}), 500
    
@app.route("/restart-service", methods=["POST"])
def restart_service():
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "restart", "glucose_matrix.service"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return jsonify({"message": "Service restarted successfully!"})
        else:
            return jsonify({"message": result.stderr}), 500
    except Exception as e:
        return jsonify({"message": str(e)}), 500
    
# Get the logs as JSON
@app.route("/logs", methods=["GET"])
def get_logs():
    try:
        max_lines = 100  # Define the maximum number of lines to show in the logs
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r") as file:
                # Read only the last `max_lines` lines from the file
                lines = file.readlines()[-max_lines:]
                logs = "".join(lines)
        else:
            logs = "No logs available yet."
        return jsonify({"logs": logs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Perform a Git pull
@app.route("/git-pull", methods=["POST"])
def git_pull():
    try:
        result = subprocess.run(
            ["git", "pull"],
            cwd=os.path.dirname(SCRIPT_PATH),
            capture_output=True,
            text=True,
        )
        return jsonify({"status": "success", "output": result.stdout})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Handle errors globally
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

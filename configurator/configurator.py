from flask import Flask, render_template, request, jsonify
import json
import os
import subprocess
import sys
from threading import Thread

try:
    from configurator.config_loader import load_config, ensure_config_exists
    from configurator.config_model import ConfigModel
except Exception:
    load_config = None
    ensure_config_exists = None
    ConfigModel = None

# Flask app setup
app = Flask(__name__)

# Determine the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths compatible with both Linux and Windows
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))  # Move up one directory
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
LOG_PATH = os.path.join(PARENT_DIR, "app.log")
# Use the main script that exists in the repository root
SCRIPT_PATH = os.path.join(PARENT_DIR, "GlucosePixelMatrix.py")
CONFIG_SCHEMA_PATH = os.path.join(BASE_DIR, "config.schema.json")
EXAMPLE_CONFIG_PATH = os.path.join(BASE_DIR, "config.example.json")

# If the config file doesn't exist, create it from the example so first-run works
if not os.path.exists(CONFIG_PATH):
    try:
        with open(EXAMPLE_CONFIG_PATH, 'r', encoding='utf-8') as src, open(CONFIG_PATH, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
        app.logger.info(f"Created initial config from example at {CONFIG_PATH}")
    except Exception as e:
        app.logger.error(f"Failed to create initial config from example: {e}")

# Function to run the Python script
def run_main_script():
    """Run the main Glucose display script in background and log output."""
    with open(LOG_PATH, "a") as log_file:
        process = subprocess.Popen(
            [sys.executable, SCRIPT_PATH],
            stdout=log_file,
            stderr=log_file,
        )
        process.wait()

@app.route("/run", methods=["POST"])
def run_script():
    """Endpoint to pull latest code and (optionally) restart service.

    On Linux systems the endpoint will attempt to run systemctl restart; on
    Windows it will only perform the git operations.
    """
    with open(LOG_PATH, "a") as log_file:
        is_windows = os.name == 'nt'
        if is_windows:
            commands = [
                "git stash",
                "git pull",
                "git stash pop",
            ]
        else:
            commands = [
                "sudo git stash",
                "sudo git pull",
                "sudo git stash pop",
                "sudo systemctl restart glucose_matrix.service",
            ]

        for command in commands:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=PARENT_DIR,
                stdout=log_file,
                stderr=log_file,
            )
            process.wait()

    return jsonify({"message": "run commands executed"})



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/schema", methods=["GET"])
def get_schema():
    try:
        with open(CONFIG_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        return jsonify(schema)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/config", methods=["GET"])
def get_config():
    try:
        if load_config is not None:
            try:
                cfg = load_config()
                return jsonify({"config": json.loads(cfg.json(by_alias=True))})
            except Exception as e:
                app.logger.error(f"Typed config load failed: {e}")

        # Fallback to file reads
        if not os.path.exists(CONFIG_PATH) and os.path.exists(EXAMPLE_CONFIG_PATH):
            with open(EXAMPLE_CONFIG_PATH, "r", encoding="utf-8") as file:
                config_data = json.load(file)
            return jsonify({"config": config_data})

        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
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
        for key, value in list(config_data.items()):
            # Only attempt conversions for string values coming from form submissions
            if isinstance(value, str):
                if value.isdigit():
                    config_data[key] = int(value)
                    continue
                try:
                    # Try int first, then float
                    as_int = int(value)
                    config_data[key] = as_int
                    continue
                except Exception:
                    pass
                try:
                    as_float = float(value)
                    config_data[key] = as_float
                    continue
                except Exception:
                    # Leave as string (could be boolean like 'true' or 'false')
                    lowered = value.strip().lower()
                    if lowered in ("true", "false"):
                        config_data[key] = lowered == "true"
                    else:
                        config_data[key] = value
            else:
                # non-string values are left as-is
                config_data[key] = value
        if ConfigModel is not None:
            try:
                from pydantic import ValidationError
            except Exception:
                ValidationError = None

            try:
                validated = ConfigModel.parse_obj(config_data)
                # write canonical JSON using schema aliases
                with open(CONFIG_PATH, "w", encoding="utf-8") as config_file:
                    config_file.write(validated.json(by_alias=True, indent=4))
                return jsonify({"message": "Config saved successfully!"})
            except Exception as e:
                if ValidationError is not None and isinstance(e, ValidationError):
                    errs = e.errors()
                    return jsonify({"errors": errs}), 400
                return jsonify({"error": str(e)}), 400

        # Fallback to existing jsonschema validation if present
        if os.path.exists(CONFIG_SCHEMA_PATH):
            try:
                from jsonschema import Draft7Validator
                with open(CONFIG_SCHEMA_PATH, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                validator = Draft7Validator(schema)
                errors = sorted(validator.iter_errors(config_data), key=lambda e: e.path)
                if errors:
                    err_list = []
                    for e in errors:
                        path = ".".join(str(p) for p in e.path) if e.path else "<root>"
                        err_list.append({"path": path, "message": e.message})
                    return jsonify({"errors": err_list}), 400
            except Exception as e:
                return jsonify({"error": f"Schema validation failed: {e}"}), 500

        with open(CONFIG_PATH, "w", encoding="utf-8") as config_file:
            json.dump(config_data, config_file, indent=4)
        return jsonify({"message": "Config saved successfully!"})
    except Exception as e:
        return jsonify({"message": str(e)}), 500
    
@app.route("/restart-service", methods=["POST"])
def restart_service():
    try:
        if os.name == 'nt':
            # On Windows there's no systemctl; attempt to restart via a script if present
            result = subprocess.run(["echo", "restart-not-supported-on-windows"], capture_output=True, text=True, shell=True)
        else:
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

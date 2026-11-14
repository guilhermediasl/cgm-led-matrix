from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from threading import Thread

# Allow `python configurator/configurator.py` as well as `python -m configurator.configurator`:
# without the project root on sys.path the typed config imports below fail silently.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from configurator.config_loader import load_config, ensure_config_exists
    from configurator.config_model import ConfigModel
except Exception:
    load_config = None
    ensure_config_exists = None
    ConfigModel = None

# Flask app setup
app = Flask(__name__)
# Keep the schema's property order: the page lays out fields in that order.
app.json.sort_keys = False

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
TOKEN_MASK = "********"
SERVICE_NAME = "glucose_matrix.service"
# The display app writes every frame here (relative to its working directory).
PREVIEW_PATH = os.path.join(PARENT_DIR, "temp", "output_image.png")

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

def _read_saved_config() -> dict:
    """The config currently on disk (or the example on first run), as raw JSON."""
    path = CONFIG_PATH if os.path.exists(CONFIG_PATH) else EXAMPLE_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _saved_token() -> str:
    try:
        return _read_saved_config().get("token") or ""
    except Exception:
        return ""


def _schema_types() -> dict:
    try:
        with open(CONFIG_SCHEMA_PATH, "r", encoding="utf-8") as f:
            props = json.load(f).get("properties", {})
        return {key: spec.get("type") for key, spec in props.items()}
    except Exception:
        return {}


def _coerce_form_values(config_data: dict) -> dict:
    """Turn form strings into numbers/booleans, but only for keys the schema
    types that way, so a numeric token or MAC address stays a string."""
    types = _schema_types()
    for key, value in list(config_data.items()):
        if not isinstance(value, str):
            continue
        kind = types.get(key)
        stripped = value.strip()
        try:
            if kind == "integer":
                config_data[key] = int(stripped) if stripped else None
            elif kind == "number":
                config_data[key] = float(stripped) if stripped else None
            elif kind == "boolean":
                config_data[key] = stripped.lower() == "true"
        except ValueError:
            pass  # leave it for validation to report
    return config_data


@app.route("/config", methods=["GET"])
def get_config():
    try:
        config_data = None
        if load_config is not None:
            try:
                config_data = load_config().model_dump(mode="json", by_alias=True)
            except Exception as e:
                app.logger.error(f"Typed config load failed: {e}")
        if config_data is None:
            config_data = _read_saved_config()
        # Never send the Nightscout token to the browser; /save keeps the stored
        # one when the mask comes back unchanged.
        if config_data.get("token"):
            config_data["token"] = TOKEN_MASK
        return jsonify({"config": config_data})
    except FileNotFoundError:
        return jsonify({"error": f"Config file not found at {CONFIG_PATH}"}), 404


# Save the configuration and trigger the script
@app.route("/save", methods=["POST"])
def save_config():
    try:
        config_data = _coerce_form_values(request.get_json() or {})
        if config_data.get("token") == TOKEN_MASK:
            config_data["token"] = _saved_token()
        if ConfigModel is not None:
            from pydantic import ValidationError
            try:
                validated = ConfigModel.model_validate(config_data)
            except ValidationError as e:
                errs = [{"path": ".".join(str(p) for p in err["loc"]), "message": err["msg"]}
                        for err in e.errors()]
                return jsonify({"errors": errs}), 400
            # write canonical JSON using schema aliases
            with open(CONFIG_PATH, "w", encoding="utf-8") as config_file:
                config_file.write(validated.model_dump_json(by_alias=True, indent=4))
            return jsonify({"message": "Config saved successfully!"})

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
            return jsonify({"message": "Restarting needs systemd, so it only works on Linux. "
                                       "Stop and start GlucosePixelMatrix.py yourself."}), 501
        else:
            result = subprocess.run(
                ["sudo", "systemctl", "restart", SERVICE_NAME],
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
        max_lines = max(10, min(request.args.get("lines", 200, type=int), 1000))
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r") as file:
                # Read only the last `max_lines` lines from the file
                lines = file.readlines()[-max_lines:]
                logs = "".join(lines)
            # Older log lines may still hold the token inside request URLs.
            token = _saved_token()
            if token:
                logs = logs.replace(token, "***")
        else:
            logs = "No logs available yet."
        return jsonify({"logs": logs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def _iso_mtime(path):
    if not os.path.exists(path):
        return None
    return datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).isoformat()


def _service_state() -> str:
    if os.name == 'nt':
        return "unsupported"
    try:
        result = subprocess.run(["systemctl", "is-active", SERVICE_NAME],
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@app.route("/status", methods=["GET"])
def get_status():
    """Display service state and when the last frame and the config were written."""
    return jsonify({
        "service": _service_state(),
        "service_name": SERVICE_NAME,
        "preview_updated": _iso_mtime(PREVIEW_PATH),
        "config_updated": _iso_mtime(CONFIG_PATH),
        "now": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/preview", methods=["GET"])
def get_preview():
    """The last frame the display app rendered (32x32 PNG)."""
    if not os.path.exists(PREVIEW_PATH):
        return jsonify({"error": "No frame rendered yet"}), 404
    response = send_file(PREVIEW_PATH, mimetype="image/png", max_age=0)
    response.headers["Cache-Control"] = "no-store"
    return response


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
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            return jsonify({"status": "error", "output": output}), 500
        return jsonify({"status": "success", "output": output})
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

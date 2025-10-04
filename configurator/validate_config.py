"""Validate `config.json` against `config.schema.json` and print friendly errors.

Usage:
    python led_matrix_configurator/validate_config.py [path/to/config.json]

If no path is provided the script looks for `led_matrix_configurator/config.json`.
"""
import json
import os
import sys

try:
    from jsonschema import Draft7Validator
except Exception as e:
    print("jsonschema is required. Install with: pip install jsonschema")
    sys.exit(2)

HERE = os.path.dirname(__file__)
SCHEMA_PATH = os.path.join(HERE, "config.schema.json")
DEFAULT_CONFIG_PATH = os.path.join(HERE, "config.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(config_path=None):
    config_path = config_path or DEFAULT_CONFIG_PATH

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return 2

    if not os.path.exists(SCHEMA_PATH):
        print(f"Schema file not found: {SCHEMA_PATH}")
        return 2

    try:
        config = load_json(config_path)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in config file: {e}")
        return 2

    schema = load_json(SCHEMA_PATH)
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda e: e.path)

    if not errors:
        print("Config is valid ✅")
        return 0

    print("Config validation errors:")
    for e in errors:
        path = ".".join(str(p) for p in e.path) if e.path else "<root>"
        print(f"- {path}: {e.message}")

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))

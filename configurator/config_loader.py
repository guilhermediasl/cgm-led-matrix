import json
import os
from .config_model import ConfigModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')
EXAMPLE_PATH = os.path.join(BASE_DIR, 'config.example.json')


def ensure_config_exists():
    if not os.path.exists(CONFIG_PATH) and os.path.exists(EXAMPLE_PATH):
        with open(EXAMPLE_PATH, 'r', encoding='utf-8') as src, open(CONFIG_PATH, 'w', encoding='utf-8') as dst:
            dst.write(src.read())


def load_config() -> ConfigModel:
    ensure_config_exists()
    path = CONFIG_PATH if os.path.exists(CONFIG_PATH) else EXAMPLE_PATH
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Pydantic will validate types and provide defaults
    # Use by_field_name to allow aliases like 'output type'
    cfg = ConfigModel.parse_obj(raw)
    return cfg

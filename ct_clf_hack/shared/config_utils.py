from ct_clf_hack.shared.path_utils import configs_dpath
import yaml

_configs_names = {
    "models": "models.yaml",
    "class_map": "class_map.yaml",
}

def open_config(config_name: str) -> dict:
    if config_name not in _configs_names:
        raise ValueError(f"Config '{config_name}' not found. Available configs: {list(_configs_names.keys())}")

    config_path = configs_dpath() / _configs_names[config_name]
    if not config_path.exists():
        raise FileNotFoundError(f"Config file '{config_path}' does not exist.")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def models_config() -> dict:
    return open_config("models")

def class_map() -> dict:
    return open_config("class_map")
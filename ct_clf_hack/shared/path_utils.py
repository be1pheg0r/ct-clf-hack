from pathlib import Path

def root_dpath() -> Path:
    current_dir = Path(__file__).resolve().parent
    while current_dir / "pyproject.toml" not in current_dir.iterdir():
        if current_dir.parent == current_dir:
            raise FileNotFoundError("Could not find the project root directory.")
        current_dir = current_dir.parent
    return current_dir

def data_dpath() -> Path:
    return root_dpath() / "data"

def checkpoints_dpath() -> Path:
    return root_dpath() / "checkpoints"

def configs_dpath() -> Path:
    return root_dpath() / "configs"

def backend_dpath() -> Path:
    return root_dpath() / "ct_clf_backend"

def backend_configs_dpath() -> Path:
    return backend_dpath() / "configs"
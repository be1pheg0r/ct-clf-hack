"""
Модуль для управления путями в проекте.
"""

from pathlib import Path

def root_dpath() -> Path:
    """Возвращает корневую директорию проекта."""
    current_dir = Path(__file__).resolve().parent
    while current_dir / "pyproject.toml" not in current_dir.iterdir():
        if current_dir.parent == current_dir:
            raise FileNotFoundError("Could not find the project root directory.")
        current_dir = current_dir.parent
    return current_dir

def data_dpath() -> Path:
    """Возвращает путь к директории с данными."""
    return root_dpath() / "data"

def checkpoints_dpath() -> Path:
    """Возвращает путь к директории с контрольными точками моделей."""
    return root_dpath() / "checkpoints"

def configs_dpath() -> Path:
    """Возвращает путь к директории с конфигурационными файлами."""
    return root_dpath() / "configs"

def backend_dpath() -> Path:
    """Возвращает путь к директории ct_clf_backend."""
    return root_dpath() / "ct_clf_backend"

def test_data_dpath() -> Path:
    """Возвращает путь к директории с тестовыми данными."""
    return root_dpath()  / "test_data"
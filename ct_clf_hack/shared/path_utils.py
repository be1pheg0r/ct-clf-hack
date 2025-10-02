"""
Утилиты для работы с путями проекта.
"""

import os
from pathlib import Path
from typing import Optional


def root_dpath() -> Path:
    """
    Возвращает путь к корневой директории проекта.

    Returns:
        Path: Путь к корневой директории проекта

    Raises:
        FileNotFoundError: Если корневая директория не найдена
    """
    # Проверяем переменную окружения PROJECT_ROOT (для Docker)
    if "PROJECT_ROOT" in os.environ:
        project_root = Path(os.environ["PROJECT_ROOT"])
        if project_root.exists():
            return project_root

    # Начи��аем поиск с текущего файла
    current_path = Path(__file__).resolve().parent

    # Поднимаемся вверх по директориям, ища маркеры проекта
    markers = ["pyproject.toml", ".project_root", ".git", "setup.py", "requirements.txt"]

    for _ in range(10):  # Ограничиваем поиск 10 уровнями
        for marker in markers:
            if (current_path / marker).exists():
                return current_path

        # Если достигли корня файловой системы
        if current_path.parent == current_path:
            break

        current_path = current_path.parent

    # Если ничего не найдено, используем /app для Docker или текущую директорию
    if Path("/app").exists() and any((Path("/app") / marker).exists() for marker in markers):
        return Path("/app")

    # В крайнем случае используем директорию, где находится этот файл
    return Path(__file__).resolve().parent.parent.parent


def configs_dpath() -> Path:
    """
    Возвращает путь к директории с конфигурационными файлами.

    Returns:
        Path: Путь к директории configs
    """
    return root_dpath() / "configs"


def data_dpath() -> Path:
    """
    Возвращает путь к директории с данными.

    Returns:
        Path: Путь к дире��тории data
    """
    return root_dpath() / "data"


def checkpoints_dpath() -> Path:
    """
    Возвращает путь к директории с чекпойнтами моделей.

    Returns:
        Path: Путь к директории checkpoints
    """
    return root_dpath() / "checkpoints"


def logs_dpath() -> Path:
    """
    Возвращает путь к директории с логами.

    Returns:
        Path: Путь к директории logs
    """
    return root_dpath() / "logs"


def ensure_dir(path: Path) -> Path:
    """
    Создает директорию, если она не существует.

    Args:
        path: Путь к директории

    Returns:
        Path: Путь к созданной директории
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

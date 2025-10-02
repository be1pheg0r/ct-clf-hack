"""
Модуль для работы с кэшем и обработки загруженных файлов.
Содержит функции для создания временных директорий, обработки ZIP архивов,
извлечения метаданных из DICOM файлов.
"""

import shutil
import tempfile
import uuid
import zipfile
import random
from pathlib import Path
from typing import List, Tuple, Optional
from io import BytesIO
import pydicom


def create_cache_dir() -> Path:
    """
    Создает временную директорию для кэша.

    Returns:
        Path: Путь к созданной временной директории
    """
    cache_dir = Path(tempfile.mkdtemp(prefix="ct_clf_cache_"))
    return cache_dir


def save_zip_to_cache(zip_bytes: bytes, cache_dir: Path) -> Path:
    """
    Сохраняет ZIP архив в кэш директорию.

    Args:
        zip_bytes: Байты ZIP архива
        cache_dir: Путь к директории кэша

    Returns:
        Path: Путь к сохраненному ZIP файлу
    """
    zip_filename = f"uploaded_{uuid.uuid4().hex}.zip"
    zip_path = cache_dir / zip_filename

    with open(zip_path, 'wb') as f:
        f.write(zip_bytes)

    return zip_path


def extract_zip_to_cache(zip_path: Path, cache_dir: Path) -> Path:
    """
    Извлекает содержимое ZIP архива в кэш директорию.

    Args:
        zip_path: Путь к ZIP файлу
        cache_dir: Путь к директории кэша

    Returns:
        Path: Путь к директории с извлеченными файлами
    """
    extract_dir = cache_dir / f"extracted_{uuid.uuid4().hex}"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    return extract_dir


def cleanup_cache(cache_dir: Path) -> None:
    """
    Очищает кэш директорию.

    Args:
        cache_dir: Путь к директории кэша для удаления
    """
    try:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
    except Exception as e:
        print(f"Warning: Failed to cleanup cache directory {cache_dir}: {e}")


def process_uploaded_file(file_bytes: bytes, max_files: int = 10) -> Tuple[List[Path], Path]:
    """
    Обрабатывает загруженный файл (ZIP архив или одиночный DICOM файл).

    Args:
        file_bytes: Байты загруженного файла
        max_files: Максимальное количество файлов для выборки

    Returns:
        Tuple[List[Path], Path]: Список путей к выбранным DICOM файлам и путь к кэш директории

    Raises:
        Exception: При ошибке обработки файла
    """
    from ct_clf_hack.shared.file_utils import find_dicom_files, is_probable_dicom_file

    cache_dir = create_cache_dir()

    try:
        if zipfile.is_zipfile(BytesIO(file_bytes)):
            zip_path = save_zip_to_cache(file_bytes, cache_dir)
            extract_dir = extract_zip_to_cache(zip_path, cache_dir)
            dicom_files = find_dicom_files(extract_dir)
        else:
            single_file_path = cache_dir / f"single_file_{uuid.uuid4().hex}"
            with open(single_file_path, 'wb') as f:
                f.write(file_bytes)

            if is_probable_dicom_file(single_file_path):
                dicom_files = [single_file_path]
            else:
                dicom_files = []

        selected_files = random.sample(dicom_files, min(len(dicom_files), max_files))
        return selected_files, cache_dir

    except Exception as e:
        cleanup_cache(cache_dir)
        raise e


def extract_dicom_metadata(file_paths: List[Path]) -> Tuple[Optional[str], Optional[str]]:
    """
    Извлекает метаданные StudyInstanceUID и SeriesInstanceUID из DICOM файлов.

    Args:
        file_paths: Список путей к DICOM файлам

    Returns:
        Tuple[Optional[str], Optional[str]]: StudyInstanceUID и SeriesInstanceUID
    """
    study_uid = None
    series_uid = None

    for file_path in file_paths:
        try:
            ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)

            if not study_uid and hasattr(ds, "StudyInstanceUID"):
                study_uid = str(ds.StudyInstanceUID)

            if not series_uid and hasattr(ds, "SeriesInstanceUID"):
                series_uid = str(ds.SeriesInstanceUID)

            if study_uid and series_uid:
                break

        except Exception:
            continue

    return study_uid, series_uid

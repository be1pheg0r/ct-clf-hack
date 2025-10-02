"""
Модуль для подключения к модели машинного обучения.
Содержит функции для обработки DICOM файлов и взаимодействия с моделью классификации.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).parent.parent.parent))

from ct_clf_backend.app.cache_utils import (cleanup_cache,
                                            extract_dicom_metadata,
                                            process_uploaded_file)
from ct_clf_hack.pipelines import base_process
from ct_clf_hack.shared.data_utils import (prepare_images_for_model,
                                           process_uploaded_dicom_zip)


def ConnectingLinkWithModel(dicom_file_paths: List[Path]) -> Dict[str, Any]:
    """
    Обрабатывает список DICOM файлов с помощью модели.

    Args:
        dicom_file_paths: Список путей к DICOM файлам

    Returns:
        Dict[str, Any]: Результаты анализа модели
    """
    t0 = time.time()

    if len(dicom_file_paths) == 0:
        path_to_study = "no_files"
    elif len(dicom_file_paths) == 1:
        path_to_study = "single_dicom"
    else:
        path_to_study = "zip_archive"

    study_uid, series_uid = extract_dicom_metadata(dicom_file_paths)

    if len(dicom_file_paths) > 0:
        results = base_process(dpath=Path(dicom_file_paths[0]).parent)
        probability = results.get("probability_of_pathology", 0.0)
        pathology = results.get("pathology", 0)
    else:
        probability = 0.0
        pathology = 0

    model_processing_time = round(time.time() - t0, 3)

    return {
        "path_to_study": path_to_study,
        "study_uid": study_uid or "unknown",
        "series_uid": series_uid or "unknown",
        "probability_of_pathology": float(probability),
        "pathology": int(pathology),
        "model_processing_time": float(model_processing_time),
    }


def ConnectingLinkWithModelFromZip(zip_bytes: bytes, max_slices: int = 10) -> Dict[str, Any]:
    """
    Обрабатывает ZIP архив с DICOM файлами с помощью модели.

    Args:
        zip_bytes: Байты ZIP архива
        max_slices: Максимальное количество срезов для обработки

    Returns:
        Dict[str, Any]: Результаты анализа модели
    """
    t0 = time.time()

    try:
        study_uid = "unknown"
        series_uid = "unknown"

        try:
            selected_dicom_files, temp_cache_dir = process_uploaded_file(zip_bytes, max_files=10)
            if selected_dicom_files:
                study_uid_temp, series_uid_temp = extract_dicom_metadata(selected_dicom_files)
                if study_uid_temp:
                    study_uid = study_uid_temp
                if series_uid_temp:
                    series_uid = series_uid_temp
            cleanup_cache(temp_cache_dir)
        except Exception:
            pass

        images, status = process_uploaded_dicom_zip(zip_bytes, max_slices)

        if not images:
            return {
                "path_to_study": "zip_archive",
                "study_uid": study_uid,
                "series_uid": series_uid,
                "probability_of_pathology": 0.0,
                "pathology": 0,
                "model_processing_time": round(time.time() - t0, 3),
                "error": status
            }

        rgb_images = prepare_images_for_model(images)

        result = base_process(images=rgb_images)
        probability = result.get("probability_of_pathology", 0.0)
        pathology = result.get("pathology", 0)
        path_to_study = result.get("path_to_study", "zip_archive")

        model_processing_time = round(time.time() - t0, 3)

        return {
            "path_to_study": path_to_study,
            "study_uid": study_uid,
            "series_uid": series_uid,
            "probability_of_pathology": float(probability),
            "pathology": int(pathology),
            "model_processing_time": float(model_processing_time),
            "processed_slices": len(images),
            "status": status
        }

    except Exception as e:
        return {
            "path_to_study": "zip_archive",
            "study_uid": "unknown",
            "series_uid": "unknown",
            "probability_of_pathology": 0.0,
            "pathology": 0,
            "model_processing_time": round(time.time() - t0, 3),
            "error": f"Processing error: {str(e)}"
        }

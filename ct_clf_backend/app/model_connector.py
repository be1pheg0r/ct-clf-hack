# backend/app/model_connector.py
import time
from typing import List, Tuple, Dict, Any
from io import BytesIO
import pydicom


def ConnectingLinkWithModel(dicom_files: List[Tuple[str, bytes]]) -> Dict[str, Any]:
    """
    Заглушка "связки с моделью", улучшенная.
    Принимает список кортежей (filename, raw_bytes) — это все кандидаты DICOM файлов.
    Возвращает словарь с ключами:
      - path_to_study: str ("uploaded_zip", "single_dicom", or filename-based)
      - study_uid: str
      - series_uid: str
      - probability_of_pathology: float
      - pathology: int (0 or 1)
      - model_processing_time: float (секунды)
    Реализация:
      - парсит заголовки DICOM (если pydicom доступен) и извлекает первые ненулевые Study/Series UID
      - path_to_study: "uploaded_zip" если >1 файл, "single_dicom" если ровно 1, "unknown" иначе
      - модель: простая детерминированная функция от числа файлов (можно заменить на реальную интеграцию)
      - симулирует небольшую задержку
    """
    t0 = time.time()

    study_uid = ""
    series_uid = ""

    n = len(dicom_files)

    # determine path_to_study
    if n == 0:
        path_to_study = "unknown"
    elif n == 1:
        path_to_study = "single_dicom"
    else:
        # if all files came from a zip, keep generic uploaded_zip
        path_to_study = "uploaded_zip"

    # Try to parse headers (first non-empty Study/Series UID)
    if pydicom is not None and n > 0:
        for filename, raw in dicom_files:
            try:
                ds = pydicom.dcmread(BytesIO(raw), stop_before_pixels=True, force=True)
                if not study_uid and hasattr(ds, "StudyInstanceUID"):
                    try:
                        study_uid = str(ds.StudyInstanceUID)
                    except Exception:
                        study_uid = ""
                if not series_uid and hasattr(ds, "SeriesInstanceUID"):
                    try:
                        series_uid = str(ds.SeriesInstanceUID)
                    except Exception:
                        series_uid = ""
                # break early if both found
                if study_uid and series_uid:
                    break
            except Exception:
                # skip unreadable files
                continue

    # Simple deterministic "model" inference based on n
    if n == 0:
        probability = 0.0
        pathology = 0
    else:
        probability = min(0.05 + 0.02 * n, 0.95)
        pathology = 1 if probability > 0.5 else 0

    # simulate model processing time (bounded)
    simulated_sleep = 0.12 * min(n, 12)
    time.sleep(simulated_sleep)

    model_processing_time = round(time.time() - t0, 3)

    return {
        "path_to_study": path_to_study,
        "study_uid": study_uid or "unknown",
        "series_uid": series_uid or "unknown",
        "probability_of_pathology": float(probability),
        "pathology": int(pathology),
        "model_processing_time": float(model_processing_time),
    }

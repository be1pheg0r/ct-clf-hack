"""
Основное FastAPI приложение для обработки медицинских изображений CT.
Предоставляет API endpoints для анализа DICOM файлов и просмотра изображений.
"""

from typing import List, Tuple, Dict, Any, Union
from io import BytesIO
import base64
import time
import traceback
import zipfile

import numpy as np
import cv2
import pandas as pd
import pydicom

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel

from ct_clf_backend.app.model_connector import ConnectingLinkWithModel, ConnectingLinkWithModelFromZip
from ct_clf_backend.app.cache_utils import process_uploaded_file, extract_dicom_metadata, cleanup_cache


class OutputData(BaseModel):
    """Модель выходных данных для API ответа."""
    path_to_study: str
    study_uid: str
    series_uid: str
    probability_of_pathology: float
    pathology: int
    processing_status: str
    time_of_processing: float


app = FastAPI(
    title="CT Classification API",
    description="API for CT scan classification and analysis",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_report_xlsx_bytes(output_dict: Dict[str, Any]) -> bytes:
    """
    Создает Excel отчет из результатов анализа.

    Args:
        output_dict: Словарь с результатами анализа

    Returns:
        bytes: Байты Excel файла
    """
    cols = [
        "path_to_study",
        "study_uid",
        "series_uid",
        "probability_of_pathology",
        "pathology",
        "processing_status",
        "time_of_processing",
    ]
    row = {c: output_dict.get(c, "") for c in cols}
    df = pd.DataFrame([row], columns=cols)
    bio = BytesIO()
    df.to_excel(bio, index=False, engine="openpyxl")
    bio.seek(0)
    return bio.read()


def _is_probable_dicom(raw_bytes: bytes) -> bool:
    """
    Проверяет, являются ли байты DICOM файлом.

    Args:
        raw_bytes: Байты файла для проверки

    Returns:
        bool: True если файл вероятно является DICOM
    """
    try:
        if len(raw_bytes) > 132 and raw_bytes[128:132] == b"DICM":
            return True
        try:
            ds = pydicom.dcmread(BytesIO(raw_bytes), stop_before_pixels=True, force=True)
            if hasattr(ds, "SOPClassUID") or hasattr(ds, "StudyInstanceUID"):
                return True
        except Exception:
            return False
    except Exception:
        return False
    return False


def _normalize_to_uint8(arr: np.ndarray, target_max_width: int = 1024) -> np.ndarray:
    """
    Нормализует массив в uint8 с изменением размера.

    Args:
        arr: Входной массив
        target_max_width: Максимальная ширина изображения

    Returns:
        np.ndarray: Нормализованный массив uint8
    """
    a = arr.astype(np.float32)
    mn = np.nanmin(a)
    mx = np.nanmax(a)
    if mx - mn > 0:
        a = (a - mn) / (mx - mn) * 255.0
    else:
        a = np.clip(a, 0, 255)
    a = np.nan_to_num(a).astype(np.uint8)

    if a.ndim == 3 and a.shape[2] > 1:
        a = a[:, :, 0]

    h, w = a.shape[:2]
    if w > target_max_width:
        scale = target_max_width / float(w)
        new_w = int(w * scale)
        new_h = max(1, int(h * scale))
        a = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return a


@app.post("/process-image")
async def process_image(file: UploadFile = File(...), dev: bool = False) -> JSONResponse:
    """
    Основной endpoint для обработки медицинских изображений.

    Args:
        file: Загруженный файл (DICOM или ZIP архив)
        dev: Режим разработки (возвращает тестовые данные)

    Returns:
        JSONResponse: Результаты анализа изображения
    """
    cache_dir = None
    try:
        orig_filename = file.filename or "uploaded_file"
        raw = await file.read()

        if dev:
            response_obj = OutputData(
                path_to_study="Путь к исследованию",
                study_uid="study_uid_example",
                series_uid="series_uid_example",
                probability_of_pathology=0.85,
                pathology=1,
                processing_status="Success",
                time_of_processing=12.34,
            )
            resp = response_obj.model_dump(mode="json")
            xlsx_bytes = _build_report_xlsx_bytes(resp)
            b64 = base64.b64encode(xlsx_bytes).decode("ascii")
            data_uri = "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64," + b64
            resp["report_xlsx"] = data_uri
            return JSONResponse(content=resp)

        t0 = time.time()

        study_uid = "unknown"
        series_uid = "unknown"

        if zipfile.is_zipfile(BytesIO(raw)):
            try:
                selected_dicom_files, temp_cache_dir = process_uploaded_file(raw, max_files=10)
                if selected_dicom_files:
                    study_uid_temp, series_uid_temp = extract_dicom_metadata(selected_dicom_files)
                    if study_uid_temp:
                        study_uid = study_uid_temp
                    if series_uid_temp:
                        series_uid = series_uid_temp
                cleanup_cache(temp_cache_dir)
            except Exception:
                pass
            try:
                model_res: Dict[str, Any] = ConnectingLinkWithModelFromZip(raw, max_slices=10)
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(
                    content={"processing_status": "Failure", "error": f"Model error: {str(e)}"},
                    status_code=500
                )
        else:
            try:
                selected_dicom_files, cache_dir = process_uploaded_file(raw, max_files=10)
                if selected_dicom_files:
                    study_uid_temp, series_uid_temp = extract_dicom_metadata(selected_dicom_files)
                    if study_uid_temp:
                        study_uid = study_uid_temp
                    if series_uid_temp:
                        series_uid = series_uid_temp
                model_res: Dict[str, Any] = ConnectingLinkWithModel(selected_dicom_files)
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(
                    content={"processing_status": "Failure", "error": f"Processing error: {str(e)}"},
                    status_code=500
                )

        path_to_study = str(model_res.get("path_to_study", "unknown"))
        final_study_uid = str(model_res.get("study_uid", study_uid))
        final_series_uid = str(model_res.get("series_uid", series_uid))
        probability = float(model_res.get("probability_of_pathology", 0.0))
        pathology_flag = int(model_res.get("pathology", 0))
        model_time = float(model_res.get("model_processing_time", 0.0))

        time_of_processing = round(time.time() - t0 + model_time, 3)
        processing_status = "Success"

        response_obj = OutputData(
            path_to_study=orig_filename or path_to_study,
            study_uid=final_study_uid,
            series_uid=final_series_uid,
            probability_of_pathology=probability,
            pathology=pathology_flag,
            processing_status=processing_status,
            time_of_processing=time_of_processing,
        )
        resp = response_obj.model_dump(mode="json")

        xlsx_bytes = _build_report_xlsx_bytes(resp)
        b64 = base64.b64encode(xlsx_bytes).decode("ascii")
        data_uri = "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64," + b64
        resp["report_xlsx"] = data_uri

        return JSONResponse(content=resp)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"processing_status": "Failure", "error": str(e)}, status_code=500)

    finally:
        if cache_dir:
            cleanup_cache(cache_dir)


@app.post("/viewer")
async def viewer(file: UploadFile = File(...)) -> JSONResponse:
    """
    Endpoint для просмотра DICOM изображений.

    Args:
        file: Загруженный файл (DICOM или ZIP архив)

    Returns:
        JSONResponse: Список кадров изображений в base64 формате
    """
    try:
        contents = await file.read()
        frames: List[str] = []

        def encode_png_b64(img_arr: np.ndarray) -> str:
            """
            Кодирует изображение в base64 PNG формат.

            Args:
                img_arr: Массив изображения

            Returns:
                str: Base64 строка изображения

            Raises:
                RuntimeError: При ошибке кодирования
            """
            ok, buf = cv2.imencode(".png", img_arr)
            if not ok:
                raise RuntimeError("CV2 encode failed")
            b = buf.tobytes()
            return "data:image/png;base64," + base64.b64encode(b).decode("ascii")

        candidates: List[Tuple[str, bytes]] = []
        try:
            if zipfile.is_zipfile(BytesIO(contents)):
                with zipfile.ZipFile(BytesIO(contents)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        try:
                            data = zf.read(info.filename)
                            if _is_probable_dicom(data):
                                candidates.append((info.filename, data))
                        except Exception:
                            continue
            else:
                if _is_probable_dicom(contents):
                    candidates.append(("uploaded_file", contents))
        except Exception:
            pass

        if not candidates:
            return JSONResponse(content={"frames": []})

        def _candidate_sort_key(item: Tuple[str, bytes]) -> Tuple[int, Union[int, float, str]]:
            """
            Функция для сортировки DICOM файлов по порядку срезов.

            Args:
                item: Кортеж (имя файла, байты)

            Returns:
                Tuple[int, Union[int, float, str]]: Ключ для сортировки
            """
            filename, raw = item
            try:
                ds_head = pydicom.dcmread(BytesIO(raw), stop_before_pixels=True, force=True)
                if hasattr(ds_head, "InstanceNumber"):
                    try:
                        return (0, int(ds_head.InstanceNumber))
                    except Exception:
                        pass
                if hasattr(ds_head, "SliceLocation"):
                    try:
                        return (0, float(ds_head.SliceLocation))
                    except Exception:
                        pass
                if hasattr(ds_head, "ImagePositionPatient"):
                    ip = ds_head.ImagePositionPatient
                    try:
                        return (0, float(ip[2]))
                    except Exception:
                        pass
            except Exception:
                pass
            return (1, filename)

        candidates_sorted = sorted(candidates, key=_candidate_sort_key)

        max_frames = 1000
        for filename, raw in candidates_sorted:
            if len(frames) >= max_frames:
                break
            try:
                ds = pydicom.dcmread(BytesIO(raw), force=True)
                if not hasattr(ds, "PixelData"):
                    continue
                arr = ds.pixel_array
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 3:
                        if arr.shape[0] > 1 and arr.shape[1] > 10:
                            for i in range(arr.shape[0]):
                                if len(frames) >= max_frames:
                                    break
                                frame = arr[i]
                                if frame.ndim == 3 and frame.shape[2] > 1:
                                    frame = frame[:, :, 0]
                                img_u8 = _normalize_to_uint8(frame)
                                frames.append(encode_png_b64(img_u8))
                        else:
                            frame = arr[:, :, 0] if arr.shape[2] > 1 else arr
                            img_u8 = _normalize_to_uint8(frame)
                            frames.append(encode_png_b64(img_u8))
                    elif arr.ndim == 2:
                        img_u8 = _normalize_to_uint8(arr)
                        frames.append(encode_png_b64(img_u8))
                    else:
                        try:
                            frame = np.squeeze(arr)
                            if frame.ndim == 2:
                                img_u8 = _normalize_to_uint8(frame)
                                frames.append(encode_png_b64(img_u8))
                        except Exception:
                            continue
                else:
                    continue
            except Exception:
                continue

        return JSONResponse(content={"frames": frames})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": f"Viewer processing failed: {str(e)}"}, status_code=500)


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker health checks."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "ct-clf-backend"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

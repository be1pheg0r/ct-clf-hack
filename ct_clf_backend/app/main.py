# backend/app/main.py
from typing import List, Tuple, Dict, Any
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

# импорт функции-заглушки: должна быть в backend/app/model_connector.py
# Интерфейс: ConnectingLinkWithModel(dicom_candidates: List[Tuple[str, bytes]]) -> Dict[str, Any]
# Ожидаемые ключи в результате: path_to_study, study_uid, series_uid,
# probability_of_pathology, pathology, model_processing_time
from model_connector import ConnectingLinkWithModel

# ----------------- Pydantic schema -----------------
class OutputData(BaseModel):
    path_to_study: str
    study_uid: str
    series_uid: str
    probability_of_pathology: float
    pathology: int
    processing_status: str
    time_of_processing: float

# ----------------- App init -----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для разработки. В проде ограничить домены.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Helpers -----------------
def _build_report_xlsx_bytes(output_dict: dict) -> bytes:
    """
    Создаёт XLSX в памяти из словаря (одна строка) с колонками:
    path_to_study, study_uid, series_uid, probability_of_pathology, pathology, processing_status, time_of_processing
    Возвращает bytes XLSX.
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
    Быстрая эвристика для определения, похожи ли байты на DICOM.
    1) Проверяем сигнатуру 'DICM' на смещении 128.
    2) Если нет, пытаемся прочитать заголовок через pydicom (stop_before_pixels=True).
    """
    try:
        if len(raw_bytes) > 132 and raw_bytes[128:132] == b"DICM":
            return True
        # Попытка чтения заголовка
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
    Приводит массив пикселей к uint8 (0..255), нормализует по min/max и
    масштабирует по ширине если очень широкий.
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
        # если RGB — возьмём первый канал (более надёжно было бы конвертировать)
        a = a[:, :, 0]

    h, w = a.shape[:2]
    if w > target_max_width:
        scale = target_max_width / float(w)
        new_w = int(w * scale)
        new_h = max(1, int(h * scale))
        a = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return a

# ----------------- Endpoints -----------------
@app.post("/process-image")
async def process_image(file: UploadFile = File(...), dev: bool = False):
    """
    Принимает ZIP (или одиночный DICOM), собирает кандидатов DICOM
    в виде списка (filename, bytes) и передаёт в ConnectingLinkWithModel.
    Возвращает JSON со стандартной метадатой + report_xlsx (data-uri base64).
    """
    try:
        raw = await file.read()

        # ---------- dev stub ----------
        if dev:
            response_obj = OutputData(
                path_to_study="Путь к исследованию",
                study_uid="Идентификатор исследования",
                series_uid="Идентификатор серии",
                probability_of_pathology=0.85,
                pathology=1,
                processing_status="Success",
                time_of_processing=12.34,
            )
            resp = response_obj.model_dump(mode="json")
            # xlsx
            xlsx_bytes = _build_report_xlsx_bytes(resp)
            b64 = base64.b64encode(xlsx_bytes).decode("ascii")
            data_uri = "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64," + b64
            resp["report_xlsx"] = data_uri
            return JSONResponse(content=resp)

        # ---------- production flow ----------
        t0 = time.time()
        dicom_candidates: List[Tuple[str, bytes]] = []

        # collect candidates from ZIP or single file
        try:
            if zipfile.is_zipfile(BytesIO(raw)):
                with zipfile.ZipFile(BytesIO(raw)) as zf:
                    infos = [info for info in zf.infolist() if not info.is_dir()]
                    for info in infos:
                        try:
                            data = zf.read(info.filename)
                            if not _is_probable_dicom(data):
                                continue
                            dicom_candidates.append((info.filename, data))
                        except Exception:
                            continue
            else:
                if _is_probable_dicom(raw):
                    dicom_candidates.append(("uploaded_file", raw))
        except Exception:
            # ignore extraction failure
            pass

        # If no dicom found, still call model connector with empty list (it can decide)
        try:
            model_res: Dict[str, Any] = ConnectingLinkWithModel(dicom_candidates)
        except Exception as e:
            # model failed
            traceback.print_exc()
            return JSONResponse(content={"processing_status": "Failure", "error": f"Model error: {str(e)}"}, status_code=500)

        # read expected fields from model_res (provide defaults)
        path_to_study = str(model_res.get("path_to_study", "unknown"))
        study_uid = str(model_res.get("study_uid", "unknown"))
        series_uid = str(model_res.get("series_uid", "unknown"))
        probability = float(model_res.get("probability_of_pathology", 0.0))
        pathology_flag = int(model_res.get("pathology", 0))
        model_time = float(model_res.get("model_processing_time", 0.0))

        time_of_processing = round(time.time() - t0 + model_time, 3)
        processing_status = "Success"

        response_obj = OutputData(
            path_to_study=path_to_study,
            study_uid=study_uid,
            series_uid=series_uid,
            probability_of_pathology=probability,
            pathology=pathology_flag,
            processing_status=processing_status,
            time_of_processing=time_of_processing,
        )
        resp = response_obj.model_dump(mode="json")

        # generate xlsx bytes and attach as data-uri (do NOT add to Pydantic model)
        xlsx_bytes = _build_report_xlsx_bytes(resp)
        b64 = base64.b64encode(xlsx_bytes).decode("ascii")
        data_uri = "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64," + b64
        resp["report_xlsx"] = data_uri

        return JSONResponse(content=resp)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"processing_status": "Failure", "error": str(e)}, status_code=500)


@app.post("/viewer")
async def viewer(file: UploadFile = File(...)):
    """
    Возвращает JSON {"frames": ["data:image/png;base64,...", ...]}
    Обрабатывает ZIP с DICOM или одиночный DICOM. Ограничивает число кадров (max_frames).
    """
    try:
        contents = await file.read()
        frames: List[str] = []

        # helper to encode numpy -> png base64
        def encode_png_b64(img_arr: np.ndarray) -> str:
            ok, buf = cv2.imencode(".png", img_arr)
            if not ok:
                raise RuntimeError("CV2 encode failed")
            b = buf.tobytes()
            return "data:image/png;base64," + base64.b64encode(b).decode("ascii")

        # extract files
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

        # sort candidates by InstanceNumber / SliceLocation / filename
        def _candidate_sort_key(item: Tuple[str, bytes]):
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
            # fallback: sort by filename
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
                arr = ds.pixel_array  # may be numpy array
                # handle multi-frame DICOMs
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 3:
                        # could be (frames, rows, cols) or (rows, cols, channels)
                        if arr.shape[0] > 1 and arr.shape[1] > 10:
                            # treat as multiple frames
                            for i in range(arr.shape[0]):
                                if len(frames) >= max_frames:
                                    break
                                frame = arr[i]
                                if frame.ndim == 3 and frame.shape[2] > 1:
                                    frame = frame[:, :, 0]
                                img_u8 = _normalize_to_uint8(frame)
                                frames.append(encode_png_b64(img_u8))
                        else:
                            # single image with channels
                            frame = arr[:, :, 0] if arr.shape[2] > 1 else arr
                            img_u8 = _normalize_to_uint8(frame)
                            frames.append(encode_png_b64(img_u8))
                    elif arr.ndim == 2:
                        img_u8 = _normalize_to_uint8(arr)
                        frames.append(encode_png_b64(img_u8))
                    else:
                        # try squeeze
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
                # on any read/encode error skip file
                continue

        return JSONResponse(content={"frames": frames})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": f"Viewer processing failed: {str(e)}"}, status_code=500)


# ----------------- run -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

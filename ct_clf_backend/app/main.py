# backend/app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
import zipfile
import base64
import time
import traceback
from typing import List, Tuple, Any

import numpy as np
import cv2
import pydicom
import pandas as pd
from pydantic import BaseModel

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
    allow_origins=["*"],  # dev; ограничь в проде
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Helpers -----------------
def _normalize_to_uint8(arr: np.ndarray, target_max_width: int = 1024) -> np.ndarray:
    a = arr.astype(np.float32)
    mn = np.nanmin(a)
    mx = np.nanmax(a)
    if mx - mn > 0:
        a = (a - mn) / (mx - mn) * 255.0
    else:
        a = np.clip(a, 0, 255)
    a = np.nan_to_num(a).astype(np.uint8)

    h, w = a.shape[:2]
    if w > target_max_width:
        scale = target_max_width / float(w)
        new_w = int(w * scale)
        new_h = max(1, int(h * scale))
        a = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return a

def _build_report_xlsx_bytes(output_dict: dict) -> bytes:
    """
    output_dict should contain keys:
    path_to_study, study_uid, series_uid, probability_of_pathology,
    pathology, processing_status, time_of_processing
    Returns bytes of xlsx file.
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
    output = BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)
    return output.read()

# ----------------- Endpoints -----------------
@app.post("/process-image")
async def process_image(file: UploadFile = File(...), dev: bool = True):
    """
    Возвращает JSON с полями:
    - path_to_study, study_uid, series_uid, probability_of_pathology, pathology, processing_status, time_of_processing
    - report_xlsx: data:...;base64,... (XLSX generated from those fields)
    (frames не возвращаем здесь — для них отдельный endpoint /viewer)
    """
    try:
        raw = await file.read()

        # dev stub
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
            resp = response_obj.model_dump(mode="json")  # dict

            # generate xlsx bytes via pandas + BytesIO
            xlsx_bytes = _build_report_xlsx_bytes(resp)
            b64 = base64.b64encode(xlsx_bytes).decode("ascii")
            data_uri = "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64," + b64
            resp["report_xlsx"] = data_uri
            # optionally: resp["frames"] = []
            return JSONResponse(content=resp)

        # ---------- production flow (example) ----------
        # try to extract some DICOM tags from uploaded zip
        t0 = time.time()
        study_uid = ""
        series_uid = ""
        path_to_study = ""
        probability = 0.0
        pathology_flag = 0
        processing_status = "Failure"

        # if uploaded file is zip, try to read first dcm and extract study/series UIDs
        try:
            if zipfile.is_zipfile(BytesIO(raw)):
                with zipfile.ZipFile(BytesIO(raw)) as zf:
                    infos = [info for info in zf.infolist() if not info.is_dir()]
                    for info in infos:
                        try:
                            data = zf.read(info.filename)
                            ds = pydicom.dcmread(BytesIO(data), stop_before_pixels=True, force=True)
                            # try to extract StudyInstanceUID / SeriesInstanceUID if present
                            if not study_uid and hasattr(ds, "StudyInstanceUID"):
                                study_uid = str(ds.StudyInstanceUID)
                            if not series_uid and hasattr(ds, "SeriesInstanceUID"):
                                series_uid = str(ds.SeriesInstanceUID)
                            # break when at least study found
                            if study_uid and series_uid:
                                break
                        except Exception:
                            continue
                    path_to_study = "uploaded_zip"
            else:
                # not a zip — try read as single DICOM
                try:
                    ds = pydicom.dcmread(BytesIO(raw), stop_before_pixels=True, force=True)
                    if hasattr(ds, "StudyInstanceUID"):
                        study_uid = str(ds.StudyInstanceUID)
                    if hasattr(ds, "SeriesInstanceUID"):
                        series_uid = str(ds.SeriesInstanceUID)
                    path_to_study = "single_dicom"
                except Exception:
                    path_to_study = "unknown"
        except Exception:
            # ignore extraction failure, keep defaults
            pass

        # here put your real model inference logic to compute probability/pathology
        # for example we just fill with demo logic:
        probability = 0.42
        pathology_flag = 0
        processing_status = "Success"
        time_of_processing = round(time.time() - t0, 2)

        response_obj = OutputData(
            path_to_study=path_to_study,
            study_uid=study_uid or "unknown",
            series_uid=series_uid or "unknown",
            probability_of_pathology=probability,
            pathology=pathology_flag,
            processing_status=processing_status,
            time_of_processing=time_of_processing,
        )
        resp = response_obj.model_dump(mode="json")

        # generate xlsx bytes
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
    Обрабатывает DICOM-файлы внутри ZIP, поддерживает одиночные и многокадровые DICOM.
    """
    try:
        contents = await file.read()
        with zipfile.ZipFile(BytesIO(contents)) as zf:
            infolist = [info for info in zf.infolist() if not info.is_dir()]

            if not infolist:
                return JSONResponse(content={"frames": []})

            candidates: List[Tuple[Any, bytes, str]] = []  # (sort_key, raw_bytes, name)

            for info in infolist:
                try:
                    data = zf.read(info.filename)
                    ds_head = pydicom.dcmread(BytesIO(data), stop_before_pixels=True, force=True)
                    key = None
                    if hasattr(ds_head, "InstanceNumber"):
                        try:
                            key = int(ds_head.InstanceNumber)
                        except Exception:
                            key = None
                    if key is None and hasattr(ds_head, "SliceLocation"):
                        try:
                            key = float(ds_head.SliceLocation)
                        except Exception:
                            key = None
                    if key is None and hasattr(ds_head, "ImagePositionPatient"):
                        ip = ds_head.ImagePositionPatient
                        try:
                            key = float(ip[2])
                        except Exception:
                            key = None
                    candidates.append((key, data, info.filename))
                except Exception:
                    continue

            if not candidates:
                return JSONResponse(content={"frames": []})

            def _sort_key(item):
                k, _, name = item
                return (0, 0) if k is None else (1, k)

            candidates_sorted = sorted(candidates, key=_sort_key)

            frames: List[str] = []
            max_frames = 300
            for key, raw_bytes, name in candidates_sorted:
                if len(frames) >= max_frames:
                    break
                try:
                    ds = pydicom.dcmread(BytesIO(raw_bytes), force=True)
                    if not hasattr(ds, "PixelData"):
                        continue
                    arr = ds.pixel_array
                    if arr is None:
                        continue

                    if arr.ndim == 3:
                        # (frames, rows, cols) or (rows, cols, channels)
                        if arr.shape[0] > 1 and arr.shape[1] > 10:
                            # multiline frames
                            for i in range(arr.shape[0]):
                                if len(frames) >= max_frames:
                                    break
                                img = arr[i]
                                if img.ndim == 3 and img.shape[2] > 1:
                                    img2 = img[:, :, 0]
                                else:
                                    img2 = img
                                img_u8 = _normalize_to_uint8(img2)
                                ok, buf = cv2.imencode(".png", img_u8)
                                if ok:
                                    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                                    frames.append("data:image/png;base64," + b64)
                        else:
                            img = arr[:, :, 0] if arr.shape[2] > 1 else arr
                            img_u8 = _normalize_to_uint8(img)
                            ok, buf = cv2.imencode(".png", img_u8)
                            if ok:
                                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                                frames.append("data:image/png;base64," + b64)
                    elif arr.ndim == 2:
                        img_u8 = _normalize_to_uint8(arr)
                        ok, buf = cv2.imencode(".png", img_u8)
                        if ok:
                            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                            frames.append("data:image/png;base64," + b64)
                    else:
                        try:
                            img = np.squeeze(arr)
                            if img.ndim == 2:
                                img_u8 = _normalize_to_uint8(img)
                                ok, buf = cv2.imencode(".png", img_u8)
                                if ok:
                                    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                                    frames.append("data:image/png;base64," + b64)
                        except Exception:
                            continue
                except Exception:
                    continue

            return JSONResponse(content={"frames": frames})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": f"Viewer processing failed: {str(e)}"}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

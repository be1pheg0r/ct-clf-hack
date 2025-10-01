# backend/app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
import zipfile
import base64
import time
import traceback
from typing import List, Dict, Any, Tuple

import numpy as np
import cv2
import pandas as pd
from pydantic import BaseModel

# импорт заглушки модели (файл model_connector.py в той же папке)
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
    xlsx_bytes: str

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

def _is_probable_dicom(raw_bytes: bytes) -> bool:
    """
    Быстрая эвристика: проверяем наличие DICM signature на позиции 128
    или попытаемся dcmread с stop_before_pixels (если pydicom доступен).
    Возвращает True если похоже на DICOM.
    """
    try:
        if len(raw_bytes) > 132 and raw_bytes[128:132] == b"DICM":
            return True
        # quick header attempt via pydicom if available
        try:
            import pydicom
            from io import BytesIO as _B
            ds = pydicom.dcmread(_B(raw_bytes), stop_before_pixels=True, force=True)
            # if read succeeded and has any common attributes, consider it DICOM
            if hasattr(ds, "SOPClassUID") or hasattr(ds, "StudyInstanceUID"):
                return True
        except Exception:
            return False
    except Exception:
        return False
    return False

# ----------------- Endpoint -----------------
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

        # ---------- production flow ----------
        t0 = time.time()
        processing_status = "Failure"

        # collect candidate dicom files as list of (filename, bytes)
        dicom_candidates: List[Tuple[str, bytes]] = []

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
                # single file — treat as a candidate if it looks like DICOM
                if _is_probable_dicom(raw):
                    dicom_candidates.append(("uploaded_file", raw))
        except Exception:
            # ignore extraction failures
            pass

        # Call the model connector with the list of (filename, bytes)
        try:
            model_result: Dict[str, Any] = ConnectingLinkWithModel(dicom_candidates)
            # expected keys: path_to_study, study_uid, series_uid, probability_of_pathology, pathology, model_processing_time
            probability = float(model_result.get("probability_of_pathology", 0.0))
            pathology_flag = int(model_result.get("pathology", 0))
            path_to_study = str(model_result.get("path_to_study", "unknown"))
            study_uid = str(model_result.get("study_uid", "unknown"))
            series_uid = str(model_result.get("series_uid", "unknown"))
            model_time = float(model_result.get("model_processing_time", 0.0))
            processing_status = "Success"
        except Exception as ex:
            # model failed — fill defaults
            probability = 0.0
            pathology_flag = 0
            path_to_study = "unknown"
            study_uid = "unknown"
            series_uid = "unknown"
            model_time = 0.0
            processing_status = f"ModelFailure: {str(ex)}"

        time_of_processing = round(time.time() - t0 + float(model_time), 3)

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

        # generate xlsx bytes
        xlsx_bytes = _build_report_xlsx_bytes(resp)
        b64 = base64.b64encode(xlsx_bytes).decode("ascii")
        data_uri = "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64," + b64
        resp["report_xlsx"] = data_uri

        return JSONResponse(content=resp)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"processing_status": "Failure", "error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

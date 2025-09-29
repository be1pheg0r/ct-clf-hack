from io import BytesIO
import zipfile
import base64
import traceback

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import pydicom

# Если у тебя есть собственные schemas/imports — подключи их или убери ниже строки
# from ct_clf_backend.db.init import engine
# from ct_clf_backend.schemas import InputData, OutputData

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _normalize_to_uint8(arr: np.ndarray, target_max_width: int = 512) -> np.ndarray:
    """Нормализуем массив пикселей в uint8 и при необходимости ресайзим по ширине."""
    arr = arr.astype(np.float32)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx - mn > 0:
        arr = (arr - mn) / (mx - mn) * 255.0
    else:
        arr = np.clip(arr, 0, 255)
    arr = np.nan_to_num(arr).astype(np.uint8)

    h, w = arr.shape[:2]
    if w > target_max_width:
        scale = target_max_width / float(w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return arr

@app.post("/process-image")
async def process_image(file: UploadFile = File(...), dev: bool = True):
    """
    Ожидается: ZIP-архив, внутри — папка с DICOM-файлами (возможно без расширений).
    В dev режиме возвращаем demo Output + поле 'frames' — массив data:image/png;base64,...
    """
    response = {}
    try:
        raw = await file.read()
        if zipfile.is_zipfile(BytesIO(raw)):
            frames_b64 = []
            with zipfile.ZipFile(BytesIO(raw)) as zf:
                candidates = []  # (sort_key, ds, arr)
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    try:
                        data = zf.read(info.filename)
                        ds = pydicom.dcmread(BytesIO(data), force=True, stop_before_pixels=False)
                        if hasattr(ds, 'PixelData'):
                            arr = ds.pixel_array
                            key = None
                            if hasattr(ds, 'InstanceNumber'):
                                try:
                                    key = int(ds.InstanceNumber)
                                except Exception:
                                    key = None
                            elif hasattr(ds, 'SliceLocation'):
                                try:
                                    key = float(ds.SliceLocation)
                                except Exception:
                                    key = None
                            candidates.append((key, ds, arr))
                    except Exception:
                        continue

                def _sort_key(x):
                    k = x[0]
                    return (0, 0) if k is None else (1, k)

                candidates = sorted(candidates, key=_sort_key)

                for key, ds, arr in candidates:
                    if arr.ndim == 3 and arr.shape[2] > 1:
                        arr = arr[:, :, 0]
                    img_u8 = _normalize_to_uint8(arr, target_max_width=512)
                    success, encoded = cv2.imencode('.png', img_u8)
                    if success:
                        b64 = base64.b64encode(encoded.tobytes()).decode('ascii')
                        frames_b64.append('data:image/png;base64,' + b64)

            response = {
                'path_to_study': 'Путь к исследованию',
                'study_uid': 'Идентификатор исследования',
                'series_uid': 'Идентификатор серии',
                'probability_of_pathology': 0.85,
                'pathology': 1,
                'processing_status': 'Success',
                'time_of_processing': 12.34,
                'frames': frames_b64,
            }
        else:
            response = {
                'path_to_study': 'Путь к исследованию',
                'study_uid': 'Идентификатор исследования',
                'series_uid': 'Идентификатор серии',
                'probability_of_pathology': 0.85,
                'pathology': 1,
                'processing_status': 'Success',
                'time_of_processing': 12.34,
                'frames': [],
            }

        return JSONResponse(content=response)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={'processing_status': 'Failure', 'error': str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

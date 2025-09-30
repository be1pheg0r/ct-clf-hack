# backend/app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
import zipfile
import cv2
import numpy as np
import pydicom
import base64
import time
from pydantic import BaseModel
from typing import List, Tuple, Any

# ----- Pydantic schema -----
class OutputData(BaseModel):
    path_to_study: str
    study_uid: str
    series_uid: str
    probability_of_pathology: float
    pathology: int
    processing_status: str
    time_of_processing: float

# ----- App -----
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для dev; в проде сузить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _normalize_to_uint8(arr: np.ndarray, target_max_width: int = 1024) -> np.ndarray:
    """Нормализует массив в uint8 и ресайзит по ширине если нужно."""
    # Приводим к float
    a = arr.astype(np.float32)
    # Если есть оконные уровни в DICOM — можно их тут применить (упрощенно не делаем)
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

@app.post("/process-image")
async def process_image(file: UploadFile = File(...), dev: bool = True):
    if dev:
        response = OutputData(
            path_to_study="Путь к исследованию",
            study_uid="Идентификатор исследования",
            series_uid="Идентификатор серии",
            probability_of_pathology=0.85,
            pathology=1,
            processing_status="Success",
            time_of_processing=12.34,
        )
        return JSONResponse(content=response.model_dump(mode="json"))

    # Здесь должна быть реальная модельная логика
    t0 = time.time()
    response = OutputData(
        path_to_study="/real/path",
        study_uid="study123",
        series_uid="series456",
        probability_of_pathology=0.42,
        pathology=0,
        processing_status="Success",
        time_of_processing=round(time.time() - t0, 2),
    )
    return JSONResponse(content=response.model_dump(mode="json"))


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

            # Пробуем открыть каждый файл как DICOM
            for info in infolist:
                try:
                    data = zf.read(info.filename)
                    # Попробуем прочитать заголовок (force=True чтобы читать без расширения)
                    ds = pydicom.dcmread(BytesIO(data), stop_before_pixels=True, force=True)
                    # Определяем сортировочный ключ (InstanceNumber, SliceLocation, ImagePositionPatient z, fallback)
                    key = None
                    if hasattr(ds, "InstanceNumber"):
                        try:
                            key = int(ds.InstanceNumber)
                        except Exception:
                            key = None
                    if key is None and hasattr(ds, "SliceLocation"):
                        try:
                            key = float(ds.SliceLocation)
                        except Exception:
                            key = None
                    if key is None and hasattr(ds, "ImagePositionPatient"):
                        ip = ds.ImagePositionPatient
                        try:
                            # ImagePositionPatient = [x, y, z]
                            key = float(ip[2])
                        except Exception:
                            key = None
                    candidates.append((key, data, info.filename))
                except Exception:
                    # если не DICOM — просто пропускаем
                    continue

            if not candidates:
                # На всякий случай попробуем прочитать все как DICOM напрямую (force)
                # но если и так ничего — возвращаем пустой список
                return JSONResponse(content={"frames": []})

            # Сортируем: те, у кого есть ключ идут первыми по ключу, потом по имени файла
            def _sort_key(item):
                k, _, name = item
                return (0, 0) if k is None else (1, k)

            candidates_sorted = sorted(candidates, key=_sort_key)

            frames: List[str] = []
            max_frames = 300  # ограничение общего числа возвращаемых срезов
            for key, raw_bytes, name in candidates_sorted:
                if len(frames) >= max_frames:
                    break
                try:
                    ds = pydicom.dcmread(BytesIO(raw_bytes), force=True)
                    if not hasattr(ds, "PixelData"):
                        continue
                    arr = ds.pixel_array  # numpy array

                    # Обработка возможных форматов pixel_array:
                    # - многокадровый: shape (frames, rows, cols)
                    # - мультиканальный: shape (rows, cols, samples)
                    # - монохромный: shape (rows, cols)
                    if arr is None:
                        continue

                    if arr.ndim == 3:
                        # возможно (frames, rows, cols) либо (rows, cols, samples)
                        # если первый размер >1 и меньше, чем небольшое число — считаем стеком
                        if arr.shape[0] > 1 and arr.shape[0] <= 1024 and arr.shape[1] > 16:
                            # многокадровый DICOM — добавляем кадры по одному
                            for i in range(arr.shape[0]):
                                if len(frames) >= max_frames:
                                    break
                                img = arr[i]
                                if img.ndim == 3 and img.shape[2] > 1:
                                    # цвет / multi-sample: берем первый канал или делаем BGR
                                    img2 = img[:, :, 0]
                                else:
                                    img2 = img
                                img_u8 = _normalize_to_uint8(img2)
                                ok, buf = cv2.imencode(".png", img_u8)
                                if ok:
                                    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                                    frames.append("data:image/png;base64," + b64)
                        else:
                            # скорее всего (rows, cols, channels) — возьмём первый канал
                            img = arr[:, :, 0]
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
                        # нестандартный формат — пытаемся свести к 2D
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
                    # игнорируем проблемный файл, продолжаем
                    continue

            return JSONResponse(content={"frames": frames})

    except Exception as e:
        return JSONResponse(content={"error": f"Viewer processing failed: {str(e)}"}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from io import BytesIO

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from ct_clf_backend.db.init import engine
from ct_clf_backend.schemas import InputData, OutputData

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-image")
async def process_image(file: UploadFile = File(...), dev: bool = True):
    if dev:
        """
        result = {
            "path_to_study": "Путь к исследованию",
            "study_uid": "Идентификатор исследования",
            "series_uid": "Идентификатор серии",
            "probability_of_pathology": "Вероятность патологии (от 0.0 к 1.0)",
            "pathology": "Норма ли патология (0 или 1)",
            "processing_status": "Статус обработки (Success/Failure)",
            "time_of_processing": "Время обработки (секунды)",
        }
        """
        response = OutputData(
            path_to_study="Путь к исследованию",
            study_uid="Идентификатор исследования",
            series_uid="Идентификатор серии",
            probability_of_pathology=0.85,
            pathology=1,
            processing_status="Success",
            time_of_processing=12.34
        )
        return JSONResponse(content=response.model_dump(mode="json"))
    else:
        response = {}
        ...
    return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
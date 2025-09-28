from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import cv2
import numpy as np
from io import BytesIO
import uvicorn
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)) -> JSONResponse:
    #print(file)

    result = {
        "path_to_study": "Путь к исследованию",
        "study_uid": "Идентификатор исследования",
        "series_uid": "Идентификатор серии",
        "probability_of_pathology": "Вероятность патологии (от 0.0 к 1.0)",
        "pathology": "Норма ли патология (0 или 1)",
        "processing_status": "Статус обработки (Success/Failure)",
        "time_of_processing": "Время обработки (секунды)",
    }
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
Схемы для валидации входных и выходных данных API.
"""
from pydantic import BaseModel, Field
from typing import List, Optional

class InputData(BaseModel):
    query_id: str = Field(..., description="Уникальный идентификатор запроса.")
    session_id: str = Field(..., description="Идентификатор сессии.")
    zip_file_path: str = Field(..., description="Путь к ZIP файлу с DICOM изображениями.")
    ts: str = Field(..., description="Таймстамп запроса.")

class OutputData(BaseModel):
    path_to_study: str = Field(..., description="Путь к исследованию.")
    study_uid: str = Field(..., description="Айди исследования.")
    series_uid: str = Field(..., description="Серия исследования.")
    probability_of_pathology: float = Field(..., description="Вероятность наличия патологии (от 0 до 1).")
    pathology: int = Field(..., description="Метка патологии (0 - нет, 1 - есть).")
    processing_status: str = Field(..., description="Статус обработки (например, 'success' или 'error').")
    time_of_processing: float = Field(..., description="Время обработки в секундах.")
    slices: Optional[List[str]] = None
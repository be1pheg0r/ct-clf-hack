from pydantic import BaseModel, Field


class InputData(BaseModel):
    query_id: str = Field(..., description="Unique identifier for the query.")
    session_id: str = Field(..., description="Session identifier.")
    zip_file_path: str = Field(..., description="Path to the input zip file containing CT scan images.")
    ts: str = Field(..., description="Timestamp.")

class OutputData(BaseModel):
    path_to_study: str = Field(..., description="Path to the study.")
    study_uid: str = Field(..., description="Study unique identifier.")
    series_uid: str = Field(..., description="Series unique identifier.")
    probability_of_pathology: float = Field(..., description="Probability of pathology (from 0.0 to 1.0).")
    pathology: int = Field(..., description="Indicates if pathology is present (0 for no pathology, 1 for pathology).")
    processing_status: str = Field(..., description="Processing status (Success/Failure).")
    time_of_processing: float = Field(..., description="Time taken for processing (in seconds).")
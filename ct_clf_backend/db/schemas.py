from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone

Base = declarative_base()

class QueriesTable(Base):
    __tablename__ = "queries"

    query_id = Column(String, primary_key=True, index=True)
    session_id = Column(String, index=True)
    zip_file_path = Column(String)
    ts = Column(DateTime, default=datetime.now(tz=timezone.utc))


class ResultsTable(Base):
    __tablename__ = "results"

    query_id = Column(String, primary_key=True, index=True)
    path_to_study = Column(String)
    study_uid = Column(String)
    series_uid = Column(String)
    probability_of_pathology = Column(String)
    pathology = Column(Integer)
    processing_status = Column(String)
    time_of_processing = Column(String)

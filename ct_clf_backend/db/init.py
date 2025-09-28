from ct_clf_backend.db.schemas import QueriesTable, ResultsTable, Base
from ct_clf_backend.db.settings import DB_URL, DB_NAME, DB_HOST, DB_PORT
from sqlalchemy import create_engine

engine = create_engine(DB_URL, echo=False)
Base.metadata.create_all(bind=engine)
print(f"Connected to database '{DB_NAME}' at '{DB_HOST}:{DB_PORT}'")
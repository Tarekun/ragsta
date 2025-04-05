from db.initialization import database_initialization
from doc_ingestion import ingest_directory
from embedding import embedding_pipeline

database_initialization()
ingest_directory("C:\\Users\\danie\\uni\\deep Learning\\papers\\scaling")
embedding_pipeline("nomic-embed-text")

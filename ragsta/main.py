from db.initialization import *
from doc_ingestion import ingest_directory
from embedding import embedding_pipeline

db_init("admin", "password")
pgvector_init()
article_db_init("admin", "password")

ingest_directory("C:\\Users\\danie\\uni\\deep Learning\\papers\\scaling")
embedding_pipeline("nomic-embed-text")

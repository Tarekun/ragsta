from db.initialization import database_initialization
from doc_ingestion import ingest_directory
from embedding import embedding_pipeline
from rag import execute_rag_inference


QUESTION = "explain neural scaling laws with references from the context given"


database_initialization()
# ingest_directory("C:\\Users\\danie\\uni\\deep Learning\\papers\\scaling")
# embedding_pipeline("qwen2.5:14b")
execute_rag_inference(QUESTION, "qwen2.5:14b", "http://192.168.1.102:11434")

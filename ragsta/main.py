from db.initialization import database_initialization
from doc_ingestion import ingest_directory
from embedding import embedding_pipeline
from rag import execute_rag_inference
import hydra
from omegaconf import DictConfig


QUESTION = "explain neural scaling laws with references from the context given"


@hydra.main(version_base=None, config_path="../config", config_name="demo")
def main(cfg: DictConfig):
    database_initialization()
    # ingest_directory("C:\\Users\\danie\\uni\\deep Learning\\papers\\scaling")
    # embedding_pipeline("qwen2.5:14b", cfg)
    execute_rag_inference(QUESTION, cfg)


if __name__ == "__main__":
    main()

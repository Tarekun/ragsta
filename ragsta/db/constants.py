from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

DB_NAME = "ragsta"

BRONZE_SCHEMA = "source"
EMBEDDINGS_SCHEMA = "embeddings"
schemas = [BRONZE_SCHEMA, EMBEDDINGS_SCHEMA]

TABLE_RAW_ARTICLES = "article"


def get_engine(
    username: str = "admin",
    password: str = "password",
    host: str = "localhost",
    port: int = 5432,
):
    return create_engine(
        f"postgresql://{username}:{password}@{host}:{port}/{DB_NAME}",
        connect_args={"options": "-csearch_path=embeddings,source"},
    )


def confirm_model_schema(embedding_model: str, engine):
    with engine.connect() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{embedding_model}"'))
        conn.commit()

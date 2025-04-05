import os
import numpy as np
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from db.models import Article, get_embedding_table_for
from db.constants import get_engine

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.1.102:11434")


def get_unembedded_articles(session, EmbeddingTable, batch_size: int) -> list[Article]:
    return (
        session.query(Article)
        .outerjoin(
            EmbeddingTable,
            Article.title == EmbeddingTable.article_title,
        )
        .filter(EmbeddingTable.article_title.is_(None))
        .limit(batch_size)
        .all()
    )


def embed_article(
    article: Article, model: str, ollama_host: str, session, EmbeddingTable
):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(article.content)
    print(f"Processing article: {article.title} -> {len(chunks)} chunks")

    for idx, chunk in enumerate(chunks):
        response = requests.post(
            f"{ollama_host}/api/embeddings", json={"model": model, "prompt": chunk}
        )
        response.raise_for_status()
        embedding = response.json()["embedding"]

        session.merge(
            EmbeddingTable(
                article_title=article.title,
                chunk_index=idx,
                embedding=np.array(embedding).tolist(),
            )
        )

    return len(chunks)


def embedding_pipeline(
    model: str,
    batch_size: int = 32,
    ollama_host: str = OLLAMA_HOST,
    db_username: str = "admin",
    db_password: str = "password",
    db_host: str = "localhost",
    db_port: int = 5432,
):
    engine = get_engine(db_username, db_password, db_host, db_port)
    EmbeddingTable = get_embedding_table_for(model, engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        articles = get_unembedded_articles(session, EmbeddingTable, batch_size)
        if not articles:
            print("No articles to embed")
            return
        else:
            print(f"Found {len(articles)} unembedded articles")

        total_chunks = 0
        for article in articles:
            total_chunks += embed_article(
                article, model, ollama_host, session, EmbeddingTable
            )

        session.commit()
        print(
            f"Embedded {total_chunks} chunks from {len(articles)} articles with model '{model}'"
        )

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

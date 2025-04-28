import os
import uuid
from sqlalchemy.orm import sessionmaker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain.schema import Document
from db.models import Article, get_embedding_table_for
from db.constants import get_engine, confirm_model_schema
from db.vectorstore import vector_store


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.1.102:11434")
ACADEMIC_PUBLICATIONS_COLLECTION = "academic-articles"


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
    article: Article,
    model: str,
    ollama_host: str,
):
    article_vectorstore = vector_store(
        model, ollama_host, ACADEMIC_PUBLICATIONS_COLLECTION
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(article.content)
    print(f"Processing article: {article.title} -> {len(chunks)} chunks")

    documents = []
    for idx, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "title": article.title,
                    "authors": article.authors,
                    "chunk_id": idx,
                    "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, article.title)) + str(idx),
                },
            )
        )
    article_vectorstore.add_documents(
        documents, ids=[doc.metadata["id"] for doc in documents]
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
    confirm_model_schema(model, engine)

    try:
        articles = get_unembedded_articles(Session(), EmbeddingTable, batch_size)
        if not articles:
            print("No articles to embed")
            return
        else:
            print(f"Found {len(articles)} unembedded articles")

        total_chunks = 0
        for article in articles:
            total_chunks += embed_article(article, model, ollama_host)

        print(
            f"Embedded {total_chunks} chunks from {len(articles)} articles with model '{model}'"
        )

    except Exception as e:
        print(f"An error occurred in the embedding pipeline: {e}")
        raise e

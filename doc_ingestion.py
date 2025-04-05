import os
from pathlib import Path
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from db.constants import DB_NAME
from db.models import Article


def read_pdf(file_path: str) -> str:
    """Reads the content of a PDF file"""
    return "test content2"


def read_article_batch(batch: list[Path]):
    articles = []

    for file_path in batch:
        print(f"Processing {file_path}")
        content = read_pdf(str(file_path))
        if not content:
            continue

        article = Article(
            title=file_path.stem,
            content=content,
            authors=[],
        )
        articles.append(article)

    return articles


def flush_article_batch(batch: list[Article], session):
    table = inspect(Article).local_table

    # Convert objects to dictionaries
    insert_values = [article.model_dump() for article in batch]
    stmt = insert(Article).values(insert_values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["title"],
        set_={
            "content": stmt.excluded.content,
        },
    )
    session.execute(stmt)
    session.commit()


def ingest_directory(dir_path: str, scan_recursively: bool = True):
    """Ingest all the PDF documents contained at `dir_path`. Stores them in the article table"""
    url = f"postgresql://admin:password@localhost:5432/{DB_NAME}"
    engine = create_engine(url)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        path = Path(dir_path)
        pattern = "**/*.pdf" if scan_recursively else "*.pdf"
        pdf_files = list(
            path.rglob(pattern) if scan_recursively else path.glob(pattern)
        )

        batch_size = 50
        for i in range(0, len(pdf_files), batch_size):
            articles = read_article_batch(pdf_files[i : i + batch_size])
            flush_article_batch(articles, session)
            print(f"Processed {len(articles)} files")

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

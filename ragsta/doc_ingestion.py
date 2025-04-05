import os
from pathlib import Path
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
import fitz
from db.constants import get_engine
from db.models import Article


def read_pdf(file_path: str) -> Article:
    """Reads the content of a PDF file"""
    try:
        doc = fitz.open(file_path)
        raw_text = "".join(page.get_text() for page in doc)
        clean_text = raw_text.replace("\x00", "")
        title = doc.metadata.get("title", Path(file_path).stem)
        title = Path(file_path).stem

        return Article(title=title, content=clean_text, authors=[])
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None
    finally:
        if "doc" in locals():
            doc.close()


def read_article_batch(batch: list[Path]) -> list[Article]:
    articles = []

    for file_path in batch:
        print(f"Processing {file_path}")
        article = read_pdf(str(file_path))
        if not article:
            continue

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
    engine = get_engine("admin", "password", "localhost", 5432)
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

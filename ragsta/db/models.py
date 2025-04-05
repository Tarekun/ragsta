from sqlalchemy import inspect, Column, Float, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector
import time
from db.initialization import Base
from db.constants import Base, BRONZE_SCHEMA, EMBEDDINGS_SCHEMA, TABLE_RAW_ARTICLES


class Article(Base):
    __tablename__ = TABLE_RAW_ARTICLES
    __table_args__ = {"schema": BRONZE_SCHEMA}

    title = Column(String, primary_key=True)
    content = Column(String, nullable=False)
    authors = Column(ARRAY(String), nullable=False, default=[])

    def model_dump(self):
        return {"title": self.title, "content": self.content, "authors": self.authors}


def get_embedding_table_for(model: str, engine):
    model_table_name = f"article_{model}"

    class EmbeddedArticleModel(Base):
        __tablename__ = model_table_name
        __table_args__ = {"schema": EMBEDDINGS_SCHEMA}

        article_title = Column(String, primary_key=True)
        chunk_index = Column(Integer, primary_key=True)
        embedding = Column(Vector(768))
        created_at = Column(Float, default=lambda: time.time())

        # @property
        # def embedder():
        #     return model

    inspector = inspect(engine)
    if not inspector.has_table(model_table_name, schema=EMBEDDINGS_SCHEMA):
        EmbeddedArticleModel.__table__.create(bind=engine, checkfirst=True)

    return EmbeddedArticleModel

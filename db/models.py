from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import ARRAY
from db.initialization import Base
from db.constants import Base, BRONZE_SCHEMA, TABLE_RAW_ARTICLES


class Article(Base):
    __tablename__ = TABLE_RAW_ARTICLES
    __table_args__ = {"schema": BRONZE_SCHEMA}

    title = Column(String, primary_key=True)
    content = Column(String, nullable=False)
    authors = Column(ARRAY(String), nullable=False, default=[])

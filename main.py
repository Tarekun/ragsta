from db.initialization import article_db_init, db_init
from doc_ingestion import ingest_directory

db_init("admin", "password")
article_db_init("admin", "password")

ingest_directory("C:\\Users\\danie\\uni\\mate")

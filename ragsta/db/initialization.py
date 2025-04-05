from sqlalchemy import create_engine, text
from sqlalchemy import inspect
from db.constants import *
from db.models import *


def db_init(username: str, password: str, host: str = "localhost", port: int = 5432):
    import psycopg2

    try:
        # connect to the default 'postgres' database
        conn = psycopg2.connect(
            dbname="postgres", user=username, password=password, host=host, port=port
        )
        # enable autocommit for CREATE DATABASE
        conn.autocommit = True
        cursor = conn.cursor()

        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname='{DB_NAME}'")
        exists = cursor.fetchone()

        if exists:
            print(f"Database '{DB_NAME}' already exists")
        else:
            print(f"Creating database '{DB_NAME}'...")
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database '{DB_NAME}' created successfully")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()


def pgvector_init(
    username: str = "admin",
    password: str = "password",
    host: str = "localhost",
    port: int = 5432,
):
    engine = get_engine(username, password, host, port)
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        ).fetchone()

        if not result:
            print("pgvector not found, creating extension...")
            conn.execute(text("CREATE EXTENSION vector"))
        else:
            print("pgvector is already enabled.")

    engine.dispose()


def article_db_init(
    username: str,
    password: str,
    host: str = "localhost",
    port: int = 5432,
):
    engine = get_engine(username, password, host, port)
    inspector = inspect(engine)

    for schema in schemas:
        if not inspector.has_schema(schema):
            print(f"Creating schema '{schema}'...")
            with engine.connect() as conn:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
                conn.commit()
        else:
            print(f"Schema '{schema}' already exists")

    if not inspector.has_table(TABLE_RAW_ARTICLES, schema=BRONZE_SCHEMA):
        print("Creating table...")
        Base.metadata.create_all(engine)
        print(f"Table '{BRONZE_SCHEMA}.{TABLE_RAW_ARTICLES}' created")
    else:
        print(f"Table '{BRONZE_SCHEMA}.{TABLE_RAW_ARTICLES}' already exists")

import os

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_USERNAME = os.getenv("DATABASE_USERNAME")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")
DATABASE_NAME = os.getenv("DATABASE_NAME")


def create_db_if_not_exists():
    conn = psycopg2.connect(
        dbname='postgres',
        user=DATABASE_USERNAME,
        password=DATABASE_PASSWORD,
        host=DATABASE_URL.replace('http://', '').replace('https://', '')
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DATABASE_NAME}';")
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {DATABASE_NAME} WITH ENCODING 'UTF8' TEMPLATE template0;")
    conn.close()


create_db_if_not_exists()

# SQLAlchemy DB Connection with UTF-8 encoding
SQLALCHEMY_DATABASE_URL = f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_URL}/{DATABASE_NAME}?client_encoding=utf8"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

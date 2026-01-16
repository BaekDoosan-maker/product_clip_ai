
import os
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DATABASE_URL = os.environ.get("DATABASE_URL")


def get_connection():

    if DATABASE_URL:
        conn_str = DATABASE_URL
    else:
        if not (DB_HOST and DB_NAME and DB_USER and DB_PASSWORD):
            raise RuntimeError("Database configuration 오류")
        user = quote_plus(DB_USER)
        password = quote_plus(DB_PASSWORD)
        conn_str = f"dbname={DB_NAME} user={user} password={password} host={DB_HOST} port={DB_PORT}"

    conn = psycopg2.connect(conn_str)
    return conn


def fetch_products(limit: int = 100) -> List[Dict[str, Any]]:

    query = "SELECT id, image_url, metadata FROM product LIMIT %s"

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (limit,))
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    finally:
        conn.close()


import os
import sys
import json
from typing import List, Tuple

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import numpy as np

load_dotenv()

SRC_DB_HOST = os.getenv("SRC_DB_HOST", "localhost")
SRC_DB_PORT = int(os.getenv("SRC_DB_PORT", "5432"))
SRC_DB_NAME = os.getenv("SRC_DB_NAME", "mall")
SRC_DB_USER = os.getenv("SRC_DB_USER")
SRC_DB_PASS = os.getenv("SRC_DB_PASS")
SRC_PRODUCT_QUERY = os.getenv("SRC_PRODUCT_QUERY", "SELECT id, image_url FROM product WHERE is_active = true LIMIT 100;")

TGT_DB_HOST = os.getenv("TGT_DB_HOST", "localhost")
TGT_DB_PORT = int(os.getenv("TGT_DB_PORT", "5433"))
TGT_DB_NAME = os.getenv("TGT_DB_NAME", "productdb")
TGT_DB_USER = os.getenv("TGT_DB_USER")
TGT_DB_PASS = os.getenv("TGT_DB_PASS")

# 필수 자격증명 체크: 누락시 익셉션으로 중단(로그 출력 없음)
if not SRC_DB_USER or not SRC_DB_PASS:
    raise RuntimeError("소스 DB 자격증명(SRC_DB_USER/SRC_DB_PASS)이 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
if not TGT_DB_USER or not TGT_DB_PASS:
    raise RuntimeError("대상 DB 자격증명(TGT_DB_USER/TGT_DB_PASS)이 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")

EMBED_DIM = int(os.getenv("EMBED_DIM", "512"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

def make_conn(host: str, port: int, dbname: str, user: str, password: str):
    return psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)


def compute_embedding_placeholder(image_ref: str, dim: int = EMBED_DIM) -> List[float]:

    vec = np.random.RandomState(abs(hash(image_ref)) % (2 ** 32)).randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-12
    return vec.tolist()


def ensure_target_table(conn):
    sql = f"""
    CREATE TABLE IF NOT EXISTS product_vectors (
        product_vector_id serial PRIMARY KEY,
        product_id varchar(255) NOT NULL,
        vector vector({EMBED_DIM}) NOT NULL,
        created_at timestamptz DEFAULT now(),
        UNIQUE (product_id)
    );
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        cur.execute("ALTER TABLE product_vectors ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();")
        conn.commit()


def fetch_products(conn, query: str) -> List[Tuple]:
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return rows


def upsert_vectors(conn, rows: List[Tuple[str, List[float]]]):
    with conn.cursor() as cur:
        insert_sql = """
        INSERT INTO product_vectors (product_id, vector)
        VALUES (%s, %s::vector)
        ON CONFLICT (product_id) DO UPDATE SET vector = EXCLUDED.vector;
        """
        for product_id, vec in rows:
            vec_text = '[' + ','.join(f"{float(x):.6f}" for x in vec) + ']'  # format
            cur.execute(insert_sql, (str(product_id), vec_text))
        conn.commit()


def main():
    try:
        try:
            src_conn = make_conn(SRC_DB_HOST, SRC_DB_PORT, SRC_DB_NAME, SRC_DB_USER, SRC_DB_PASS)
        except Exception as e:
            raise

        try:
            tgt_conn = make_conn(TGT_DB_HOST, TGT_DB_PORT, TGT_DB_NAME, TGT_DB_USER, TGT_DB_PASS)
        except Exception as e:
            src_conn.close()
            raise

        ensure_target_table(tgt_conn)

        try:
            products = fetch_products(src_conn, SRC_PRODUCT_QUERY)
        except Exception as e:
            src_conn.close(); tgt_conn.close(); raise

        try:
            with src_conn.cursor() as cur:
                cur.execute("SELECT id FROM product WHERE is_active = false")
                inactive_rows = cur.fetchall()
            if inactive_rows:
                inactive_ids = [str(r[0]) for r in inactive_rows]
                with tgt_conn.cursor() as cur:
                    batch_size = 100
                    for i in range(0, len(inactive_ids), batch_size):
                        chunk = inactive_ids[i:i+batch_size]
                        cur.execute("DELETE FROM product_vectors WHERE product_id = ANY(%s)", (chunk,))
                tgt_conn.commit()
        except Exception as e:
            pass

        if not products:
            src_conn.close(); tgt_conn.close(); return

        normalized = []
        for row in products:
            if len(row) == 1:
                pid = row[0]
                imgref = str(row[0])
            else:
                pid = row[0]
                imgref = row[1]
            normalized.append((pid, imgref))

        batch = []
        total = 0
        for pid, imgref in normalized:
            emb = compute_embedding_placeholder(str(imgref))
            batch.append((pid, emb))
            if len(batch) >= BATCH_SIZE:
                upsert_vectors(tgt_conn, batch)
                total += len(batch)
                batch = []
        if batch:
            upsert_vectors(tgt_conn, batch)
            total += len(batch)

        src_conn.close(); tgt_conn.close()
    except Exception as e:
        raise


if __name__ == '__main__':
    main()


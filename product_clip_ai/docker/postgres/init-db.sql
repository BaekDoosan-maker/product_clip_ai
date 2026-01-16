CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS product_vectors (
    product_vector_id serial PRIMARY KEY,
    product_id varchar(100) NOT NULL,
    vector vector(512) NOT NULL,
    UNIQUE (product_id)
);


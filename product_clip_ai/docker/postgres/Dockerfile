FROM postgres:16

RUN apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      ca-certificates \
      postgresql-server-dev-16; \
    git clone --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector; \
    cd /tmp/pgvector; \
    make; \
    make install; \
    cd /; \
    rm -rf /tmp/pgvector; \
    apt-get remove -y git build-essential postgresql-server-dev-16; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

COPY init.sql /docker-entrypoint-initdb.d/

EXPOSE 5432


# syntax=docker/dockerfile:1

# Stage 1 – build & install dependencies
FROM python:3.10-slim AS builder
WORKDIR /app

# 1) System libraries: build tools + OpenMP/BLAS for torch & sklearn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    libgomp1 libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# 2) Poetry CLI
RUN pip install --no-cache-dir poetry==1.8.1

# 3) Copy only lockfiles for dependency export
COPY pyproject.toml poetry.lock ./

# 4) Export all locked deps to requirements.txt
RUN poetry export --format=requirements.txt --without-hashes > requirements.txt

# 5) Remove torch line so we can install it via pip
RUN sed -i '/^torch==/d' requirements.txt

# 6) Install everything except torch
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# 7) Install CPU-only torch wheel (no “+cpu” suffix) so pip actually matches the one
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2  ## :contentReference[oaicite:0]{index=0}

# 8) Copy your application code
COPY . .


# Stage 2 – minimal runtime
FROM python:3.10-slim
WORKDIR /app

# 9) Only the runtime libs that torch needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# 10) Copy installed Python packages
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# 11) Copy your app
COPY --from=builder /app /app

# 12) Expose & run
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]  ## :contentReference[oaicite:1]{index=1}

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer caches independently of code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app/ .

# Precomputed artifacts (embeddings, features, honeypot flags) — needed at runtime
COPY artifacts/ ./artifacts/

# Candidates file is mounted at runtime, not baked into the image:
#   docker run -v D:\India.run\resume_ranker\data:/data <image>
ENV DATA_DIR=/data

CMD ["python", "rank.py"]

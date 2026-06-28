# PaperSage — Streamlit app on Hugging Face Spaces (Docker SDK).
FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch first (no GPU on the free Space; keeps the image small + avoids CUDA).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Then the rest of the dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + slim runtime data (the upload step already excluded .env, chats.db, raw PDFs, models).
COPY . .

# Writable caches; the reranker/embedder download here at first start.
ENV HOME=/app \
    HF_HOME=/app/.cache/huggingface \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Hugging Face Spaces expects the app on port 7860.
EXPOSE 7860

CMD ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]

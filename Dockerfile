# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim

# 1. Copy uv binary directly from the official image (Fastest method)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

EXPOSE 8000

# 2. Optimization Configuration
# UV_COMPILE_BYTECODE=1: Compiles python files during install for faster app startup
# UV_LINK_MODE=copy: Ensures files are copied (safer for Docker layers)
# PYTHONUNBUFFERED=1: Turns off buffering for easier container logging
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PYTHONUNBUFFERED=1

# 3. Install dependencies using uv into system Python
# We removed "PYTHONDONTWRITEBYTECODE=1" because we WANT compiled files for speed now
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "src.main:app"]
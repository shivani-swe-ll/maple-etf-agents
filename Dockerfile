# Maple ETF Agents - one image for both the REST API and the web app.
#   docker build -t maple-etf-agents .
#   docker run -p 8000:8000 maple-etf-agents                       # API  -> http://localhost:8000/docs
#   docker run -p 8501:8501 maple-etf-agents \
#       streamlit run app/streamlit_app.py --server.address=0.0.0.0 # Web  -> http://localhost:8501
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer between code changes.
COPY pyproject.toml README.md ./
COPY maple_etf/__init__.py maple_etf/__init__.py
RUN pip install --upgrade pip && pip install ".[api,ui]" && pip uninstall -y maple-etf-agents

# Now copy the code.
COPY maple_etf ./maple_etf
COPY app ./app
COPY evals ./evals

# Run as a normal user, not root.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "maple_etf.api:app", "--host", "0.0.0.0", "--port", "8000"]

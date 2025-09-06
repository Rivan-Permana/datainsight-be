from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "CSV Analyzer API"
    VERSION: str = "1.0.0"

    # GCP
    GCP_PROJECT_ID: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    GCS_BUCKET: str = os.getenv("GCS_BUCKET", "")              # wajib diisi di env Cloud Run
    FIRESTORE_COLLECTION: str = os.getenv("FIRESTORE_COLLECTION", "tasks")

    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = "gpt-5"
    LLM_SEED: int = 1

    # App
    CORS_ORIGINS: List[str] = ["*"]          # batasi di produksi
    CHARTS_DIR: str = "/tmp/charts"          # penyimpanan sementara di Cloud Run

    # Upload guard (multipart langsung ke API)
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "25"))  # aman < 32MiB (HTTP/1)

    # SSE
    SSE_POLL_INTERVAL_SEC: float = float(os.getenv("SSE_POLL_INTERVAL_SEC", "1.0"))
    SSE_KEEPALIVE_SEC: float = float(os.getenv("SSE_KEEPALIVE_SEC", "15"))

    # Streaming compiler (default global—bisa dioverride per-job via form field 'stream')
    STREAM_COMPILER: bool = os.getenv("STREAM_COMPILER", "true").lower() == "true"
    PARTIAL_FLUSH_CHARS: int = int(os.getenv("PARTIAL_FLUSH_CHARS", "256"))
    PARTIAL_FLUSH_SECONDS: float = float(os.getenv("PARTIAL_FLUSH_SECONDS", "1"))

    class Config:
        case_sensitive = True

settings = Settings()

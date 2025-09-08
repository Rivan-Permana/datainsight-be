# app/core/config.py
from __future__ import annotations

import json, os
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # baca .env saat lokal; forbid extras -> ubah jadi ignore biar lebih tahan
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # <- agar variabel tak dikenal tidak bikin error di masa depan
    )

    # ===== API =====
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "CSV Analyzer API"
    VERSION: str = "1.0.0"

    # ===== Ops (tambahan dari .env.example) =====
    ENVIRONMENT: str = Field("development", validation_alias="ENVIRONMENT")
    LOG_LEVEL: str = Field("INFO", validation_alias="LOG_LEVEL")
    # Hanya untuk lokal bila pakai file SA; ADC tidak butuh ini
    GOOGLE_APPLICATION_CREDENTIALS: str = Field("", validation_alias="GOOGLE_APPLICATION_CREDENTIALS")

    # ===== GCP =====
    GCP_PROJECT_ID: str = Field("", validation_alias="GOOGLE_CLOUD_PROJECT")
    GCS_BUCKET: str = Field("", validation_alias="GCS_BUCKET")  # wajib diisi
    FIRESTORE_COLLECTION: str = Field("tasks", validation_alias="FIRESTORE_COLLECTION")

    # ===== LLM =====
    OPENAI_API_KEY: str = Field("", validation_alias="OPENAI_API_KEY")
    LLM_MODEL: str = "gpt-5"
    LLM_SEED: int = 1

    # ===== App =====
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"], validation_alias="CORS_ORIGINS")
    CHARTS_DIR: str = Field("/tmp/charts", validation_alias="CHARTS_DIR")

    # Upload guard
    MAX_UPLOAD_MB: int = Field(25, validation_alias="MAX_UPLOAD_MB")

    # ===== SSE =====
    SSE_POLL_INTERVAL_SEC: float = Field(1.0, validation_alias="SSE_POLL_INTERVAL_SEC")
    SSE_KEEPALIVE_SEC: float = Field(15.0, validation_alias="SSE_KEEPALIVE_SEC")

    # Streaming compiler
    STREAM_COMPILER: bool = Field(True, validation_alias="STREAM_COMPILER")
    PARTIAL_FLUSH_CHARS: int = Field(256, validation_alias="PARTIAL_FLUSH_CHARS")
    PARTIAL_FLUSH_SECONDS: float = Field(1.0, validation_alias="PARTIAL_FLUSH_SECONDS")

    # ---------- Validators ----------
    @field_validator("STREAM_COMPILER", mode="before")
    @classmethod
    def _coerce_bool(cls, v):
        if isinstance(v, bool) or v is None:
            return True if v is None else v
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_cors(cls, v):
        if v in (None, ""): return ["*"]
        if isinstance(v, list): return v
        s = str(v).strip()
        if s.startswith("["):
            try: return json.loads(s)
            except Exception: pass
        return [p.strip() for p in s.split(",") if p.strip()]

    @field_validator("CHARTS_DIR", mode="after")
    @classmethod
    def _ensure_dir(cls, v):
        try: os.makedirs(v, exist_ok=True)
        except Exception: pass
        return v

    @property
    def MAX_UPLOAD_BYTES(self) -> int:
        return int(self.MAX_UPLOAD_MB) * 1024 * 1024

settings = Settings()

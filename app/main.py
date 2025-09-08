import uuid
import io
import re
import time
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse

from google.cloud import storage

from app.core.config import settings
from app.services import firestore as fstore
from app.services import storage as gcs
from app.services.llm_pipeline import run_pipeline

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Helpers ----------
def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

def _sse_event(event: str, data: dict | str) -> str:
    payload = json.dumps(data, ensure_ascii=False) if isinstance(data, (dict, list)) else str(data)
    return f"event: {event}\ndata: {payload}\n\n"

def _sse_comment(msg: str) -> str:
    return f": {msg}\n\n"

# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Frontpage ----------
@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# ---------- Resumable Upload Session ----------
@app.post(f"{settings.API_V1_STR}/uploads/session")
def create_upload_session(filename: str = Form(...), content_type: str = Form("text/csv")):
    if not settings.GCS_BUCKET:
        raise HTTPException(500, "GCS_BUCKET not configured")
    client = storage.Client()
    bucket = client.bucket(settings.GCS_BUCKET)
    blob_path = f"uploads/{uuid.uuid4().hex}-{_sanitize(filename)}"
    session_url = bucket.blob(blob_path).create_resumable_upload_session(content_type=content_type)
    return {"upload_url": session_url, "gcs_uri": f"gs://{settings.GCS_BUCKET}/{blob_path}"}

# ---------- Start Query (file ATAU gcs_uri) + toggle stream per-job ----------
@app.post(f"{settings.API_V1_STR}/data/query")
async def start_query(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(None),
    user_prompt: str = Form(...),
    gcs_uri: Optional[str] = Form(None),
    stream: Optional[str] = Form(None),  # toggle per-job
):
    if not settings.GCS_BUCKET:
        raise HTTPException(500, "GCS_BUCKET not configured")

    if file is None and not gcs_uri:
        raise HTTPException(400, "Provide either 'file' or 'gcs_uri'")

    # guard ukuran multipart
    if file is not None:
        cl = request.headers.get("content-length")
        if cl:
            try:
                if int(cl) > settings.MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(413, f"File too large (> {settings.MAX_UPLOAD_MB} MB)")
            except ValueError:
                pass
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)
        if size > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"File too large (> {settings.MAX_UPLOAD_MB} MB)")

    task_id = uuid.uuid4().hex

    if file is not None:
        blob_path = f"uploads/{task_id}.csv"
        gcs.upload_fileobj(settings.GCS_BUCKET, blob_path, file.file, content_type="text/csv")
        src_blob_path = blob_path
        stored_gcs_uri = f"gs://{settings.GCS_BUCKET}/{blob_path}"
    else:
        if not gcs_uri.startswith("gs://"):
            raise HTTPException(400, "Invalid gcs_uri")
        bucket, path = gcs_uri[5:].split("/", 1)
        if bucket != settings.GCS_BUCKET:
            raise HTTPException(400, "gcs_uri bucket mismatch")
        src_blob_path = path
        stored_gcs_uri = gcs_uri

    stream_compiler = settings.STREAM_COMPILER if stream is None else _truthy(stream)

    fstore.init_task(
        settings.FIRESTORE_COLLECTION,
        task_id,
        {
            "user_prompt": user_prompt,
            "gcs_uri": stored_gcs_uri,
            "chart_url": None,
            "result": None,
            "partial_result": "" if stream_compiler else None,
            "stream": stream_compiler,
        },
    )

    background_tasks.add_task(process_task, task_id, user_prompt, src_blob_path, stream_compiler)
    return JSONResponse({"task_id": task_id, "status": "queued"})

# ---------- Status (REST) ----------
@app.get(f"{settings.API_V1_STR}/status/{{task_id}}")
def get_status(task_id: str):
    data = fstore.get_task(settings.FIRESTORE_COLLECTION, task_id)
    if not data:
        raise HTTPException(404, "task not found")
    return data

# ---------- Chart ----------
@app.get(f"{settings.API_V1_STR}/chart/{{task_id}}")
def get_chart(task_id: str):
    blob_path = f"charts/{task_id}.png"
    tmp_path = Path(f"/tmp/{task_id}.png")
    try:
        gcs.download_to_path(settings.GCS_BUCKET, blob_path, tmp_path)
    except Exception:
        raise HTTPException(404, "chart not found")
    return FileResponse(str(tmp_path), media_type="image/png")

# ---------- SSE status stream ----------
@app.get(f"{settings.API_V1_STR}/status/stream/{{task_id}}")
def stream_status(task_id: str):
    def event_gen():
        last_sent_status = None
        last_partial = None
        last_keepalive = time.time()
        # percepat polling tanpa mengubah .env
        POLL = max(0.10, min(0.25, settings.SSE_POLL_INTERVAL_SEC))
        while True:
            data = fstore.get_task(settings.FIRESTORE_COLLECTION, task_id)
            now = time.time()

            if not data:
                yield _sse_event("error", {"error": "task not found"})
                return

            status = data.get("status")
            partial = data.get("partial_result")
            chart_url = data.get("chart_url")
            result = data.get("result")

            if status != last_sent_status:
                last_sent_status = status
                yield _sse_event("status", {"status": status})

            if partial and partial != last_partial and status == "processing":
                last_partial = partial
                yield _sse_event("partial", {"text": partial})

            if status == "completed":
                yield _sse_event("completed", {"status": status, "result": result, "chart_url": chart_url})
                return
            if status == "failed":
                yield _sse_event("failed", {"status": status, "error": data.get("error")})
                return

            if now - last_keepalive >= settings.SSE_KEEPALIVE_SEC:
                last_keepalive = now
                yield _sse_comment("keepalive")

            time.sleep(POLL)

    return StreamingResponse(event_gen(), media_type="text/event-stream; charset=utf-8")

# ---------- Worker ----------
def process_task(task_id: str, user_prompt: str, csv_blob_path: str, stream_compiler: bool):
    fstore.set_status(settings.FIRESTORE_COLLECTION, task_id, "processing")

    try:
        csv_bytes = gcs.download_bytes(settings.GCS_BUCKET, csv_blob_path)
        df = pd.read_csv(io.BytesIO(csv_bytes))
        charts_dir = Path(settings.CHARTS_DIR)

        if stream_compiler:
            partial_buf: list[str] = []
            last_flush_t = time.time()

            # low-latency local thresholds (tidak ubah .env)
            FLUSH_CHARS = min(128, settings.PARTIAL_FLUSH_CHARS)
            FLUSH_SECS = min(0.25, settings.PARTIAL_FLUSH_SECONDS)

            def on_token(tok: str):
                nonlocal partial_buf, last_flush_t
                if not tok:
                    return
                partial_buf.append(tok)
                now = time.time()
                should_flush = (
                    sum(len(x) for x in partial_buf) >= FLUSH_CHARS
                    or (now - last_flush_t) >= FLUSH_SECS
                    or tok.endswith((".", "!", "?", "\n"))  # flush di akhir kalimat
                )
                if should_flush:
                    text = (fstore.get_task(settings.FIRESTORE_COLLECTION, task_id) or {}).get("partial_result", "") or ""
                    text += "".join(partial_buf)
                    partial_buf = []
                    last_flush_t = now
                    fstore.set_status(
                        settings.FIRESTORE_COLLECTION,
                        task_id,
                        "processing",
                        {"partial_result": text},
                    )

            final_text, chart_path, _ = run_pipeline(
                df, user_prompt, charts_dir, stream_compiler=True, on_token=on_token
            )

            # flush sisa buffer
            if partial_buf:
                text = (fstore.get_task(settings.FIRESTORE_COLLECTION, task_id) or {}).get("partial_result", "") or ""
                text += "".join(partial_buf)
                fstore.set_status(
                    settings.FIRESTORE_COLLECTION,
                    task_id,
                    "processing",
                    {"partial_result": text},
                )
        else:
            final_text, chart_path, _ = run_pipeline(
                df, user_prompt, charts_dir, stream_compiler=False, on_token=None
            )

        chart_url: Optional[str] = None
        if chart_path and chart_path.exists():
            dest_blob = f"charts/{task_id}.png"
            gcs.upload_bytes(
                settings.GCS_BUCKET, dest_blob, chart_path.read_bytes(), content_type="image/png"
            )
            chart_url = f"{settings.API_V1_STR}/chart/{task_id}"

        extra = {"result": final_text, "chart_url": chart_url}
        if stream_compiler:
            extra["partial_result"] = final_text
        fstore.set_status(settings.FIRESTORE_COLLECTION, task_id, "completed", extra)

    except Exception as e:
        fstore.set_status(settings.FIRESTORE_COLLECTION, task_id, "failed", {"error": str(e)})

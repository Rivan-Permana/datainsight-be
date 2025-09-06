from google.cloud import storage
from typing import Optional
from pathlib import Path

def _client() -> storage.Client:
    return storage.Client()

def upload_bytes(bucket_name: str, blob_path: str, data: bytes, content_type: Optional[str] = None) -> str:
    client = _client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{bucket_name}/{blob_path}"

def upload_fileobj(bucket_name: str, blob_path: str, fileobj, content_type: Optional[str] = None) -> str:
    client = _client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_file(fileobj, content_type=content_type)
    return f"gs://{bucket_name}/{blob_path}"

def download_bytes(bucket_name: str, blob_path: str) -> bytes:
    client = _client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_bytes()

def download_to_path(bucket_name: str, blob_path: str, local_path: Path) -> Path:
    client = _client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    return local_path

from google.cloud import firestore
from datetime import datetime
from typing import Dict, Any, Optional

def _client() -> firestore.Client:
    return firestore.Client()

def init_task(collection: str, task_id: str, data: Dict[str, Any]):
    db = _client()
    doc = db.collection(collection).document(task_id)
    now = datetime.utcnow().isoformat()
    payload = {**data, "status": "queued", "created_at": now, "updated_at": now}
    doc.set(payload)

def set_status(collection: str, task_id: str, status: str, extra: Optional[Dict[str, Any]] = None):
    db = _client()
    doc = db.collection(collection).document(task_id)
    now = datetime.utcnow().isoformat()
    payload = {"status": status, "updated_at": now}
    if extra:
        payload.update(extra)
    doc.set(payload, merge=True)

def get_task(collection: str, task_id: str) -> Optional[Dict[str, Any]]:
    db = _client()
    snap = db.collection(collection).document(task_id).get()
    return snap.to_dict() if snap.exists else None

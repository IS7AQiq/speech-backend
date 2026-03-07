# main.py - OPTIMIZED VERSION WITH FIXED STREAMING & PROPER SESSION MANAGEMENT
import logging
import os
import json
import time
import shutil
import tempfile
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from database import db
from whisper_service import WhisperTranscriber

# ── App lifecycle ─────────────────────────────────────────────────
whisper_transcriber: WhisperTranscriber | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_transcriber
    model_size = os.getenv("WHISPER_MODEL", "base")
    try:
        whisper_transcriber = WhisperTranscriber(model_size=model_size)
        logger.info("✅ Whisper ready")
    except Exception as e:
        logger.error(f"⚠️ Whisper initialization failed: {e}")
        whisper_transcriber = None
    yield
    # Shutdown cleanup

app = FastAPI(
    title="Arabic Speech Therapy API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# ── Timing middleware ──────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.monotonic() - start:.4f}"
    return response

# ── Timing middleware ──────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════
# Root / info
# ═══════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
def root():
    return JSONResponse(content={
        "message": "🎯 Arabic Speech Therapy API",
        "status": "online",
        "whisper_ready": whisper_transcriber is not None,
        "endpoints": {
            "categories":        "GET  /categories",
            "words_by_category": "GET  /categories/{id}/words",
            "transcribe":        "POST /transcribe",
            "transcribe_bytes":  "POST /transcribe-bytes",
            "vosk_status":       "GET  /vosk-status",
            "health":            "GET  /health",
        },
        "timestamp": time.time(),
    })

# ═══════════════════════════════════════════════════════════════════
# Database endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/categories", tags=["Database"])
def get_categories():
    """Return all vocabulary categories."""
    try:
        categories = db.get_categories()
        return JSONResponse(content={
            "success": True,
            "count": len(categories),
            "data": categories,
            "timestamp": time.time(),
        })
    except Exception as e:
        logger.error(f"❌ get_categories error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories/{category_id}/words", tags=["Database"])
def get_words_by_category(category_id: int):
    """Return all words in a given category."""
    try:
        words = db.get_words_by_category(category_id)
        return JSONResponse(content={
            "success": True,
            "count": len(words),
            "category_id": category_id,
            "data": words,
            "timestamp": time.time(),
        })
    except Exception as e:
        logger.error(f"❌ get_words_by_category error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════
# Speech recognition endpoints
# ═══════════════════════════════════════════════════════════════════

def _require_whisper():
    """Raise HTTP 503 if Whisper failed to initialise."""
    if whisper_transcriber is None:
        raise HTTPException(
            status_code=503,
            detail="Whisper model not available. Check server logs for details.",
        )


from fastapi import Depends, Header
from typing import Optional

def _get_current_user(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Extracts user_id from Supabase JWT if passed in the Authorization header"""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        return db.verify_token(token)
    return None

@app.post("/transcribe", tags=["Speech Recognition"])
async def transcribe_audio(
    file: UploadFile = File(...),
    return_words: bool = Form(False),
    category_id: Optional[int] = Form(None),
    user_id: Optional[str] = Depends(_get_current_user)
):
    """Upload an audio file (WAV/MP4/OGG etc.) for full transcription."""
    _require_whisper()

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        # Whisper runs ffmpeg externally which needs the file to not be locked by python on Windows
        result = whisper_transcriber.transcribe_file(tmp_path, return_words=return_words)
        
        # If user is authenticated, save the transcript to Supabase
        if user_id and return_words and "stuttering_analysis" in result:
             # Optionally upload audio to Supabase Storage
             audio_url = db.upload_audio(user_id, tmp_path)
             
             db.save_transcription(
                 user_id=user_id,
                 audio_path=tmp_path,
                 transcript=result["text"],
                 stuttering_data=result["stuttering_analysis"],
                 category_id=category_id,
                 audio_url=audio_url
             )
             
        return result
    except Exception as e:
        logger.error(f"❌ transcribe_audio error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/transcribe-bytes", tags=["Speech Recognition"])
async def transcribe_bytes(
    audio: bytes = File(...),
    return_words: bool = Form(False),
    category_id: Optional[int] = Form(None),
    user_id: Optional[str] = Depends(_get_current_user)
):
    """Direct binary audio upload."""
    _require_whisper()
    suffix = ".wav"
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio)
            tmp_path = tmp.name
        
        result = whisper_transcriber.transcribe_file(tmp_path, return_words=return_words)
        
        # If user is authenticated, save the transcript to Supabase
        if user_id and return_words and "stuttering_analysis" in result:
             # Optionally upload audio to Supabase Storage
             audio_url = db.upload_audio(user_id, tmp_path)
             
             db.save_transcription(
                 user_id=user_id,
                 audio_path=tmp_path,
                 transcript=result["text"],
                 stuttering_data=result["stuttering_analysis"],
                 category_id=category_id,
                 audio_url=audio_url
             )
             
        return result
    except Exception as e:
        logger.error(f"❌ transcribe_bytes error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.get("/transcriptions", tags=["Transcriptions"])
async def get_transcriptions(user_id: str = Depends(_get_current_user)):
    """Get all past transcriptions for the authenticated user."""
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    transcriptions = db.get_user_transcriptions(user_id)
    if transcriptions is None:
        raise HTTPException(status_code=500, detail="Failed to fetch transcriptions")
        
    return {
        "success": True,
        "count": len(transcriptions),
        "data": transcriptions
    }

@app.get("/transcription/{id}", tags=["Transcriptions"])
async def get_transcription(id: int, user_id: str = Depends(_get_current_user)):
    """Get a specific transcription by ID for the authenticated user."""
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    transcription = db.get_transcription_by_id(user_id, id)
    if transcription is None:
        raise HTTPException(status_code=404, detail="Transcription not found")
        
    return {
        "success": True,
        "data": transcription
    }

@app.delete("/transcription/{id}", tags=["Transcriptions"])
async def delete_transcription(id: int, user_id: str = Depends(_get_current_user)):
    """Delete a specific transcription by ID for the authenticated user."""
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    success = db.delete_transcription_by_id(user_id, id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete transcription")
        
    return {
        "success": True,
        "message": "Transcription deleted successfully"
    }

# Streams and websocket are no longer supported because Whisper natively expects a full file

# ═══════════════════════════════════════════════════════════════════
# Status / health
# ═══════════════════════════════════════════════════════════════════

@app.get("/vosk-status", tags=["Status"])
async def vosk_status():
    """Kept the name vosk-status for flutter compatibility"""
    return {
        "initialized": whisper_transcriber is not None,
        "model": "whisper" if whisper_transcriber else None,
        "timestamp": time.time(),
    }


@app.get("/health", tags=["Status"])
def health_check():
    try:
        categories = db.get_categories()
        return {
            "status": "healthy",
            "database": "connected",
            "whisper": "ready" if whisper_transcriber else "not_initialised",
            "categories_count": len(categories),
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "whisper": "unknown",
                "error": str(e),
                "timestamp": time.time(),
            },
        )


@app.get("/test", tags=["Status"])
def test_endpoint():
    return {"message": "API is working!", "timestamp": time.time()}


# ── Dev entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
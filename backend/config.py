"""Settings, read once from backend/.env and the environment.

Secrets live here and are never exposed through any route or WebSocket message.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
REPO_DIR = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")

# The fixed demo vocabulary: the only labels collection accepts and the model should know.
VOCAB = ["help", "yes", "no", "Jordan", "water"]
NOTHING = "nothing"  # "no word" class, recorded alongside VOCAB


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


@dataclass(frozen=True)
class Settings:
    # sensors
    fake_stream: bool = field(default_factory=lambda: _bool("FAKE_STREAM", False))
    serial_port: str = field(default_factory=lambda: _str("SERIAL_PORT"), repr=False)
    serial_baud: int = field(default_factory=lambda: int(_float("SERIAL_BAUD", 921600)))
    # secrets (repr=False so they never land in logs by accident)
    tiger_db_url: str = field(default_factory=lambda: _str("TIGER_DB_URL"), repr=False)
    elevenlabs_api_key: str = field(default_factory=lambda: _str("ELEVENLABS_API_KEY"), repr=False)
    elevenlabs_voice_id: str = field(default_factory=lambda: _str("ELEVENLABS_VOICE_ID"), repr=False)
    elevenlabs_model: str = field(default_factory=lambda: _str("ELEVENLABS_MODEL", "eleven_flash_v2_5"))
    store_raw_stream: bool = field(default_factory=lambda: _bool("STORE_RAW_STREAM", True))
    # learn mode
    whisper_model: str = field(default_factory=lambda: _str("WHISPER_MODEL", "base.en"))
    whisper_language: str = field(default_factory=lambda: _str("WHISPER_LANGUAGE", "en"))
    learn_pad_ms: float = field(default_factory=lambda: _float("LEARN_PAD_MS", 80))
    learn_audio_offset_ms: float = field(default_factory=lambda: _float("LEARN_AUDIO_OFFSET_MS", 0))
    # inference
    conf_threshold: float = field(default_factory=lambda: _float("CONF_THRESHOLD", 0.6))
    phrase_pause_s: float = field(default_factory=lambda: _float("PHRASE_PAUSE_S", 1.5))
    activity_threshold: float = field(default_factory=lambda: _float("ACTIVITY_THRESHOLD", 0.12))
    single_word: bool = field(default_factory=lambda: _bool("SINGLE_WORD", True))  # each word is spoken on its own, no combos
    # on-site calibration (Calibrate tab)
    calib_words: tuple[str, ...] = field(
        default_factory=lambda: tuple(w.strip() for w in _str("CALIB_WORDS", "help,Jordan,no,nothing").split(",") if w.strip())
    )
    calib_takes: int = field(default_factory=lambda: int(_float("CALIB_TAKES", 10)))
    calib_weight: float = field(default_factory=lambda: _float("CALIB_WEIGHT", 5))
    # server
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip() for o in _str("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()
        )
    )
    public_base_url: str = field(default_factory=lambda: _str("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/"))
    data_dir: Path = field(default_factory=lambda: Path(_str("DATA_DIR", str(REPO_DIR / "data"))))
    # MODEL_DIR (absolute, or relative to the repo) lets you try another model without overwriting backend/models
    models_dir: Path = field(default_factory=lambda: (REPO_DIR / _str("MODEL_DIR", "backend/models")))


settings = Settings()

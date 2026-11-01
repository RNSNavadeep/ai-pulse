import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Config:
    """Central configuration for AI Pulse Agent."""

    # Telegram Bot
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    # Agent Polling & Delivery Limits
    POLL_INTERVAL_SECONDS: int = int(os.getenv("POLL_INTERVAL_SECONDS", "300"))
    MAX_NOTIFICATIONS_PER_RUN: int = int(os.getenv("MAX_NOTIFICATIONS_PER_RUN", "5"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # Retry and Fault Tolerance
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_BACKOFF_SECONDS: float = float(os.getenv("RETRY_BACKOFF_SECONDS", "2.0"))

    # Initial Run Behavior (prevent flood of historical articles on empty database)
    INITIAL_RUN_SEND_ALERTS: bool = os.getenv("INITIAL_RUN_SEND_ALERTS", "false").lower() in ("true", "1", "yes")

    # Local Data and SQLite Storage
    DATA_DIR: Path = BASE_DIR / "data"
    DATABASE_PATH: Path = DATA_DIR / os.getenv("DB_NAME", "ai_pulse.db")

    # Separate sources configuration file
    SOURCES_FILE: Path = BASE_DIR / "sources.json"

    # Phase 4: Gemini LLM Integration (Free-Tier Only)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    GEMINI_MAX_REQUESTS_PER_CYCLE: int = int(os.getenv("GEMINI_MAX_REQUESTS_PER_CYCLE", "5"))
    GEMINI_DAILY_LIMIT: int = int(os.getenv("GEMINI_DAILY_LIMIT", "100"))
    LLM_ENABLED: bool = os.getenv("LLM_ENABLED", "true").lower() in ("true", "1", "yes")
    # Per-request HTTP timeout in milliseconds (25 000 ms = 25 s).
    # Prevents a single slow/hanging Gemini call from blocking the cycle for minutes.
    GEMINI_REQUEST_TIMEOUT_MS: int = int(os.getenv("GEMINI_REQUEST_TIMEOUT_MS", "25000"))
    # How many times the SDK internally retries transient HTTP errors (503/429/408)
    # before surfacing the exception. 0 = 1 total SDK-level attempt (0 retries).
    GEMINI_MAX_SDK_RETRIES: int = int(os.getenv("GEMINI_MAX_SDK_RETRIES", "0"))


    # Default fallback RSS Sources
    DEFAULT_RSS_SOURCES: List[Dict[str, Any]] = [
        {
            "name": "OpenAI News",
            "url": "https://openai.com/news/rss.xml",
            "category": "releases",
            "is_pure_ai": True,
            "enabled": True,
        },
        {
            "name": "Google AI Blog",
            "url": "https://blog.google/technology/ai/rss/",
            "category": "industry",
            "is_pure_ai": True,
            "enabled": True,
        },
        {
            "name": "Google DeepMind",
            "url": "https://deepmind.google/blog/rss.xml",
            "category": "research",
            "is_pure_ai": True,
            "enabled": True,
        },
        {
            "name": "Hugging Face Blog",
            "url": "https://huggingface.co/blog/feed.xml",
            "category": "research",
            "is_pure_ai": True,
            "enabled": True,
        },
        {
            "name": "arXiv AI Research (cs.AI & cs.LG)",
            "url": "https://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.LG&sortBy=submittedDate&sortOrder=descending&max_results=10",
            "category": "research",
            "is_pure_ai": True,
            "enabled": True,
        },
        {
            "name": "MIT Technology Review - AI",
            "url": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
            "category": "industry",
            "is_pure_ai": True,
            "enabled": True,
        },
    ]

    @classmethod
    def get_sources(cls) -> List[Dict[str, Any]]:
        """Load sources from sources.json if present; fallback to DEFAULT_RSS_SOURCES."""
        if cls.SOURCES_FILE.exists():
            try:
                with open(cls.SOURCES_FILE, "r", encoding="utf-8") as f:
                    sources = json.load(f)
                    if isinstance(sources, list) and len(sources) > 0:
                        return sources
            except Exception as e:
                logger.error("Could not load sources from %s: %s", cls.SOURCES_FILE, e)
        return cls.DEFAULT_RSS_SOURCES

    @classmethod
    def is_telegram_configured(cls) -> bool:
        """Check if Telegram credentials have been supplied."""
        return bool(
            cls.TELEGRAM_BOT_TOKEN
            and cls.TELEGRAM_CHAT_ID
            and cls.TELEGRAM_BOT_TOKEN != "your_bot_token_here"
            and cls.TELEGRAM_CHAT_ID != "your_chat_id_here"
        )

    @classmethod
    def is_gemini_configured(cls) -> bool:
        """Check if Gemini API key has been supplied and is valid (not default placeholder)."""
        return bool(
            cls.GEMINI_API_KEY
            and cls.GEMINI_API_KEY != "your_gemini_api_key_here"
            and not cls.GEMINI_API_KEY.startswith("your_")
        )


config = Config()


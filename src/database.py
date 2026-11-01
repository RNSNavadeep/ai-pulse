import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from config import config
from src.collector import normalize_url
from src.models import Article, ArticleAnalysis

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class Database:
    """SQLite persistence layer for AI Pulse news articles and notification tracking."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self._mem_conn: Optional[sqlite3.Connection] = None

        if db_path is None:
            self.db_path = config.DATABASE_PATH
        elif isinstance(db_path, str) and db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path)

        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", timeout=15)
            self._mem_conn.row_factory = sqlite3.Row
        elif isinstance(self.db_path, Path):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return an authenticated SQLite connection with row factory."""
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(str(self.db_path), timeout=15)
        conn.row_factory = sqlite3.Row
        return conn


    def _init_db(self):
        """Create tables and indexes if they do not exist, and safely migrate existing tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    normalized_url TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    published TEXT,
                    summary TEXT,
                    category TEXT DEFAULT 'general',
                    relevance_score REAL DEFAULT 0.0,
                    matched_keywords TEXT DEFAULT '[]',
                    status TEXT DEFAULT 'discovered',
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    discovered_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    ai_summary TEXT,
                    ai_category TEXT,
                    ai_importance INTEGER,
                    ai_relevance INTEGER,
                    why_it_matters TEXT,
                    suggested_action TEXT,
                    score_explanation TEXT,
                    summarization_status TEXT DEFAULT 'pending',
                    llm_model TEXT,
                    llm_processed_at TEXT
                )
                """
            )

            # Safe in-place migration for existing databases
            cursor.execute("PRAGMA table_info(articles)")
            existing_cols = {row["name"] for row in cursor.fetchall()}

            phase4_columns = [
                ("ai_summary", "TEXT"),
                ("ai_category", "TEXT"),
                ("ai_importance", "INTEGER"),
                ("ai_relevance", "INTEGER"),
                ("why_it_matters", "TEXT"),
                ("suggested_action", "TEXT"),
                ("score_explanation", "TEXT"),
                ("summarization_status", "TEXT DEFAULT 'pending'"),
                ("llm_model", "TEXT"),
                ("llm_processed_at", "TEXT"),
            ]

            for col_name, col_type in phase4_columns:
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE articles ADD COLUMN {col_name} {col_type};")
                    logger.debug("Migrated SQLite schema: Added column '%s' to articles", col_name)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(normalized_url);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_score ON articles(relevance_score);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_sum_status ON articles(summarization_status);")
            conn.commit()
            logger.debug("Database initialized at %s", self.db_path)

    def is_empty(self) -> bool:
        """Check whether the articles database contains zero entries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(1) FROM articles")
            count = cursor.fetchone()[0]
            return count == 0

    def is_duplicate(self, article_id: str, normalized_url: str) -> bool:
        """Check if an article already exists by stable ID or normalized URL."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM articles WHERE id = ? OR normalized_url = ? LIMIT 1",
                (article_id, normalized_url),
            )
            return cursor.fetchone() is not None

    def save_article(
        self, article: Article, default_status: str = "discovered"
    ) -> Tuple[bool, str]:
        """
        Save an article if not already present.

        Returns:
            Tuple of (is_new: bool, article_id: str)
        """
        norm_url = article.normalized_url or normalize_url(article.link)
        article_id = article.id or norm_url
        now = utc_now_iso()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Duplicate check
            cursor.execute(
                "SELECT id, status FROM articles WHERE id = ? OR normalized_url = ? LIMIT 1",
                (article_id, norm_url),
            )
            existing = cursor.fetchone()
            if existing:
                return False, existing["id"]

            keywords_json = json.dumps(article.matched_keywords or [])
            cursor.execute(
                """
                INSERT INTO articles (
                    id, normalized_url, title, source, published, summary,
                    category, relevance_score, matched_keywords, status,
                    retry_count, error_message, discovered_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article_id,
                    norm_url,
                    article.title,
                    article.source,
                    article.published,
                    article.summary,
                    article.category or "general",
                    article.relevance_score or 0.0,
                    keywords_json,
                    default_status,
                    0,
                    None,
                    now,
                    now,
                ),
            )
            conn.commit()
            return True, article_id

    def save_articles_batch(
        self, articles: List[Article], default_status: str = "discovered"
    ) -> Tuple[int, int]:
        """
        Save multiple articles in a single transaction.

        Returns:
            Tuple of (new_count: int, duplicate_count: int)
        """
        new_count = 0
        duplicate_count = 0
        for art in articles:
            is_new, _ = self.save_article(art, default_status=default_status)
            if is_new:
                new_count += 1
            else:
                duplicate_count += 1
        return new_count, duplicate_count

    def get_article(self, article_id: str) -> Optional[Article]:
        """Retrieve an article by its unique ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_article(row)

    def get_unprocessed_articles(
        self, limit: int = 5, max_retries: int = 3
    ) -> List[Article]:
        """
        Fetch articles eligible for Telegram notification.
        Excludes ignored articles and retries only eligible failures.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM articles
                WHERE status IN ('discovered', 'pending')
                   OR (status = 'failed' AND retry_count < ?)
                ORDER BY relevance_score DESC, discovered_at ASC
                LIMIT ?
                """,
                (max_retries, limit),
            )
            rows = cursor.fetchall()
            return [self._row_to_article(r) for r in rows]

    def mark_as_pending(self, article_id: str):
        """Mark an article as currently pending dispatch."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE articles SET status = 'pending', updated_at = ? WHERE id = ?",
                (utc_now_iso(), article_id),
            )
            conn.commit()
    def ignore_existing_backlog(self) -> int:
        """Exclude existing queued articles from future notifications."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE articles
                SET status = 'ignored', updated_at = ?
                WHERE status IN ('discovered', 'pending', 'failed')
                """,
                (utc_now_iso(),),
            )
            count = cursor.rowcount
            conn.commit()
            return count


    def mark_as_sent(self, article_id: str):
        """Mark an article as confirmed sent by Telegram."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE articles SET status = 'sent', error_message = NULL, updated_at = ? WHERE id = ?",
                (utc_now_iso(), article_id),
            )
            conn.commit()

    def mark_as_failed(self, article_id: str, error_message: str):
        """Record dispatch failure, increment retry count, and update status to 'failed'."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE articles
                SET status = 'failed',
                    retry_count = retry_count + 1,
                    error_message = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error_message[:400], utc_now_iso(), article_id),
            )
            conn.commit()

    def recover_stale_pending(self, timeout_minutes: int = 15) -> int:
        """
        Recover any articles that remained stuck in 'pending' status due to
        an unexpected shutdown or network disconnect. Resets them to 'discovered'.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Reset pending articles back to discovered
            cursor.execute(
                """
                UPDATE articles
                SET status = 'discovered', updated_at = ?
                WHERE status = 'pending'
                """,
                (utc_now_iso(),),
            )
            recovered = cursor.rowcount
            conn.commit()
            if recovered > 0:
                logger.info("Recovered %d stale pending articles back to 'discovered'", recovered)
            return recovered

    def count_articles(self, status: Optional[str] = None) -> int:
        """Return the count of articles, optionally filtered by status."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute("SELECT COUNT(1) FROM articles WHERE status = ?", (status,))
            else:
                cursor.execute("SELECT COUNT(1) FROM articles")
            return cursor.fetchone()[0]

    def get_stats(self) -> Dict[str, Any]:
        """Return counts of articles by status for monitoring and reporting."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status, COUNT(1) FROM articles GROUP BY status")
            counts = {row[0]: row[1] for row in cursor.fetchall()}
            total = sum(counts.values())

            # Summarization stats
            cursor.execute("PRAGMA table_info(articles)")
            cols = {r["name"] for r in cursor.fetchall()}
            sum_counts: Dict[str, int] = {}
            pending_actionable = 0
            pending_historic = 0

            if "summarization_status" in cols:
                cursor.execute(
                    "SELECT summarization_status, COUNT(1) FROM articles GROUP BY summarization_status"
                )
                sum_counts = {row[0]: row[1] for row in cursor.fetchall() if row[0] is not None}

                # Split pending into actionable vs historic:
                # - Actionable: summarization_status='pending' AND status != 'sent'
                #   These are articles the agent *will* process on the next cycle.
                # - Historic: summarization_status='pending' AND status = 'sent'
                #   These were sent before Phase 4 existed and will never be re-summarized.
                cursor.execute(
                    """
                    SELECT COUNT(1) FROM articles
                    WHERE summarization_status = 'pending' AND status NOT IN ('sent','ignored')
                    """
                )
                pending_actionable = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT COUNT(1) FROM articles
                    WHERE summarization_status = 'pending' AND status = 'sent'
                    """
                )
                pending_historic = cursor.fetchone()[0]

                # Ignored articles whose summarization_status is still 'pending'
                # (ignore_existing_backlog() does not touch summarization_status).
                # These are permanently excluded from both pipelines.
                cursor.execute(
                    """
                    SELECT COUNT(1) FROM articles
                    WHERE summarization_status = 'pending' AND status = 'ignored'
                    """
                )
                pending_ignored = cursor.fetchone()[0]

            return {
                "total": total,
                "discovered": counts.get("discovered", 0),
                "pending": counts.get("pending", 0),
                "sent": counts.get("sent", 0),
                "failed": counts.get("failed", 0),
                "ignored": counts.get("ignored", 0),
                "summaries_completed": sum_counts.get("completed", 0),
                "summaries_fallback": sum_counts.get("fallback", 0),
                "summaries_insufficient": sum_counts.get("insufficient", 0),
                # Preserved for backward compatibility — equals actionable + historic + ignored
                "summaries_pending": sum_counts.get("pending", 0),
                # Split counts for accurate display (these three partition summaries_pending exactly)
                "summaries_pending_actionable": pending_actionable,
                "summaries_pending_historic": pending_historic,
                "summaries_pending_ignored": pending_ignored,
                "summaries_failed": sum_counts.get("failed", 0),
            }

    def save_analysis(
        self,
        article_id: str,
        analysis: ArticleAnalysis,
        model: str,
        status: str = "completed",
    ) -> None:
        """
        Persist structured LLM analysis results into the database.
        Caches the output to prevent redundant API calls.
        """
        now = utc_now_iso()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE articles
                SET ai_summary = ?,
                    ai_category = ?,
                    ai_importance = ?,
                    ai_relevance = ?,
                    why_it_matters = ?,
                    suggested_action = ?,
                    score_explanation = ?,
                    summarization_status = ?,
                    llm_model = ?,
                    llm_processed_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    analysis.summary,
                    analysis.category,
                    analysis.importance_score,
                    analysis.relevance_score,
                    analysis.why_it_matters,
                    analysis.suggested_action,
                    analysis.score_explanation,
                    status,
                    model,
                    now,
                    now,
                    article_id,
                ),
            )
            conn.commit()
            logger.debug("Saved LLM analysis for article %s (status: %s)", article_id, status)

    def save_fallback_analysis(
        self,
        article_id: str,
        summary: str,
        category: str,
        importance: int = 5,
        relevance: int = 5,
        why_it_matters: str = "",
        suggested_action: Optional[str] = None,
        explanation: str = "Rule-based fallback summary",
    ) -> None:
        """
        Save a rule-based fallback summary when the LLM is disabled, offline, or quota-exhausted.
        """
        now = utc_now_iso()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE articles
                SET ai_summary = ?,
                    ai_category = ?,
                    ai_importance = ?,
                    ai_relevance = ?,
                    why_it_matters = ?,
                    suggested_action = ?,
                    score_explanation = ?,
                    summarization_status = 'fallback',
                    llm_model = 'rule-based-fallback',
                    llm_processed_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    summary,
                    category,
                    importance,
                    relevance,
                    why_it_matters,
                    suggested_action,
                    explanation,
                    now,
                    now,
                    article_id,
                ),
            )
            conn.commit()
            logger.debug("Saved fallback analysis for article %s", article_id)

    def mark_summarization_failed(self, article_id: str, error_message: str) -> None:
        """Record an LLM summarization failure without halting the notification pipeline."""
        now = utc_now_iso()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE articles
                SET summarization_status = 'failed',
                    error_message = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error_message[:400], now, article_id),
            )
            conn.commit()

    def get_cached_analysis(self, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached analysis for an article if already processed.
        Returns None if analysis has not been performed yet.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ai_summary, ai_category, ai_importance, ai_relevance,
                       why_it_matters, suggested_action, score_explanation,
                       summarization_status, llm_model, llm_processed_at
                FROM articles
                WHERE id = ? AND summarization_status IN ('completed', 'fallback', 'insufficient')
                  AND ai_summary IS NOT NULL
                LIMIT 1
                """,
                (article_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row)

    def get_articles_needing_summary(self, limit: int = 5) -> List[Article]:
        """
        Retrieve articles that need summarization.
        Crucial requirement: Never summarize articles that have already been sent (status != 'sent')
        just to resend them.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM articles
                WHERE status NOT IN ('sent','ignored')
                  AND (summarization_status IS NULL OR summarization_status IN ('pending', 'failed'))
                ORDER BY relevance_score DESC, discovered_at ASC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [self._row_to_article(r) for r in rows]

    def reset(self):
        """Drop tables and recreate schema (used for tests or administrative resets)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS articles")
            conn.commit()
        self._init_db()

    def _row_to_article(self, row: sqlite3.Row) -> Article:
        """Helper to transform an SQLite Row into an Article Pydantic instance."""
        keywords_raw = row["matched_keywords"] or "[]"
        try:
            keywords = json.loads(keywords_raw)
        except Exception:
            keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]

        keys = row.keys()
        return Article(
            id=row["id"],
            title=row["title"],
            link=row["normalized_url"],
            source=row["source"],
            published=row["published"],
            summary=row["summary"],
            category=row["category"],
            relevance_score=row["relevance_score"],
            matched_keywords=keywords,
            normalized_url=row["normalized_url"],
            status=row["status"],
            retry_count=row["retry_count"],
            error_message=row["error_message"],
            discovered_at=row["discovered_at"],
            updated_at=row["updated_at"],
            ai_summary=row["ai_summary"] if "ai_summary" in keys else None,
            ai_category=row["ai_category"] if "ai_category" in keys else None,
            ai_importance=row["ai_importance"] if "ai_importance" in keys else None,
            ai_relevance=row["ai_relevance"] if "ai_relevance" in keys else None,
            why_it_matters=row["why_it_matters"] if "why_it_matters" in keys else None,
            suggested_action=row["suggested_action"] if "suggested_action" in keys else None,
            score_explanation=row["score_explanation"] if "score_explanation" in keys else None,
            summarization_status=(
                row["summarization_status"]
                if "summarization_status" in keys and row["summarization_status"]
                else "pending"
            ),
            llm_model=row["llm_model"] if "llm_model" in keys else None,
            llm_processed_at=row["llm_processed_at"] if "llm_processed_at" in keys else None,
        )

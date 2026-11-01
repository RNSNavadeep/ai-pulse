import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import config
from src.collector import NewsCollector
from src.database import Database
from src.filter import TopicFilter
from src.models import Article
from src.notifier import TelegramNotifier
from src.summarizer import GeminiSummarizer

logger = logging.getLogger(__name__)


class PollingLock:
    """
    OS-level single-instance guard for the agent polling loop.

    Uses platform file locking (msvcrt on Windows, fcntl on POSIX) as the primary
    mechanism.  The lock is held by keeping a file descriptor open and byte-locked;
    the OS automatically releases it when the process exits for *any* reason
    (clean shutdown, Ctrl+C, crash, or SIGKILL), so stale locks are impossible.

    A PID is also written into the file purely for human diagnostics; it does not
    participate in the locking protocol.
    """

    _LOCK_BYTES = 1  # number of bytes locked (at offset 0)

    def __init__(self, lock_path: Path):
        self.lock_path = Path(lock_path)
        self._fh: Optional[Any] = None  # kept open while lock is held

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_held(self) -> bool:
        """True if this instance currently owns the lock."""
        return self._fh is not None

    def acquire(self) -> Tuple[bool, Optional[str]]:
        """
        Try to acquire the OS-level lock (non-blocking).

        Returns:
            (True, None)              – lock acquired; this process is the sole owner.
            (False, owner_pid_str)    – lock held by another process; *owner_pid_str*
                                        is the PID read from the file (for diagnostics).
        """
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Snapshot the current PID from the file *before* opening in write mode
        # so we can report the owner even if our open() truncates the content.
        existing_pid = self._read_pid()

        fh = None
        try:
            fh = open(self.lock_path, "w")  # creates or truncates; does NOT acquire lock yet
            self._os_lock(fh)               # raises OSError if another process holds the lock

            # We own the lock – write our PID for human diagnostics
            fh.write(str(os.getpid()))
            fh.flush()

            self._fh = fh
            logger.debug("Polling lock acquired by PID %d at %s", os.getpid(), self.lock_path)
            return True, None

        except OSError:
            # Another process holds the OS lock
            if fh is not None:
                fh.close()
            return False, existing_pid or "unknown"

    def release(self) -> None:
        """
        Release the OS lock and remove the lock file.

        Safe to call multiple times; no-ops if the lock is not currently held.
        """
        if self._fh is None:
            return
        fh, self._fh = self._fh, None
        try:
            self._os_unlock(fh)
        except Exception as exc:
            logger.debug("PollingLock: unlock error (ignored): %s", exc)
        try:
            fh.close()
        except Exception:
            pass
        try:
            self.lock_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("PollingLock: lock-file removal error (ignored): %s", exc)
        logger.debug("Polling lock released by PID %d", os.getpid())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _os_lock(self, fh) -> None:
        """Acquire a non-blocking exclusive byte-range lock via the OS."""
        if sys.platform == "win32":
            import msvcrt
            fh.seek(0)
            # LK_NBLCK raises OSError immediately if the region is already locked
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, self._LOCK_BYTES)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _os_unlock(self, fh) -> None:
        """Release the OS byte-range lock."""
        if sys.platform == "win32":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, self._LOCK_BYTES)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _read_pid(self) -> Optional[str]:
        """Return the PID string written by the current lock owner, or None."""
        try:
            with open(self.lock_path, "r") as f:
                return f.read().strip() or None
        except OSError:
            return None


class AIPulseAgent:
    """
    Autonomous AI News Monitoring Agent.
    Coordinates collection, topic filtering, SQLite deduplication,
    free-tier Gemini AI summarization/categorization, and Telegram
    notification dispatch with fault recovery and continuous polling.
    """

    def __init__(
        self,
        collector: Optional[NewsCollector] = None,
        topic_filter: Optional[TopicFilter] = None,
        db: Optional[Database] = None,
        notifier: Optional[TelegramNotifier] = None,
        summarizer: Optional[GeminiSummarizer] = None,
        poll_interval_seconds: Optional[int] = None,
        max_notifications_per_cycle: Optional[int] = None,
        max_retries: Optional[int] = None,
        initial_run_send_alerts: Optional[bool] = None,
    ):
        self.collector = collector or NewsCollector()
        self.topic_filter = topic_filter or TopicFilter()
        self.db = db or Database()
        self.notifier = notifier or TelegramNotifier(
            bot_token=config.TELEGRAM_BOT_TOKEN,
            chat_id=config.TELEGRAM_CHAT_ID,
        )
        self.summarizer = summarizer or GeminiSummarizer()
        self.poll_interval_seconds = poll_interval_seconds or config.POLL_INTERVAL_SECONDS
        self.max_notifications_per_cycle = (
            max_notifications_per_cycle or config.MAX_NOTIFICATIONS_PER_RUN
        )
        self.max_retries = max_retries or config.MAX_RETRIES
        self.initial_run_send_alerts = (
            initial_run_send_alerts
            if initial_run_send_alerts is not None
            else config.INITIAL_RUN_SEND_ALERTS
        )

        self._stop_requested = False
        self._is_cycling = False

    def run_cycle(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute a single complete monitoring and alert cycle:
        1. Recover stale pending articles from previous interrupted runs.
        2. Ingest articles from all enabled RSS/Atom feeds.
        3. Filter articles for AI topics and score relevance.
        4. Deduplicate and persist new articles in SQLite.
        5. Dispatch unnotified articles to Telegram (rate-limited).
        """
        start_time = time.time()
        self._is_cycling = True
        logger.info("Starting AI Pulse Agent collection cycle...")

        stats = {
            "sources_queried": 0,
            "articles_fetched": 0,
            "relevant_ai_articles": 0,
            "new_articles_saved": 0,
            "duplicates_skipped": 0,
            "summaries_generated": 0,
            "summaries_cached": 0,
            "summaries_fallback": 0,
            "notifications_sent": 0,
            "notifications_failed": 0,
            "duration_seconds": 0.0,
            "first_run_seeded": False,
        }

        try:
            # 1. Reset per-cycle Gemini API limits
            self.summarizer.reset_cycle()

            # 2. Crash recovery for stale 'pending' articles
            self.db.recover_stale_pending()

            # 3. Check if this is the first run on an empty database
            is_initial_run = self.db.is_empty()

            # 4. Load sources & Ingest
            sources = config.get_sources()
            enabled_sources = [s for s in sources if s.get("enabled", True)]
            stats["sources_queried"] = len(enabled_sources)

            raw_articles = self.collector.fetch_all(enabled_sources, max_items_per_source=5)
            stats["articles_fetched"] = len(raw_articles)

            if not raw_articles:
                logger.warning("No articles fetched across enabled feeds.")
                stats["duration_seconds"] = round(time.time() - start_time, 2)
                return stats

            # 5. Topic Filter
            sources_lookup = {s["name"]: s for s in sources}
            relevant_articles = self.topic_filter.filter_articles(
                raw_articles, sources_lookup=sources_lookup
            )
            stats["relevant_ai_articles"] = len(relevant_articles)

            # 6. Persist with Safe First-Run Logic
            if is_initial_run and not self.initial_run_send_alerts:
                # Safe first-run: Seed existing articles as 'sent' baseline so phone is not spammed
                new_cnt, dup_cnt = self.db.save_articles_batch(
                    relevant_articles, default_status="sent"
                )
                stats["new_articles_saved"] = new_cnt
                stats["duplicates_skipped"] = dup_cnt
                stats["first_run_seeded"] = True
                logger.info(
                    "🛡️ Safe first-run: Recorded %d existing articles as baseline. "
                    "Only fresh updates in future cycles will alert.",
                    new_cnt,
                )
            else:
                new_cnt, dup_cnt = self.db.save_articles_batch(
                    relevant_articles, default_status="discovered"
                )
                stats["new_articles_saved"] = new_cnt
                stats["duplicates_skipped"] = dup_cnt

            if dry_run:
                # 7a. Dry-Run Processing: Enrich eligible queued articles without dispatching Telegram alerts
                logger.info("Dry-run mode: Notification dispatch skipped. Enriching eligible queued articles...")

                # Fetch articles that need summarization (up to per-cycle limit)
                eligible_articles = self.db.get_articles_needing_summary(
                    limit=self.max_notifications_per_cycle
                )

                if not eligible_articles:
                    # If all queued articles already have summaries, inspect unprocessed queue to count cached
                    unprocessed = self.db.get_unprocessed_articles(
                        limit=self.max_notifications_per_cycle,
                        max_retries=self.max_retries,
                    )
                    for article in unprocessed:
                        _, sum_type = self.enrich_article(article)
                        if sum_type == "cached":
                            stats["summaries_cached"] += 1
                    logger.info("Dry-run: All %d queued articles already have cached analyses.", len(unprocessed))
                else:
                    for article in eligible_articles:
                        if self._stop_requested:
                            break
                        _, sum_type = self.enrich_article(article)
                        if sum_type == "generated":
                            stats["summaries_generated"] += 1
                        elif sum_type == "cached":
                            stats["summaries_cached"] += 1
                        elif sum_type == "fallback":
                            stats["summaries_fallback"] += 1
                    logger.info(
                        "Dry-run: Enriched %d eligible articles (gen: %d, cache: %d, fallback: %d) and saved to SQLite.",
                        len(eligible_articles),
                        stats["summaries_generated"],
                        stats["summaries_cached"],
                        stats["summaries_fallback"],
                    )

                stats["duration_seconds"] = round(time.time() - start_time, 2)
                return stats

            # 7. Dispatch Unprocessed Articles (discovered or failed retries)
            if not self.notifier.is_configured:
                logger.warning("Telegram is not configured. Notifications skipped.")
                stats["duration_seconds"] = round(time.time() - start_time, 2)
                return stats

            unprocessed = self.db.get_unprocessed_articles(
                limit=self.max_notifications_per_cycle,
                max_retries=self.max_retries,
            )

            for idx, article in enumerate(unprocessed):
                if self._stop_requested:
                    logger.info("Shutdown requested mid-cycle. Stopping notification dispatch.")
                    break

                # Enrich article with intelligent summary, cached analysis, or fallback
                enriched_article, sum_type = self.enrich_article(article)
                if sum_type == "generated":
                    stats["summaries_generated"] += 1
                elif sum_type == "cached":
                    stats["summaries_cached"] += 1
                elif sum_type == "fallback":
                    stats["summaries_fallback"] += 1

                self.db.mark_as_pending(enriched_article.id)
                success = self.notifier.send_article(enriched_article)

                if success:
                    self.db.mark_as_sent(enriched_article.id)
                    stats["notifications_sent"] += 1
                else:
                    self.db.mark_as_failed(enriched_article.id, error_message="Telegram API dispatch failed")
                    stats["notifications_failed"] += 1

                # Rate-limit delay between messages
                if idx < len(unprocessed) - 1:
                    time.sleep(1.0)

        finally:
            self._is_cycling = False
            stats["duration_seconds"] = round(time.time() - start_time, 2)
            logger.info(
                "Cycle finished in %ss: %d new saved, %d summaries (gen: %d, cache: %d, fallback: %d), %d sent, %d failed.",
                stats["duration_seconds"],
                stats["new_articles_saved"],
                stats["summaries_generated"] + stats["summaries_cached"] + stats["summaries_fallback"],
                stats["summaries_generated"],
                stats["summaries_cached"],
                stats["summaries_fallback"],
                stats["notifications_sent"],
                stats["notifications_failed"],
            )

        return stats

    def enrich_article(self, article: Article) -> Tuple[Article, str]:
        """
        Enrich an article with LLM analysis or rule-based fallback,
        reusing cached results when available.

        Returns:
            Tuple of (enriched_article, summary_type) where summary_type is 'cached', 'generated', or 'fallback'.
        """
        # 1. Check SQLite cache (never re-summarize already analyzed article)
        cached = self.db.get_cached_analysis(article.id)
        if cached and cached.get("ai_summary"):
            article.ai_summary = cached["ai_summary"]
            article.ai_category = cached.get("ai_category")
            article.ai_importance = cached.get("ai_importance")
            article.ai_relevance = cached.get("ai_relevance")
            article.why_it_matters = cached.get("why_it_matters")
            article.suggested_action = cached.get("suggested_action")
            article.score_explanation = cached.get("score_explanation")
            article.summarization_status = cached.get("summarization_status", "completed")
            article.llm_model = cached.get("llm_model")
            article.llm_processed_at = cached.get("llm_processed_at")
            logger.debug("Reusing cached LLM analysis for '%s'", article.title[:40])
            return article, "cached"

        # 2. Check if Gemini request is permitted under free-tier caps
        allowed, reason = self.summarizer.can_make_request()
        if allowed:
            analysis, status_msg = self.summarizer.summarize_article(article)
            if analysis is not None:
                status = "insufficient" if analysis.is_insufficient_info else "completed"
                self.db.save_analysis(
                    article_id=article.id,
                    analysis=analysis,
                    model=self.summarizer.model_name,
                    status=status,
                )
                article.ai_summary = analysis.summary
                article.ai_category = analysis.category
                article.ai_importance = analysis.importance_score
                article.ai_relevance = analysis.relevance_score
                article.why_it_matters = analysis.why_it_matters
                article.suggested_action = analysis.suggested_action
                article.score_explanation = analysis.score_explanation
                article.summarization_status = status
                article.llm_model = self.summarizer.model_name
                return article, "generated"

            logger.info("Gemini call skipped/failed (%s). Using rule-based fallback.", status_msg)

        # 3. Deterministic rule-based fallback (free, offline, never halts pipeline)
        fallback = self.summarizer.create_fallback_analysis(article)
        self.db.save_fallback_analysis(
            article_id=article.id,
            summary=fallback.summary,
            category=fallback.category,
            importance=fallback.importance_score,
            relevance=fallback.relevance_score,
            why_it_matters=fallback.why_it_matters,
            suggested_action=fallback.suggested_action,
            explanation=fallback.score_explanation,
        )
        article.ai_summary = fallback.summary
        article.ai_category = fallback.category
        article.ai_importance = fallback.importance_score
        article.ai_relevance = fallback.relevance_score
        article.why_it_matters = fallback.why_it_matters
        article.suggested_action = fallback.suggested_action
        article.score_explanation = fallback.score_explanation
        article.summarization_status = "fallback"
        article.llm_model = "rule-based-fallback"
        return article, "fallback"

    def start_polling(
        self,
        poll_interval: Optional[int] = None,
        max_iterations: Optional[int] = None,
        lock_path: Optional[Path] = None,
    ):
        """
        Run continuous automated polling loop with clean interrupt handling.

        Only one polling process may run at a time.  An OS-level file lock is
        acquired before the loop starts and automatically released – even on
        crashes or SIGKILL – so stale locks are impossible.

        Args:
            poll_interval:  Seconds to wait between cycles (defaults to config).
            max_iterations: Optional limit on cycles (useful for tests / bounded runs).
            lock_path:      Override the default lock-file path (used in tests).
        """
        interval = poll_interval or self.poll_interval_seconds
        iterations = 0

        # ── Determine lock-file path ──────────────────────────────────────────
        if lock_path is None:
            if hasattr(self.db, "db_path") and self.db.db_path != ":memory:":
                lock_path = Path(str(self.db.db_path)).parent / "agent.lock"
            elif hasattr(self.db, "db_path") and self.db.db_path == ":memory:":
                import tempfile
                lock_path = Path(tempfile.gettempdir()) / f"agent_mem_{id(self)}.lock"
            else:
                lock_path = Path("data") / "agent.lock"

        # ── Acquire the OS-level single-instance lock ─────────────────────────
        lock = PollingLock(lock_path)
        acquired, owner_pid = lock.acquire()

        if not acquired:
            print(
                f"\n❌  AI Pulse Agent is already running (lock owner PID: {owner_pid}).\n"
                "    Only one polling process may run at a time.\n"
                "    Stop the existing process with Ctrl+C and retry.\n"
            )
            return

        # Setup graceful signal handlers
        def _handle_exit(signum, frame):
            logger.info("Shutdown signal received (%s). Stopping agent gracefully...", signum)
            self._stop_requested = True

        signal.signal(signal.SIGINT, _handle_exit)
        signal.signal(signal.SIGTERM, _handle_exit)

        print("\n" + "=" * 65)
        print("🤖 [AI PULSE AGENT] Starting Autonomous Polling Service")
        print(f"⏱️  Interval: {interval} seconds ({interval / 60:.1f} minutes)")
        print(f"📬 Max notifications per cycle: {self.max_notifications_per_cycle}")
        print(f"💾 Database: {self.db.db_path}")
        print(f"🔒 Lock: {lock_path} (PID: {os.getpid()})")
        print("💡 Press Ctrl+C at any time to stop cleanly.")
        print("=" * 65 + "\n")

        try:
            while not self._stop_requested:
                iterations += 1
                print(f"[{time.strftime('%X')}] 🔄 Starting cycle #{iterations}...")

                try:
                    stats = self.run_cycle(dry_run=False)
                    print(
                        f"[{time.strftime('%X')}] ✅ Cycle #{iterations} complete: "
                        f"Fetched: {stats['articles_fetched']} | "
                        f"New: {stats['new_articles_saved']} | "
                        f"Sent: {stats['notifications_sent']} | "
                        f"Duration: {stats['duration_seconds']}s"
                    )
                    if stats["first_run_seeded"]:
                        print("   🛡️ (Safe first-run: baseline established without alerting)")
                except Exception as e:
                    logger.error("Error in cycle #%d: %s", iterations, e, exc_info=True)
                    print(f"[{time.strftime('%X')}] ❌ Cycle #{iterations} encountered an error: {e}")

                if max_iterations and iterations >= max_iterations:
                    print(f"🏁 Reached maximum requested iterations ({max_iterations}). Polling complete.")
                    break

                if self._stop_requested:
                    break

                # Sleep in 1-second chunks for instant Ctrl+C response
                print(f"💤 Sleeping for {interval}s until next cycle... (Ctrl+C to stop)\n")
                sleep_remaining = interval
                while sleep_remaining > 0 and not self._stop_requested:
                    time.sleep(min(1.0, sleep_remaining))
                    sleep_remaining -= 1.0

        finally:
            lock.release()
            print("\n🛑 AI Pulse Agent stopped safely. Goodbye!")

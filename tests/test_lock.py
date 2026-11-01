"""
Tests for PollingLock (OS-level single-instance guard) and the updated
get_stats() pending-summary breakdown.

All tests use temporary directories or in-memory databases; the live
data/ai_pulse.db is never touched.  No Telegram messages are sent and
no Gemini quota is consumed.
"""

import multiprocessing
import os
import sys
import time
import pytest
from pathlib import Path

from src.agent import PollingLock
from src.database import Database
from src.models import Article


# ---------------------------------------------------------------------------
# Subprocess helpers (must be module-level for multiprocessing 'spawn' context)
# ---------------------------------------------------------------------------

def _proc_acquire_and_hold(lock_path_str: str, ready_ev, release_ev, result_q):
    """
    Worker: acquire the lock, signal ready, wait for the release event, then
    release cleanly.  Communicates success/failure via result_q.
    """
    lock = PollingLock(lock_path_str)
    acquired, _ = lock.acquire()
    result_q.put(acquired)
    if acquired:
        ready_ev.set()
        release_ev.wait(timeout=15)
        lock.release()
    else:
        ready_ev.set()   # unblock parent even on failure


def _proc_acquire_and_crash(lock_path_str: str, ready_ev):
    """
    Worker: acquire the lock then crash immediately with os._exit(1) —
    intentionally skips lock.release() to simulate a hard crash.
    """
    lock = PollingLock(lock_path_str)
    acquired, _ = lock.acquire()
    if acquired:
        ready_ev.set()
        os._exit(1)   # hard crash, no atexit / finally
    else:
        ready_ev.set()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lock_path(tmp_path) -> Path:
    return tmp_path / "agent.lock"


@pytest.fixture
def spawn_ctx():
    """Multiprocessing context using 'spawn' so tests work on Windows and POSIX."""
    return multiprocessing.get_context("spawn")


# ---------------------------------------------------------------------------
# Single-process lock lifecycle tests
# ---------------------------------------------------------------------------

class TestPollingLockBasic:

    def test_acquire_and_release_succeed(self, lock_path):
        """A single process can acquire and then release the lock."""
        lock = PollingLock(lock_path)
        acquired, owner = lock.acquire()
        assert acquired is True
        assert owner is None
        assert lock.is_held
        lock.release()
        assert not lock.is_held

    def test_lock_file_exists_while_held(self, lock_path):
        """The lock file must exist while the lock is held."""
        lock = PollingLock(lock_path)
        lock.acquire()
        assert lock_path.exists(), "Lock file should exist while lock is held"
        lock.release()

    def test_lock_file_removed_after_release(self, lock_path):
        """The lock file is deleted when the lock is released."""
        lock = PollingLock(lock_path)
        lock.acquire()
        assert lock_path.exists()
        lock.release()
        assert not lock_path.exists()

    def test_double_release_is_safe(self, lock_path):
        """Calling release() twice must not raise."""
        lock = PollingLock(lock_path)
        lock.acquire()
        lock.release()
        lock.release()   # second call is a no-op

    def test_release_without_acquire_is_safe(self, lock_path):
        """Calling release() without a prior acquire must not raise."""
        lock = PollingLock(lock_path)
        lock.release()  # should be a no-op

    def test_lock_released_via_try_finally(self, lock_path):
        """Lock is released correctly even when an exception occurs in the protected block."""
        lock = PollingLock(lock_path)
        lock.acquire()
        try:
            raise RuntimeError("Simulated cycle error")
        except RuntimeError:
            pass
        finally:
            lock.release()

        assert not lock.is_held
        assert not lock_path.exists()

        # A subsequent acquisition must succeed
        lock2 = PollingLock(lock_path)
        acquired, _ = lock2.acquire()
        assert acquired is True
        lock2.release()

    def test_reacquire_after_release(self, lock_path):
        """The same path can be re-acquired after a clean release."""
        lock = PollingLock(lock_path)
        lock.acquire()
        lock.release()

        lock2 = PollingLock(lock_path)
        acquired, _ = lock2.acquire()
        assert acquired is True
        lock2.release()


# ---------------------------------------------------------------------------
# Cross-process lock tests
# ---------------------------------------------------------------------------

class TestPollingLockCrossProcess:

    def test_second_process_is_rejected(self, lock_path, spawn_ctx):
        """
        While one process holds the OS lock, a second acquisition attempt
        from a different process must fail non-blocking.
        """
        ready = spawn_ctx.Event()
        release = spawn_ctx.Event()
        result_q = spawn_ctx.Queue()

        p = spawn_ctx.Process(
            target=_proc_acquire_and_hold,
            args=(str(lock_path), ready, release, result_q),
        )
        p.start()
        assert ready.wait(timeout=15), "Subprocess did not signal ready in time"
        assert result_q.get() is True, "Subprocess failed to acquire lock"

        # Main process tries to acquire while subprocess holds it
        lock2 = PollingLock(lock_path)
        acquired, owner = lock2.acquire()
        assert acquired is False, "Second acquisition should have been rejected"
        assert not lock2.is_held
        # owner should be the subprocess PID (best-effort diagnostic)
        assert owner is not None

        # Clean up subprocess
        release.set()
        p.join(timeout=15)
        assert p.exitcode == 0

    def test_lock_is_freed_after_clean_shutdown(self, lock_path, spawn_ctx):
        """
        After the lock-holding subprocess exits cleanly, the lock must be
        acquirable by a new process.
        """
        ready = spawn_ctx.Event()
        release = spawn_ctx.Event()
        result_q = spawn_ctx.Queue()

        p = spawn_ctx.Process(
            target=_proc_acquire_and_hold,
            args=(str(lock_path), ready, release, result_q),
        )
        p.start()
        ready.wait(timeout=15)
        result_q.get()   # consume result

        release.set()
        p.join(timeout=15)
        assert p.exitcode == 0

        # Now the lock must be available
        lock = PollingLock(lock_path)
        acquired, _ = lock.acquire()
        assert acquired is True, "Lock was not released after clean subprocess shutdown"
        lock.release()

    def test_stale_lock_auto_released_after_crash(self, lock_path, spawn_ctx):
        """
        When a process crashes (os._exit) without calling release(), the OS
        automatically frees the byte-range lock, so a subsequent acquisition
        from any process must succeed.
        """
        ready = spawn_ctx.Event()
        p = spawn_ctx.Process(
            target=_proc_acquire_and_crash,
            args=(str(lock_path), ready),
        )
        p.start()
        assert ready.wait(timeout=15), "Crash subprocess did not signal ready"
        p.join(timeout=15)
        # Process crashed — nonzero exit code expected
        assert p.exitcode != 0

        # Small delay to let the OS flush the lock tables
        time.sleep(0.05)

        lock = PollingLock(lock_path)
        acquired, _ = lock.acquire()
        assert acquired is True, (
            "Stale OS lock was not automatically released after subprocess crash"
        )
        lock.release()

    def test_only_one_of_two_concurrent_acquisitions_succeeds(self, lock_path, spawn_ctx):
        """
        When two processes race to acquire the same lock, exactly one succeeds
        and the other is rejected.
        """
        ready1 = spawn_ctx.Event()
        ready2 = spawn_ctx.Event()
        release1 = spawn_ctx.Event()
        release2 = spawn_ctx.Event()
        q1 = spawn_ctx.Queue()
        q2 = spawn_ctx.Queue()

        p1 = spawn_ctx.Process(
            target=_proc_acquire_and_hold,
            args=(str(lock_path), ready1, release1, q1),
        )
        p2 = spawn_ctx.Process(
            target=_proc_acquire_and_hold,
            args=(str(lock_path), ready2, release2, q2),
        )
        p1.start()
        p2.start()

        ready1.wait(timeout=15)
        ready2.wait(timeout=15)

        r1 = q1.get()
        r2 = q2.get()

        # Exactly one must have succeeded
        assert r1 != r2, "Both processes acquired the lock — mutual exclusion violated"
        assert r1 is True or r2 is True, "Neither process acquired the lock"

        release1.set()
        release2.set()
        p1.join(timeout=15)
        p2.join(timeout=15)


# ---------------------------------------------------------------------------
# start_polling() integration — lock prevents duplicate polling loops
# ---------------------------------------------------------------------------

class TestPollingLockInStartPolling:

    def test_start_polling_acquires_lock(self, tmp_path, capsys):
        """
        start_polling() with max_iterations=1 must write the PID file and
        release it cleanly on exit (no leftover lock file).
        """
        from unittest.mock import MagicMock, patch
        from src.agent import AIPulseAgent

        db = Database(":memory:")
        mock_collector = MagicMock()
        mock_collector.fetch_all.return_value = []
        mock_notifier = MagicMock()
        mock_summarizer = MagicMock()
        mock_summarizer.reset_cycle = MagicMock()
        mock_summarizer.can_make_request.return_value = (False, "disabled")

        agent = AIPulseAgent(
            collector=mock_collector,
            db=db,
            notifier=mock_notifier,
            summarizer=mock_summarizer,
        )

        lock_path = tmp_path / "agent.lock"
        with patch("time.sleep"):
            agent.start_polling(poll_interval=0, max_iterations=1, lock_path=lock_path)

        # Lock file must have been removed after clean exit
        assert not lock_path.exists(), "Lock file was not cleaned up after start_polling"

    def test_start_polling_rejects_when_lock_held(self, tmp_path, capsys, spawn_ctx):
        """
        If the OS lock is already held, start_polling() must print a rejection
        message and return immediately without launching a cycle.
        """
        from unittest.mock import MagicMock, patch
        from src.agent import AIPulseAgent

        ready = spawn_ctx.Event()
        release = spawn_ctx.Event()
        result_q = spawn_ctx.Queue()

        lock_path = tmp_path / "agent.lock"

        p = spawn_ctx.Process(
            target=_proc_acquire_and_hold,
            args=(str(lock_path), ready, release, result_q),
        )
        p.start()
        ready.wait(timeout=15)
        result_q.get()

        # Now try to start polling — should be rejected
        db = Database(":memory:")
        mock_collector = MagicMock()
        mock_notifier = MagicMock()
        mock_summarizer = MagicMock()
        mock_summarizer.reset_cycle = MagicMock()

        agent = AIPulseAgent(
            collector=mock_collector,
            db=db,
            notifier=mock_notifier,
            summarizer=mock_summarizer,
        )

        with patch("time.sleep"):
            agent.start_polling(poll_interval=0, max_iterations=5, lock_path=lock_path)

        captured = capsys.readouterr().out
        assert "already running" in captured.lower() or "❌" in captured, (
            "start_polling should print a rejection message when lock is held"
        )
        # Cycle must NOT have been executed
        assert mock_collector.fetch_all.call_count == 0, (
            "fetch_all must not be called when lock is rejected"
        )

        release.set()
        p.join(timeout=15)


# ---------------------------------------------------------------------------
# get_stats() — pending-summary three-way split
# ---------------------------------------------------------------------------

class TestStatsPendingBreakdown:

    @pytest.fixture
    def populated_db(self):
        """
        In-memory DB with articles covering every relevant
        (status × summarization_status) combination so we can verify the
        three-way pending split precisely.

        Layout:
          • act-new-1, act-new-2  → discovered + pending   (actionable = 2)
          • hist-1, hist-2, hist-3 → sent    + pending   (historic  = 3)
          • ign-1, ign-2, ign-3, ign-4 → ignored + pending (ignored   = 4)

        Sequence:
          1. Save the 4 future-ignored articles first.
          2. Call ignore_existing_backlog() — converts those 4 to 'ignored'.
          3. Save the 3 historic articles and mark them sent.
          4. Save the 2 actionable articles — they remain 'discovered'.
        """
        db = Database(":memory:")

        def _art(art_id, link_suffix):
            return Article(
                id=art_id,
                title=f"Article {art_id}",
                link=f"https://example.com/{link_suffix}",
                source="TestSrc",
            )

        # Step 1 – articles that will become 'ignored'
        for n in range(1, 5):
            db.save_article(_art(f"ign-{n}", f"i{n}"))

        # Step 2 – ignore the backlog (only the 4 above are in the DB)
        ignored = db.ignore_existing_backlog()
        assert ignored == 4, f"Expected 4 ignored, got {ignored}"

        # Step 3 – historic (sent, pre-Phase-4, summarization_status stays 'pending')
        for n in range(1, 4):
            db.save_article(_art(f"hist-{n}", f"h{n}"))
            db.mark_as_sent(f"hist-{n}")

        # Step 4 – actionable discovered articles
        db.save_article(_art("act-new-1", "an1"))
        db.save_article(_art("act-new-2", "an2"))

        return db

    def test_pending_three_way_split(self, populated_db):
        """summaries_pending == actionable + historic + ignored_pending (no leakage)."""
        stats = populated_db.get_stats()

        assert stats["summaries_pending_actionable"] == 2
        assert stats["summaries_pending_historic"] == 3
        assert stats["summaries_pending_ignored"] == 4

        total_pending = (
            stats["summaries_pending_actionable"]
            + stats["summaries_pending_historic"]
            + stats["summaries_pending_ignored"]
        )
        assert total_pending == stats["summaries_pending"], (
            "actionable + historic + ignored must equal the raw summaries_pending total"
        )

    def test_actionable_count_is_zero_when_no_discovered_articles(self):
        """If all articles are sent or ignored, actionable must be 0."""
        db = Database(":memory:")
        art = Article(id="x1", title="T", link="https://ex.com/x1", source="S")
        db.save_article(art)
        db.mark_as_sent("x1")
        stats = db.get_stats()
        assert stats["summaries_pending_actionable"] == 0
        assert stats["summaries_pending_historic"] == 1
        assert stats["summaries_pending_ignored"] == 0

    def test_ignored_pending_zero_when_no_ignored_articles(self):
        """If no articles are ignored, summaries_pending_ignored must be 0."""
        db = Database(":memory:")
        art = Article(id="y1", title="T", link="https://ex.com/y1", source="S")
        db.save_article(art)
        stats = db.get_stats()
        assert stats["summaries_pending_ignored"] == 0
        assert stats["summaries_pending_actionable"] == 1

    def test_completed_summaries_not_counted_in_pending(self):
        """Articles with completed summaries must not appear in any pending bucket."""
        from src.models import ArticleAnalysis
        db = Database(":memory:")
        art = Article(id="z1", title="T", link="https://ex.com/z1", source="S")
        db.save_article(art)
        analysis = ArticleAnalysis(
            summary="Done",
            main_development="Done",
            why_it_matters="Done",
            category="Other AI news",
        )
        db.save_analysis("z1", analysis, model="test", status="completed")
        stats = db.get_stats()
        assert stats["summaries_pending_actionable"] == 0
        assert stats["summaries_pending_historic"] == 0
        assert stats["summaries_pending_ignored"] == 0
        assert stats["summaries_completed"] == 1

    def test_print_status_shows_all_three_pending_buckets(self, capsys):
        """print_status() must display actionable, historic, and ignored buckets."""
        from app import print_status
        db = Database(":memory:")

        # Step 1: one article that will become 'ignored'
        art_ign = Article(id="i1", title="I", link="https://ex.com/i1", source="S")
        db.save_article(art_ign)
        db.ignore_existing_backlog()   # only i1 goes to 'ignored'

        # Step 2: one historic (sent, summarization_status stays 'pending')
        art_sent = Article(id="s1", title="S", link="https://ex.com/s1", source="S")
        db.save_article(art_sent)
        db.mark_as_sent("s1")

        # Step 3: one actionable discovered article
        art_new = Article(id="new1", title="N", link="https://ex.com/new1", source="S")
        db.save_article(art_new)

        print_status(db)
        out = capsys.readouterr().out

        assert "Actionable" in out, "print_status must show Actionable pending bucket"
        assert "Historic" in out,   "print_status must show Historic pending bucket"
        assert "Ignored" in out,    "print_status must show Ignored pending bucket"

    def test_print_status_no_misleading_total_pending(self, capsys):
        """
        The status display must not imply that ignored or historic articles will
        be processed.  Check that the output contains clarifying labels.
        """
        from app import print_status
        db = Database(":memory:")
        print_status(db)
        out = capsys.readouterr().out

        # The old undifferentiated 'Pending – Actionable' alone should NOT be the
        # only line; it must be accompanied by labels for historic and ignored.
        assert "Historic" in out or "historic" in out, (
            "print_status must clearly label historic pending articles"
        )
        assert "Ignored" in out or "ignored" in out, (
            "print_status must clearly label ignored pending articles"
        )


# ---------------------------------------------------------------------------
# Live DB safety check (read-only)
# ---------------------------------------------------------------------------

class TestLiveDbUnchanged:

    def test_live_db_sent_count_unchanged(self):
        """
        Sanity check: the live database preserves all 60 ignored articles and
        all 33 historic sent articles. This test is read-only and fails
        loudly if historical records were accidentally modified or reset.
        """
        import sqlite3
        from pathlib import Path

        db_path = Path(__file__).resolve().parent.parent / "data" / "ai_pulse.db"
        if not db_path.exists():
            pytest.skip("Live database not present in this environment")

        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(1) FROM articles WHERE status = 'ignored'")
        ignored = cur.fetchone()[0]
        cur.execute("SELECT COUNT(1) FROM articles WHERE status = 'sent' AND summarization_status = 'pending'")
        historic_sent = cur.fetchone()[0]
        cur.execute("SELECT COUNT(1) FROM articles WHERE status = 'sent'")
        sent = cur.fetchone()[0]
        conn.close()

        assert ignored == 60, f"Expected 60 ignored articles, found {ignored}"
        assert historic_sent == 33, f"Expected 33 historic sent articles, found {historic_sent}"
        assert sent >= 38, f"Expected at least 38 sent articles, found {sent}"

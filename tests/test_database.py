import pytest
from src.database import Database
from src.models import Article


@pytest.fixture
def test_db():
    """In-memory SQLite database for isolated unit testing."""
    return Database(":memory:")


def test_database_init(test_db):
    assert test_db.is_empty()
    stats = test_db.get_stats()
    assert stats["total"] == 0
    assert stats["discovered"] == 0
    assert stats["sent"] == 0


def test_save_and_retrieve_article(test_db):
    art = Article(
        id="art-101",
        title="OpenAI Releases GPT-5 Weights",
        link="https://openai.com/news/gpt-5?utm_source=twitter",
        source="OpenAI News",
        published="2026-10-18T12:00:00Z",
        summary="A major leap forward in AI reasoning.",
        category="releases",
        relevance_score=0.95,
        matched_keywords=["gpt", "reasoning"],
    )

    is_new, art_id = test_db.save_article(art)
    assert is_new is True
    assert art_id == "art-101"

    retrieved = test_db.get_article("art-101")
    assert retrieved is not None
    assert retrieved.title == "OpenAI Releases GPT-5 Weights"
    assert retrieved.link == "https://openai.com/news/gpt-5"  # Normalized
    assert retrieved.relevance_score == 0.95
    assert "gpt" in retrieved.matched_keywords
    assert retrieved.status == "discovered"


def test_duplicate_detection_same_id(test_db):
    art = Article(
        id="art-dup-1",
        title="First Title",
        link="https://example.com/art1",
        source="Source A",
    )
    is_new1, _ = test_db.save_article(art)
    assert is_new1 is True

    # Attempt to insert same ID
    is_new2, _ = test_db.save_article(art)
    assert is_new2 is False
    assert test_db.count_articles() == 1


def test_duplicate_detection_across_feeds(test_db):
    # Same article syndicated across two different sources with different tracking links
    art_source1 = Article(
        id="guid-source-1",
        title="DeepMind Gemini Robot Paper",
        link="https://deepmind.google/blog/robotics?utm_source=rss",
        source="DeepMind Feed",
    )
    is_new1, _ = test_db.save_article(art_source1)
    assert is_new1 is True

    # Same normalized URL from a secondary aggregator with different ID
    art_source2 = Article(
        id="guid-source-2",
        title="DeepMind Gemini Robot Paper (Repost)",
        link="https://deepmind.google/blog/robotics?utm_medium=email",
        source="TechCrunch Aggregator",
    )
    is_new2, _ = test_db.save_article(art_source2)
    assert is_new2 is False
    assert test_db.count_articles() == 1


def test_notification_states(test_db):
    art = Article(
        id="art-state-1",
        title="Testing State Transitions",
        link="https://example.com/state-test",
        source="Test Source",
    )
    test_db.save_article(art)

    # Transition: discovered -> pending
    test_db.mark_as_pending("art-state-1")
    assert test_db.get_article("art-state-1").status == "pending"

    # Transition: pending -> failed
    test_db.mark_as_failed("art-state-1", error_message="Telegram 503 Service Unavailable")
    failed_art = test_db.get_article("art-state-1")
    assert failed_art.status == "failed"
    assert failed_art.retry_count == 1
    assert "503" in failed_art.error_message

    # Transition: failed -> sent
    test_db.mark_as_sent("art-state-1")
    sent_art = test_db.get_article("art-state-1")
    assert sent_art.status == "sent"
    assert sent_art.error_message is None


def test_recovery_of_stale_pending(test_db):
    art1 = Article(id="art-p1", title="Pending 1", link="https://example.com/p1", source="Src")
    art2 = Article(id="art-p2", title="Pending 2", link="https://example.com/p2", source="Src")
    test_db.save_article(art1)
    test_db.save_article(art2)

    test_db.mark_as_pending("art-p1")
    test_db.mark_as_pending("art-p2")
    assert test_db.count_articles(status="pending") == 2

    # Simulate recovery after ungraceful crash
    recovered = test_db.recover_stale_pending()
    assert recovered == 2
    assert test_db.count_articles(status="pending") == 0
    assert test_db.count_articles(status="discovered") == 2


def test_unprocessed_articles_retry_limit(test_db):
    art = Article(id="art-retry", title="Retry Article", link="https://example.com/retry", source="Src")
    test_db.save_article(art)

    # Fail 3 times
    for _ in range(3):
        test_db.mark_as_failed("art-retry", "Network Error")

    # If max_retries is 3, an article with retry_count == 3 should NOT be returned
    unprocessed = test_db.get_unprocessed_articles(limit=10, max_retries=3)
    assert len(unprocessed) == 0

    # If max_retries is 5, it should still be retried
    unprocessed_higher = test_db.get_unprocessed_articles(limit=10, max_retries=5)
    assert len(unprocessed_higher) == 1
    assert unprocessed_higher[0].id == "art-retry"


def test_get_stats_accounting_all_summarization_states(test_db):
    """Verify get_stats accounts for all summarization states including insufficient."""
    from src.models import ArticleAnalysis
    from app import print_status

    # 1. Base article with 'pending' summarization
    art_pending = Article(id="s-pending", title="Pending Article", link="https://ex.com/p", source="Src")
    test_db.save_article(art_pending)

    # 2. Article with 'completed' summarization
    art_comp = Article(id="s-comp", title="Completed Article", link="https://ex.com/c", source="Src")
    test_db.save_article(art_comp)
    analysis_comp = ArticleAnalysis(
        summary="Completed summary",
        main_development="Dev",
        why_it_matters="Why",
        category="Other AI news",
        importance_score=7,
        relevance_score=7,
        is_insufficient_info=False,
    )
    test_db.save_analysis("s-comp", analysis_comp, model="gemini-3.8-flash", status="completed")

    # 3. Article with 'insufficient' context
    art_insuff = Article(id="s-insuff", title="Insufficient Article", link="https://ex.com/i", source="Src")
    test_db.save_article(art_insuff)
    analysis_insuff = ArticleAnalysis(
        summary="Brief headline",
        main_development="Brief",
        why_it_matters="Brief",
        category="Other AI news",
        importance_score=4,
        relevance_score=4,
        is_insufficient_info=True,
    )
    test_db.save_analysis("s-insuff", analysis_insuff, model="gemini-3.8-flash", status="insufficient")

    # 4. Article with 'fallback' summarization
    art_fallback = Article(id="s-fall", title="Fallback Article", link="https://ex.com/f", source="Src")
    test_db.save_article(art_fallback)
    test_db.save_fallback_analysis("s-fall", summary="Fallback sum", category="Other AI news")

    # 5. Article with 'failed' summarization
    art_failed = Article(id="s-fail", title="Failed Article", link="https://ex.com/fail", source="Src")
    test_db.save_article(art_failed)
    test_db.mark_summarization_failed("s-fail", error_message="LLM API Timeout")

    stats = test_db.get_stats()

    assert stats["total"] == 5
    assert stats["summaries_completed"] == 1
    assert stats["summaries_fallback"] == 1
    assert stats["summaries_insufficient"] == 1
    assert stats["summaries_failed"] == 1
    assert stats["summaries_pending"] == 1

    # Exact equality: all articles are accounted for across all states
    total_summaries = (
        stats["summaries_completed"]
        + stats["summaries_fallback"]
        + stats["summaries_insufficient"]
        + stats["summaries_failed"]
        + stats["summaries_pending"]
    )
    assert total_summaries == stats["total"]


def test_print_status_displays_insufficient_context(test_db, capsys):
    """Verify that print_status in app.py renders 'Insufficient Context' correctly."""
    from src.models import ArticleAnalysis
    from app import print_status

    art = Article(id="art-ins", title="Short Item", link="https://ex.com/short", source="Src")
    test_db.save_article(art)
    analysis = ArticleAnalysis(
        summary="Short",
        main_development="Short",
        why_it_matters="Short",
        category="Other AI news",
        is_insufficient_info=True,
    )
    test_db.save_analysis("art-ins", analysis, model="gemini-3.8-flash", status="insufficient")

    print_status(test_db)
    captured = capsys.readouterr().out

    assert "Insufficient Context:     1" in captured
    assert "Completed (AI Summaries): 0" in captured
    assert "Fallback (Rule-Based):    0" in captured
    # Verify new fields are present
    assert "Actionable" in captured
    assert "Historic" in captured
    assert "Ignored" in captured


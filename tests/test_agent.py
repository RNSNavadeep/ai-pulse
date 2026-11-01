import pytest
from unittest.mock import MagicMock, patch
from src.agent import AIPulseAgent
from src.database import Database
from src.models import Article


def test_agent_safe_first_run(memory_db, mock_summarizer):
    """Verify that on an initial empty database, historical articles are recorded without alert spam."""
    mock_collector = MagicMock()
    mock_collector.fetch_all.return_value = [
        Article(
            id=f"hist-{i}",
            title=f"New LLM and AI Agent Breakthrough {i}",
            link=f"https://example.com/h{i}",
            source="OpenAI News",
        )
        for i in range(5)
    ]

    mock_notifier = MagicMock()
    mock_notifier.is_configured = True
    mock_notifier.send_article.return_value = True

    agent = AIPulseAgent(
        collector=mock_collector,
        db=memory_db,
        notifier=mock_notifier,
        summarizer=mock_summarizer,
        initial_run_send_alerts=False,  # Safe first run
    )

    stats = agent.run_cycle(dry_run=False)

    assert stats["first_run_seeded"] is True
    assert stats["new_articles_saved"] == 5
    assert stats["notifications_sent"] == 0
    # Verified: Telegram was NOT spammed on first run
    assert mock_notifier.send_article.call_count == 0
    # All 5 articles were recorded as baseline 'sent'
    assert memory_db.count_articles(status="sent") == 5
    # Summarizer should not be called during safe baseline seeding
    assert mock_summarizer.summarize_article.call_count == 0


def test_agent_subsequent_cycle_alerts(memory_db, mock_summarizer):
    """Verify that new articles discovered on subsequent cycles trigger notifications."""
    mock_notifier = MagicMock()
    mock_notifier.is_configured = True
    mock_notifier.send_article.return_value = True

    mock_collector = MagicMock()

    agent = AIPulseAgent(
        collector=mock_collector,
        db=memory_db,
        notifier=mock_notifier,
        summarizer=mock_summarizer,
        initial_run_send_alerts=False,
    )

    # 1. Establish baseline
    mock_collector.fetch_all.return_value = [
        Article(
            id="old-1",
            title="Old LLM Model Release",
            link="https://example.com/old1",
            source="OpenAI News",
        )
    ]
    agent.run_cycle()
    assert mock_notifier.send_article.call_count == 0

    # 2. Next cycle discovers a brand new breaking model release
    mock_collector.fetch_all.return_value = [
        Article(
            id="old-1",
            title="Old LLM Model Release",
            link="https://example.com/old1",
            source="OpenAI News",
        ),
        Article(
            id="new-2",
            title="Breaking: New AI Agent System",
            link="https://example.com/new2",
            source="OpenAI News",
        ),
    ]

    with patch("time.sleep"):
        stats2 = agent.run_cycle()

    assert stats2["new_articles_saved"] == 1
    assert stats2["duplicates_skipped"] == 1
    assert stats2["notifications_sent"] == 1
    assert mock_notifier.send_article.call_count == 1
    assert mock_summarizer.summarize_article.call_count == 1

    # Article in database should now be 'sent' and have AI summary populated
    new_art = memory_db.get_article("new-2")
    assert new_art.status == "sent"
    assert "Breaking: New AI Agent System" in new_art.ai_summary


def test_agent_deduplication_prevents_duplicate_alerts(memory_db, mock_summarizer):
    """Verify that repeated cycles with the same articles do not resend alerts."""
    mock_notifier = MagicMock()
    mock_notifier.is_configured = True
    mock_notifier.send_article.return_value = True

    mock_collector = MagicMock()
    articles = [
        Article(id="art-1", title="New LLM 1", link="https://example.com/1", source="OpenAI News"),
        Article(id="art-2", title="New AI Agent 2", link="https://example.com/2", source="OpenAI News"),
    ]
    mock_collector.fetch_all.return_value = articles

    agent = AIPulseAgent(
        collector=mock_collector,
        db=memory_db,
        notifier=mock_notifier,
        summarizer=mock_summarizer,
        initial_run_send_alerts=True,  # Dispatches alerts for these 2
    )

    with patch("time.sleep"):
        stats1 = agent.run_cycle()
    assert stats1["notifications_sent"] == 2
    assert mock_notifier.send_article.call_count == 2
    assert mock_summarizer.summarize_article.call_count == 2

    # Second cycle with exact same articles
    with patch("time.sleep"):
        stats2 = agent.run_cycle()
    assert stats2["new_articles_saved"] == 0
    assert stats2["duplicates_skipped"] == 2
    assert stats2["notifications_sent"] == 0
    # Still only 2 calls from cycle 1 - no duplicate alerts or summaries
    assert mock_notifier.send_article.call_count == 2
    assert mock_summarizer.summarize_article.call_count == 2


def test_agent_deduplication_cross_feed_no_resummarize(memory_db, mock_summarizer):
    """Verify that articles syndicated across different feeds with the same canonical URL are deduplicated."""
    mock_notifier = MagicMock()
    mock_notifier.is_configured = True
    mock_notifier.send_article.return_value = True

    # Same article URL with different UTM tracking tags and different IDs from two feeds
    art_feed1 = Article(
        id="feed1-guid",
        title="Google DeepMind Publishes New Gemini LLM Reasoning Paper",
        link="https://deepmind.google/blog/frontier-paper?utm_source=rss",
        source="Google DeepMind",
    )
    art_feed2 = Article(
        id="feed2-guid",
        title="Google DeepMind Publishes New Gemini LLM Reasoning Paper (Syndicated)",
        link="https://deepmind.google/blog/frontier-paper?utm_medium=aggregators",
        source="TechCrunch",
    )

    mock_collector = MagicMock()
    mock_collector.fetch_all.return_value = [art_feed1, art_feed2]

    agent = AIPulseAgent(
        collector=mock_collector,
        db=memory_db,
        notifier=mock_notifier,
        summarizer=mock_summarizer,
        initial_run_send_alerts=True,
    )

    with patch("time.sleep"):
        stats = agent.run_cycle()

    # Only 1 unique article saved; 1 detected as cross-feed duplicate
    assert stats["new_articles_saved"] == 1
    assert stats["duplicates_skipped"] == 1
    assert stats["notifications_sent"] == 1
    assert mock_summarizer.summarize_article.call_count == 1
    assert mock_notifier.send_article.call_count == 1


def test_agent_summary_caching_on_retry(memory_db, mock_summarizer):
    """Verify that when an alert fails to send, its summary is cached, and retry reuses cache without calling Gemini."""
    mock_collector = MagicMock()
    mock_article = Article(
        id="retry-art-1",
        title="Breakthrough in LLM Reasoning and Verification",
        link="https://example.com/reasoning-verification",
        source="OpenAI News",
    )
    mock_collector.fetch_all.return_value = [mock_article]

    mock_notifier = MagicMock()
    mock_notifier.is_configured = True
    # First attempt fails (e.g. Telegram network error)
    mock_notifier.send_article.return_value = False

    agent = AIPulseAgent(
        collector=mock_collector,
        db=memory_db,
        notifier=mock_notifier,
        summarizer=mock_summarizer,
        initial_run_send_alerts=True,
    )

    with patch("time.sleep"):
        stats1 = agent.run_cycle()

    assert stats1["notifications_sent"] == 0
    assert stats1["notifications_failed"] == 1
    assert stats1["summaries_generated"] == 1
    assert mock_summarizer.summarize_article.call_count == 1

    # Article in DB should be marked 'failed', but summary is already cached
    art_in_db = memory_db.get_article("retry-art-1")
    assert art_in_db.status == "failed"
    assert art_in_db.ai_summary is not None
    assert art_in_db.summarization_status == "completed"

    # Cycle 2: Telegram is now back online
    mock_notifier.send_article.return_value = True

    with patch("time.sleep"):
        stats2 = agent.run_cycle()

    assert stats2["notifications_sent"] == 1
    assert stats2["notifications_failed"] == 0
    # Must reuse cached summary!
    assert stats2["summaries_cached"] == 1
    assert stats2["summaries_generated"] == 0
    # Gemini should NOT have been called again! Call count remains 1
    assert mock_summarizer.summarize_article.call_count == 1

    # Article should now be marked 'sent'
    assert memory_db.get_article("retry-art-1").status == "sent"


def test_agent_fallback_when_gemini_quota_exhausted(memory_db, mock_summarizer):
    """Verify that when Gemini quota is exhausted, agent falls back to rule-based summary without halting."""
    # Simulate quota exhaustion
    mock_summarizer.can_make_request.return_value = (False, "Free-tier quota was exhausted in current cycle.")

    mock_collector = MagicMock()
    mock_article = Article(
        id="fallback-art-1",
        title="OpenAI Releases New Autonomous Agent Framework",
        link="https://example.com/agent-framework",
        source="OpenAI News",
        summary="A new open-source agent framework for complex reasoning.",
    )
    mock_collector.fetch_all.return_value = [mock_article]

    mock_notifier = MagicMock()
    mock_notifier.is_configured = True
    mock_notifier.send_article.return_value = True

    agent = AIPulseAgent(
        collector=mock_collector,
        db=memory_db,
        notifier=mock_notifier,
        summarizer=mock_summarizer,
        initial_run_send_alerts=True,
    )

    with patch("time.sleep"):
        stats = agent.run_cycle()

    assert stats["notifications_sent"] == 1
    assert stats["summaries_fallback"] == 1
    assert stats["summaries_generated"] == 0
    # Rule-based fallback was invoked
    assert mock_summarizer.create_fallback_analysis.call_count == 1

    # In database, article should have fallback status
    art_in_db = memory_db.get_article("fallback-art-1")
    assert art_in_db.status == "sent"
    assert art_in_db.summarization_status == "fallback"
    assert art_in_db.llm_model == "rule-based-fallback"
    assert art_in_db.ai_summary is not None


def test_agent_fallback_when_gemini_api_fails(memory_db, mock_summarizer):
    """Verify that if Gemini summarize_article returns None (API error), agent uses rule-based fallback."""
    mock_summarizer.can_make_request.return_value = (True, "OK")
    mock_summarizer.summarize_article.side_effect = None
    mock_summarizer.summarize_article.return_value = (None, "API error: 503 Service Unavailable")

    mock_collector = MagicMock()
    mock_article = Article(
        id="api-err-art",
        title="New RAG Vector Database Breakthrough",
        link="https://example.com/rag-db",
        source="Hugging Face Blog",
        summary="High-speed vector indexing for RAG pipelines.",
    )
    mock_collector.fetch_all.return_value = [mock_article]

    mock_notifier = MagicMock()
    mock_notifier.is_configured = True
    mock_notifier.send_article.return_value = True

    agent = AIPulseAgent(
        collector=mock_collector,
        db=memory_db,
        notifier=mock_notifier,
        summarizer=mock_summarizer,
        initial_run_send_alerts=True,
    )

    with patch("time.sleep"):
        stats = agent.run_cycle()

    assert stats["notifications_sent"] == 1
    assert stats["summaries_fallback"] == 1
    assert mock_summarizer.create_fallback_analysis.call_count == 1
    assert memory_db.get_article("api-err-art").summarization_status == "fallback"


def test_agent_polling_loop_bounded(memory_db, mock_summarizer):
    """Verify continuous polling loop runs requested iterations without hanging."""
    mock_collector = MagicMock()
    mock_collector.fetch_all.return_value = []
    mock_notifier = MagicMock()

    agent = AIPulseAgent(
        collector=mock_collector,
        db=memory_db,
        notifier=mock_notifier,
        summarizer=mock_summarizer,
        poll_interval_seconds=1,
    )

    with patch("time.sleep"):
        agent.start_polling(poll_interval=0.01, max_iterations=2)

    assert mock_collector.fetch_all.call_count == 2


def test_agent_dry_run_enriches_and_caches_without_dispatch(memory_db, mock_summarizer):
    """Verify dry-run collects, filters, saves, enriches with Gemini, and caches, but never sends Telegram alerts."""
    mock_collector = MagicMock()
    articles = [
        Article(
            id="dry-art-1",
            title="Google DeepMind Unveils New LLM Reasoning Agent",
            link="https://deepmind.google/blog/reasoning-agent",
            source="Google DeepMind",
            summary="New agentic model with test-time compute scaling.",
        ),
        Article(
            id="dry-art-2",
            title="OpenAI Releases Open Weights for Coding Assistant",
            link="https://openai.com/news/open-weights",
            source="OpenAI News",
            summary="Open weights released for developer tools and coding benchmarks.",
        ),
    ]
    mock_collector.fetch_all.return_value = articles

    mock_notifier = MagicMock()
    mock_notifier.is_configured = True
    mock_notifier.send_article.return_value = True

    agent = AIPulseAgent(
        collector=mock_collector,
        db=memory_db,
        notifier=mock_notifier,
        summarizer=mock_summarizer,
        initial_run_send_alerts=True,
    )

    stats = agent.run_cycle(dry_run=True)

    # 1. Pipeline saved both articles
    assert stats["new_articles_saved"] == 2
    # 2. Both articles were enriched and summarized
    assert stats["summaries_generated"] == 2
    assert mock_summarizer.summarize_article.call_count == 2
    # 3. ZERO Telegram notifications dispatched
    assert stats["notifications_sent"] == 0
    assert stats["notifications_failed"] == 0
    assert mock_notifier.send_article.call_count == 0

    # 4. Verified in SQLite: summaries are cached, but status remains 'discovered'
    art1 = memory_db.get_article("dry-art-1")
    assert art1.status == "discovered"
    assert art1.summarization_status == "completed"
    assert art1.ai_summary is not None

    art2 = memory_db.get_article("dry-art-2")
    assert art2.status == "discovered"
    assert art2.summarization_status == "completed"
    assert art2.ai_summary is not None


def test_agent_dry_run_avoids_reprocessing_cached_summaries(memory_db, mock_summarizer):
    """Verify repeated dry-run cycles do not re-call Gemini for already cached articles."""
    mock_collector = MagicMock()
    article = Article(
        id="dry-cache-1",
        title="Meta Releases LLaMA 4 Open Weights Foundation Model",
        link="https://example.com/llama4",
        source="OpenAI News",
        summary="A new open foundation model.",
    )
    mock_collector.fetch_all.return_value = [article]

    mock_notifier = MagicMock()
    mock_notifier.is_configured = True

    agent = AIPulseAgent(
        collector=mock_collector,
        db=memory_db,
        notifier=mock_notifier,
        summarizer=mock_summarizer,
        initial_run_send_alerts=True,
    )

    # Run 1: Dry run enriches and caches
    stats1 = agent.run_cycle(dry_run=True)
    assert stats1["summaries_generated"] == 1
    assert mock_summarizer.summarize_article.call_count == 1
    assert mock_notifier.send_article.call_count == 0

    # Run 2: Next dry run on same database
    stats2 = agent.run_cycle(dry_run=True)
    assert stats2["new_articles_saved"] == 0
    assert stats2["duplicates_skipped"] == 1
    # Does NOT call Gemini again!
    assert stats2["summaries_generated"] == 0
    assert stats2["summaries_cached"] == 1
    assert mock_summarizer.summarize_article.call_count == 1
    assert mock_notifier.send_article.call_count == 0


def test_agent_normal_run_reuses_cached_dry_run_summaries_and_delivers(memory_db, mock_summarizer):
    """Verify that a subsequent live run picks up dry-run cached summaries without re-calling Gemini."""
    mock_collector = MagicMock()
    article = Article(
        id="dry-to-live-1",
        title="Anthropic Claude Agent Framework Benchmark",
        link="https://example.com/claude-bench",
        source="Google AI Blog",
        summary="Evaluation framework for multi-agent systems.",
    )
    mock_collector.fetch_all.return_value = [article]

    mock_notifier = MagicMock()
    mock_notifier.is_configured = True
    mock_notifier.send_article.return_value = True

    agent = AIPulseAgent(
        collector=mock_collector,
        db=memory_db,
        notifier=mock_notifier,
        summarizer=mock_summarizer,
        initial_run_send_alerts=True,
    )

    # 1. First run: Dry run generates analysis, zero alerts
    stats_dry = agent.run_cycle(dry_run=True)
    assert stats_dry["summaries_generated"] == 1
    assert stats_dry["notifications_sent"] == 0
    assert mock_summarizer.summarize_article.call_count == 1
    assert mock_notifier.send_article.call_count == 0

    # 2. Second run: Normal live cycle
    with patch("time.sleep"):
        stats_live = agent.run_cycle(dry_run=False)

    # Reuses cache from dry-run!
    assert stats_live["summaries_cached"] == 1
    assert stats_live["summaries_generated"] == 0
    # Gemini call count remains 1 (no re-call!)
    assert mock_summarizer.summarize_article.call_count == 1
    # Successfully sent alert to Telegram!
    assert stats_live["notifications_sent"] == 1
    assert mock_notifier.send_article.call_count == 1

    # Article in DB is now marked 'sent'
    assert memory_db.get_article("dry-to-live-1").status == "sent"


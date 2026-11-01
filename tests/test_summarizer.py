import json
import sqlite3
import pytest
from unittest.mock import MagicMock, patch

from src.agent import AIPulseAgent
from src.database import Database
from src.models import Article, ArticleAnalysis, VALID_AI_CATEGORIES
from src.notifier import TelegramNotifier
from src.summarizer import GeminiSummarizer


@pytest.fixture
def memory_db():
    return Database(":memory:")


@pytest.fixture
def sample_article():
    return Article(
        id="sample-101",
        title="OpenAI Releases New Autonomous Coding Agent",
        link="https://openai.com/news/agent-release",
        source="OpenAI News",
        published="2026-10-19T08:00:00Z",
        summary="A new breakthrough in autonomous programming agents capable of resolving SWE-bench tasks.",
    )


# ============================================================================
# 1. Structured Summarization & Schema Validation Tests
# ============================================================================

def test_successful_structured_summarization(sample_article):
    """Verify successful parsing of structured JSON from Gemini API."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "summary": "OpenAI launched a specialized coding agent that automates software tasks.",
        "main_development": "Autonomous agent capable of solving complex engineering benchmarks.",
        "why_it_matters": "Demonstrates practical agentic workflows and tool-use in production codebases.",
        "suggested_action": "Experiment with the API for unit test generation.",
        "category": "AI agents and agentic AI",
        "importance_score": 9,
        "relevance_score": 9,
        "score_explanation": "Directly impacts developer workflows and agent architecture.",
        "is_insufficient_info": False,
    })
    mock_client.models.generate_content.return_value = mock_response

    summarizer = GeminiSummarizer(api_key="valid_dummy_key", client=mock_client)
    analysis, status = summarizer.summarize_article(sample_article)

    assert status == "success"
    assert analysis is not None
    assert analysis.category == "AI agents and agentic AI"
    assert analysis.importance_score == 9
    assert analysis.relevance_score == 9
    assert "OpenAI launched" in analysis.summary
    assert "Demonstrates practical" in analysis.why_it_matters
    assert mock_client.models.generate_content.call_count == 1


def test_markdown_wrapped_json_parsing(sample_article):
    """Verify parser strips ```json ... ``` markdown wrappers without failing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Response wrapped in markdown code fence
    mock_response.text = """```json
    {
        "summary": "New model weights released under open Apache 2.0 license.",
        "main_development": "Weights released for open research.",
        "why_it_matters": "Allows local fine-tuning on consumer hardware.",
        "suggested_action": "Download weights and test with Ollama.",
        "category": "LLMs and model releases",
        "importance_score": 8,
        "relevance_score": 8,
        "score_explanation": "Crucial for local LLM deployment.",
        "is_insufficient_info": false
    }
    ```"""
    mock_client.models.generate_content.return_value = mock_response

    summarizer = GeminiSummarizer(api_key="valid_dummy_key", client=mock_client)
    analysis, status = summarizer.summarize_article(sample_article)

    assert status == "success"
    assert analysis is not None
    assert analysis.category == "LLMs and model releases"
    assert analysis.importance_score == 8


def test_category_validation_and_score_clamping():
    """Verify that invalid/misspelled categories and out-of-range scores are clamped safely."""
    summarizer = GeminiSummarizer(api_key="valid_dummy_key")

    # 1. Out of range scores (score > 10 and score < 1)
    raw_json = json.dumps({
        "summary": "Test summary",
        "main_development": "Test dev",
        "why_it_matters": "Test why",
        "category": "Unknown Random Category",
        "importance_score": 99,   # Should clamp to 10
        "relevance_score": -5,    # Should clamp to 1
    })

    analysis = summarizer._parse_and_validate(raw_json)
    assert analysis.category == "Other AI news"  # Fallback category
    assert analysis.importance_score == 10
    assert analysis.relevance_score == 1

    # 2. Case-insensitive category match
    raw_json_case = json.dumps({
        "summary": "Test summary",
        "main_development": "Test dev",
        "why_it_matters": "Test why",
        "category": "robotics",  # Lowercase
        "importance_score": 7,
        "relevance_score": 7,
    })
    analysis_case = summarizer._parse_and_validate(raw_json_case)
    assert analysis_case.category == "Robotics"


def test_invalid_json_handling_and_bounded_retries(sample_article):
    """Verify that invalid JSON triggers bounded retries and fails safely without crash."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is not valid JSON at all. Just arbitrary text."
    mock_client.models.generate_content.return_value = mock_response

    summarizer = GeminiSummarizer(api_key="valid_dummy_key", client=mock_client)
    with patch("time.sleep"):  # Speed up tests
        analysis, status = summarizer.summarize_article(sample_article, max_retries=1)

    assert analysis is None
    assert "API error" in status or "retries exhausted" in status
    # Initial attempt + 1 retry = 2 calls
    assert mock_client.models.generate_content.call_count == 2


# ============================================================================
# 2. Quota & Rate Limit Protection Tests
# ============================================================================

def test_quota_exhaustion_pauses_llm_processing(sample_article):
    """Verify HTTP 429 / RESOURCE_EXHAUSTED sets quota_exhausted and halts further calls."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: quota exceeded")

    summarizer = GeminiSummarizer(api_key="valid_dummy_key", client=mock_client)
    analysis, status = summarizer.summarize_article(sample_article)

    assert analysis is None
    assert status == "quota_exhausted"
    assert summarizer.quota_exhausted is True

    # Subsequent call in same cycle is blocked without touching the API
    can_request, reason = summarizer.can_make_request()
    assert can_request is False
    assert "quota was exhausted" in reason


def test_per_cycle_and_daily_caps(sample_article):
    """Verify per-cycle and daily request limits are strictly enforced."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "summary": "Summary",
        "main_development": "Dev",
        "why_it_matters": "Why",
        "category": "Generative AI",
        "importance_score": 6,
        "relevance_score": 6,
    })
    mock_client.models.generate_content.return_value = mock_response

    summarizer = GeminiSummarizer(
        api_key="valid_dummy_key",
        max_requests_per_cycle=2,
        daily_limit=5,
        client=mock_client,
    )

    # 1st call
    summarizer.summarize_article(sample_article)
    # 2nd call
    summarizer.summarize_article(sample_article)
    assert summarizer.cycle_requests == 2

    # 3rd call should be blocked by per-cycle limit
    allowed, reason = summarizer.can_make_request()
    assert allowed is False
    assert "Per-cycle limit reached" in reason

    # Reset cycle
    summarizer.reset_cycle()
    assert summarizer.cycle_requests == 0
    assert summarizer.daily_requests == 2
    allowed_after_reset, _ = summarizer.can_make_request()
    assert allowed_after_reset is True


# ============================================================================
# 3. Deterministic Rule-Based Fallback Tests
# ============================================================================

def test_deterministic_rule_based_fallback(sample_article):
    """Verify fallback analysis provides meaningful categorization without calling Gemini."""
    summarizer = GeminiSummarizer(api_key="")  # Unconfigured

    fallback = summarizer.create_fallback_analysis(sample_article)
    assert fallback.category == "AI agents and agentic AI"  # Detected from "Coding Agent"
    assert fallback.importance_score == 6
    assert fallback.relevance_score >= 5
    assert "Relevant to AI engineering" in fallback.why_it_matters
    assert "OpenAI News" in fallback.suggested_action


# ============================================================================
# 4. SQLite Caching & Safe Migration Tests
# ============================================================================

def test_database_caching_and_retrieval(memory_db, sample_article):
    """Verify LLM analysis is persisted and retrieved from cache."""
    memory_db.save_article(sample_article)

    analysis = ArticleAnalysis(
        summary="Cached AI Summary",
        main_development="Main Dev",
        why_it_matters="Student engineering perspective",
        suggested_action="Try it out",
        category="Robotics",
        importance_score=8,
        relevance_score=7,
        score_explanation="Impacts robotic learning algorithms.",
    )

    memory_db.save_analysis(sample_article.id, analysis, model="gemini-2.5-flash")

    cached = memory_db.get_cached_analysis(sample_article.id)
    assert cached is not None
    assert cached["ai_summary"] == "Cached AI Summary"
    assert cached["ai_category"] == "Robotics"
    assert cached["ai_importance"] == 8
    assert cached["llm_model"] == "gemini-2.5-flash"
    assert cached["summarization_status"] == "completed"


def test_safe_schema_migration_preserves_existing_records(tmp_path):
    """Verify that opening an older Phase 3 database preserves all existing articles."""
    db_file = tmp_path / "legacy.db"

    # 1. Create legacy Phase 3 table without Phase 4 columns
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """
            CREATE TABLE articles (
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
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO articles (
                id, normalized_url, title, source, status, discovered_at, updated_at
            ) VALUES ('leg-1', 'https://example.com/leg1', 'Legacy Article', 'Source', 'sent', '2026-10-18', '2026-10-18')
            """
        )
        conn.commit()

    # 2. Instantiate Database on the legacy file (triggers safe migration)
    migrated_db = Database(db_path=db_file)

    # 3. Verify original article is intact
    legacy_art = migrated_db.get_article("leg-1")
    assert legacy_art is not None
    assert legacy_art.title == "Legacy Article"
    assert legacy_art.status == "sent"

    # 4. Verify new Phase 4 columns exist
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(articles)")
        cols = {r[1] for r in cursor.fetchall()}
        assert "ai_summary" in cols
        assert "ai_category" in cols
        assert "ai_importance" in cols
        assert "summarization_status" in cols


# ============================================================================
# 5. Agent Pipeline Integration & Telegram Formatting Tests
# ============================================================================

def test_agent_enriches_with_gemini_and_caches(memory_db, sample_article):
    """Verify agent uses Gemini, stores analysis, and Telegram receives rich card."""
    mock_collector = MagicMock()
    mock_collector.fetch_all.return_value = [sample_article]

    mock_notifier = MagicMock()
    mock_notifier.is_configured = True
    mock_notifier.send_article.return_value = True

    mock_summarizer = MagicMock()
    mock_summarizer.can_make_request.return_value = (True, "OK")
    mock_summarizer.model_name = "gemini-2.5-flash"
    mock_summarizer.summarize_article.return_value = (
        ArticleAnalysis(
            summary="Gemini summary of coding agent.",
            main_development="Autonomous code generation benchmark.",
            why_it_matters="Relevant to agent design patterns.",
            category="AI developer tools and coding assistants",
            importance_score=9,
            relevance_score=8,
        ),
        "success",
    )

    agent = AIPulseAgent(
        collector=mock_collector,
        db=memory_db,
        notifier=mock_notifier,
        summarizer=mock_summarizer,
        initial_run_send_alerts=True,
    )

    stats = agent.run_cycle()

    assert stats["notifications_sent"] == 1
    assert stats["summaries_generated"] == 1
    assert mock_summarizer.summarize_article.call_count == 1

    # Verify article passed to notifier had AI fields populated
    sent_article = mock_notifier.send_article.call_args[0][0]
    assert sent_article.ai_summary == "Gemini summary of coding agent."
    assert sent_article.ai_category == "AI developer tools and coding assistants"
    assert sent_article.ai_importance == 9

    # Verify cached in database
    cached = memory_db.get_cached_analysis(sample_article.id)
    assert cached is not None
    assert cached["ai_summary"] == "Gemini summary of coding agent."


def test_agent_reuses_cache_without_calling_gemini(memory_db, sample_article):
    """Verify that an already summarized article reuses the cached analysis."""
    memory_db.save_article(sample_article)
    memory_db.save_analysis(
        sample_article.id,
        ArticleAnalysis(
            summary="Previously cached summary",
            main_development="Dev",
            why_it_matters="Why",
            category="Generative AI",
            importance_score=7,
            relevance_score=7,
        ),
        model="gemini-2.5-flash",
    )

    mock_summarizer = MagicMock()
    agent = AIPulseAgent(db=memory_db, summarizer=mock_summarizer)

    enriched, sum_type = agent.enrich_article(sample_article)

    assert sum_type == "cached"
    assert enriched.ai_summary == "Previously cached summary"
    # Gemini should NOT have been called
    assert mock_summarizer.summarize_article.call_count == 0


def test_telegram_html_formatting_with_phase4_fields(sample_article):
    """Verify Telegram HTML card formatting with Phase 4 scores and sections."""
    sample_article.ai_summary = "Concise student summary."
    sample_article.ai_category = "AI agents and agentic AI"
    sample_article.ai_importance = 9
    sample_article.ai_relevance = 8
    sample_article.why_it_matters = "Explains agentic architectures."
    sample_article.suggested_action = "Review GitHub repo."
    sample_article.matched_keywords = ["agent", "reasoning"]

    notifier = TelegramNotifier(bot_token="dummy", chat_id="12345")
    html_card = notifier.format_article(sample_article)

    assert "⚡ <b>AI PULSE ALERT</b> | <code>AI AGENTS AND AGENTIC AI</code>" in html_card
    assert "⭐ <b>Importance:</b> 9/10 | 🎯 <b>Relevance:</b> 8/10" in html_card
    assert "📝 <b>Summary:</b> Concise student summary." in html_card
    assert "🎓 <b>Why it matters:</b> Explains agentic architectures." in html_card
    assert "💡 <b>Suggested action:</b> Review GitHub repo." in html_card
    assert f'<a href="{sample_article.link}">' in html_card
    assert len(html_card) < 4000


def test_summarizer_is_configured_detection():
    """Verify is_configured correctly checks for missing or placeholder keys."""
    # Blank key
    s1 = GeminiSummarizer(api_key="")
    assert s1.is_configured is False
    can_req, reason = s1.can_make_request()
    assert can_req is False
    assert "not configured" in reason

    # Default placeholder key
    s2 = GeminiSummarizer(api_key="your_gemini_api_key_here")
    assert s2.is_configured is False

    # Valid dummy key
    s3 = GeminiSummarizer(api_key="AIzaSyDummyValidKey123")
    assert s3.is_configured is True


def test_fallback_analysis_diverse_topics():
    """Verify fallback rule-based analysis accurately assigns categories for various AI topics."""
    summarizer = GeminiSummarizer(api_key="")

    # RAG topic
    art_rag = Article(id="rag-1", title="ChromaDB RAG Pipeline with Vector Search", link="https://ex.com/1", source="Src")
    res_rag = summarizer.create_fallback_analysis(art_rag)
    assert res_rag.category == "RAG and vector databases"

    # Robotics topic
    art_robot = Article(id="rob-1", title="New Humanoid Robot Walk Controller", link="https://ex.com/2", source="Src")
    res_rob = summarizer.create_fallback_analysis(art_robot)
    assert res_rob.category == "Robotics"

    # Coding assistant topic
    art_code = Article(id="cod-1", title="Cursor and Copilot SWE-bench benchmark", link="https://ex.com/3", source="Src")
    res_code = summarizer.create_fallback_analysis(art_code)
    assert res_code.category == "AI developer tools and coding assistants"


# ============================================================================
# 7. Timeout, Retry Bounds & Permanent-Error Classification Tests
# ============================================================================

def test_timeout_triggers_fallback_without_infinite_retry(sample_article):
    """Verify that a timeout exception is treated as transient, retried at most max_retries (default 1)
    times (2 total attempts), and does NOT indefinitely block the cycle."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        "ReadTimeout: timed out waiting for server response"
    )

    summarizer = GeminiSummarizer(
        api_key="valid_dummy_key",
        client=mock_client,
        request_timeout_ms=25000,
    )

    with patch("time.sleep") as mock_sleep:
        analysis, status = summarizer.summarize_article(sample_article)

    # Must fail gracefully (not hang)
    assert analysis is None
    assert "API error" in status or "retries exhausted" in status.lower()
    # At most 2 application-level attempts (1 original + 1 retry default)
    assert mock_client.models.generate_content.call_count == 2
    # Outer backoff sleep was called once between the 2 attempts
    assert mock_sleep.call_count == 1
    # Outer sleep was bounded by 5.0s cap
    for call in mock_sleep.call_args_list:
        assert call.args[0] <= 5.0


def test_permanent_4xx_error_does_not_retry(sample_article):
    """Verify that a permanent 403/401 error exits immediately without consuming extra retries."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        "403 PERMISSION_DENIED: API key not authorized for this model"
    )

    summarizer = GeminiSummarizer(api_key="valid_dummy_key", client=mock_client)

    with patch("time.sleep") as mock_sleep:
        analysis, status = summarizer.summarize_article(sample_article)

    assert analysis is None
    assert "Permanent API error" in status
    # CRITICAL: must not retry a 403 — only 1 call made
    assert mock_client.models.generate_content.call_count == 1
    # No sleep between retries
    assert mock_sleep.call_count == 0


@pytest.mark.parametrize("error_text", [
    "400 Bad Request: Invalid argument",
    "401 Unauthorized: Invalid API key",
    "404 Not Found: Model does not exist",
    "permission_denied: User does not have access",
])
def test_various_permanent_errors_do_not_retry(sample_article, error_text):
    """Verify that any permanent client error exits immediately with only 1 attempt."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(error_text)

    summarizer = GeminiSummarizer(api_key="valid_dummy_key", client=mock_client)

    with patch("time.sleep") as mock_sleep:
        analysis, status = summarizer.summarize_article(sample_article)

    assert analysis is None
    assert "Permanent API error" in status
    assert mock_client.models.generate_content.call_count == 1
    assert mock_sleep.call_count == 0


def test_408_transient_error_retries_and_falls_back(sample_article):
    """Verify that 408 Request Timeout is treated as transient (retried once, max 2 attempts)."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        "408 Request Timeout: Client deadline exceeded"
    )

    summarizer = GeminiSummarizer(api_key="valid_dummy_key", client=mock_client)

    with patch("time.sleep") as mock_sleep:
        analysis, status = summarizer.summarize_article(sample_article)

    assert analysis is None
    assert "API error" in status
    # 1 original + 1 retry = 2 total attempts
    assert mock_client.models.generate_content.call_count == 2
    assert mock_sleep.call_count == 1
    assert mock_sleep.call_args_list[0].args[0] <= 5.0


def test_503_transient_error_retries_bounded(sample_article):
    """Verify that a 503 Service Unavailable is retried up to default 1 retry and falls back."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        "503 UNAVAILABLE: Service temporarily unavailable"
    )

    summarizer = GeminiSummarizer(api_key="valid_dummy_key", client=mock_client)

    with patch("time.sleep") as mock_sleep:
        analysis, status = summarizer.summarize_article(sample_article)

    assert analysis is None
    assert "API error" in status
    # 1 original + 1 retry = 2 total calls
    assert mock_client.models.generate_content.call_count == 2
    # Outer sleep was called once (between 2 attempts)
    assert mock_sleep.call_count == 1
    assert mock_sleep.call_args_list[0].args[0] == 2.0  # Initial delay of 2.0s


def test_client_defaults_sdk_retries_zero_and_timeout_25s():
    """Verify default GeminiSummarizer configuration sets max_sdk_retries=0 and request_timeout_ms=25000."""
    from config import config

    summarizer = GeminiSummarizer(api_key="AIzaSyDummyKey")
    assert summarizer.max_sdk_retries == 0
    assert summarizer.request_timeout_ms == 25000
    assert config.GEMINI_MAX_SDK_RETRIES == 0
    assert config.GEMINI_REQUEST_TIMEOUT_MS == 25000

    mock_client_instance = MagicMock()

    with patch("google.genai.Client", return_value=mock_client_instance) as mock_client_cls:
        client = summarizer._get_client()

    assert client is mock_client_instance
    _, kwargs = mock_client_cls.call_args
    http_opts = kwargs["http_options"]
    assert http_opts.timeout == 25000

    retry_opts = http_opts.retry_options
    # attempts = max_sdk_retries + 1 = 0 + 1 = 1 (no SDK-level retries)
    assert retry_opts.attempts == 1
    assert 503 in retry_opts.http_status_codes
    assert 429 in retry_opts.http_status_codes
    assert 408 in retry_opts.http_status_codes


def test_client_initialized_with_custom_timeout_and_bounded_sdk_retries():
    """Verify _get_client() passes HttpOptions with custom timeout and retry configuration when supplied."""
    from unittest.mock import patch, MagicMock

    summarizer = GeminiSummarizer(
        api_key="AIzaSyDummyKey",
        request_timeout_ms=45000,
        max_sdk_retries=2,
    )

    mock_client_instance = MagicMock()

    with patch("google.genai.Client", return_value=mock_client_instance) as mock_client_cls:
        client = summarizer._get_client()

    assert client is mock_client_instance
    assert mock_client_cls.call_count == 1

    _, kwargs = mock_client_cls.call_args
    assert kwargs["api_key"] == "AIzaSyDummyKey"

    http_opts = kwargs["http_options"]
    assert http_opts.timeout == 45000

    retry_opts = http_opts.retry_options
    # attempts = max_sdk_retries + 1 = 3
    assert retry_opts.attempts == 3
    assert retry_opts.max_delay == 20.0
    # Only transient status codes are retried
    assert 503 in retry_opts.http_status_codes
    assert 429 in retry_opts.http_status_codes
    assert 408 in retry_opts.http_status_codes
    # Permanent 4xx are NOT in the retry list
    assert 400 not in retry_opts.http_status_codes
    assert 401 not in retry_opts.http_status_codes
    assert 403 not in retry_opts.http_status_codes

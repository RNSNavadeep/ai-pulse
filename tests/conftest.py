import pytest
from unittest.mock import MagicMock, patch
from src.database import Database
from src.models import Article, ArticleAnalysis
from src.summarizer import GeminiSummarizer


@pytest.fixture(autouse=True)
def guard_external_network_calls():
    """
    Global safety net: Ensure no live network requests are made to Google GenAI API
    or unmocked external endpoints during unit test execution.
    """
    with patch("google.genai.Client") as mock_genai_cls:
        mock_instance = MagicMock()
        mock_genai_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def memory_db():
    """Provide an isolated, clean in-memory SQLite database for testing."""
    return Database(":memory:")


@pytest.fixture
def mock_summarizer():
    """
    Provide a pre-configured mock GeminiSummarizer that performs deterministic,
    instant structured analysis without external API dependencies.
    """
    summarizer = MagicMock(spec=GeminiSummarizer)
    summarizer.can_make_request.return_value = (True, "OK")
    summarizer.model_name = "gemini-3.8-flash"
    summarizer.cycle_requests = 0
    summarizer.daily_requests = 0
    summarizer.max_requests_per_cycle = 5
    summarizer.daily_limit = 100
    summarizer.quota_exhausted = False

    def _mock_summarize(article: Article, **kwargs):
        analysis = ArticleAnalysis(
            summary=f"Automated AI summary: {article.title}",
            main_development=article.title,
            why_it_matters="Relevant to AI engineering and agent architectures.",
            suggested_action="Review technical paper and benchmark results.",
            category="AI agents and agentic AI" if "agent" in article.title.lower() else "LLMs and model releases",
            importance_score=8,
            relevance_score=8,
            score_explanation="Impacts personal agent development workflows.",
            is_insufficient_info=False,
        )
        return analysis, "success"

    def _mock_fallback(article: Article):
        return ArticleAnalysis(
            summary=article.summary or article.title,
            main_development=article.title,
            why_it_matters="Relevant to AI engineering studies based on topic keywords.",
            suggested_action=f"Review announcement at source: {article.source}",
            category="Other AI news",
            importance_score=6,
            relevance_score=6,
            score_explanation="Calculated via rule-based keyword relevance scoring (Gemini LLM fallback).",
            is_insufficient_info=bool(not article.summary),
        )

    summarizer.summarize_article.side_effect = _mock_summarize
    summarizer.create_fallback_analysis.side_effect = _mock_fallback
    return summarizer

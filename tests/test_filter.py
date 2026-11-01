import pytest
from src.filter import TopicFilter
from src.models import Article


def test_keyword_boundary_matching():
    filter_engine = TopicFilter()

    # Should match "rag" as an acronym
    art_rag = Article(
        id="1",
        title="Modern RAG architectures for production",
        link="https://example.com/rag",
        source="Tech Blog",
    )
    is_rel, kws, score = filter_engine.score_article(art_rag, is_pure_ai_source=False)
    assert is_rel is True
    assert "rag" in kws

    # Should NOT match "rag" inside "dragon" or "courage"
    art_no_rag = Article(
        id="2",
        title="How dragons showed great courage in stories",
        link="https://example.com/dragons",
        source="General Blog",
    )
    is_rel, kws, score = filter_engine.score_article(art_no_rag, is_pure_ai_source=False)
    assert is_rel is False
    assert kws == []
    assert score == 0.0


def test_pure_ai_source_vs_general_source():
    filter_engine = TopicFilter()

    # Article from an official pure AI source with no explicit AI buzzwords
    art_pure_ai = Article(
        id="3",
        title="Welcome our new research cohort for fall",
        link="https://openai.com/cohort",
        source="OpenAI News",
    )
    is_rel, kws, score = filter_engine.score_article(art_pure_ai, is_pure_ai_source=True)
    assert is_rel is True
    assert score >= 0.5

    # Article from a general tech source with no AI keywords
    art_general = Article(
        id="4",
        title="Quarterly earnings and tax incentives for tech",
        link="https://techcrunch.com/earnings",
        source="TechCrunch",
    )
    is_rel, kws, score = filter_engine.score_article(art_general, is_pure_ai_source=False)
    assert is_rel is False
    assert score == 0.0


def test_filter_articles_ranking():
    filter_engine = TopicFilter()

    art_high = Article(
        id="10",
        title="New Open-Weights LLM with Autonomous AI Agent capabilities",
        link="https://example.com/high",
        source="TechCrunch",
        summary="Featuring reinforcement learning and reasoning benchmarks.",
    )
    art_low = Article(
        id="11",
        title="Cloud infrastructure updates",
        link="https://example.com/low",
        source="TechCrunch",
        summary="A tiny note on machine learning support.",
    )
    art_none = Article(
        id="12",
        title="Smartphone battery charging tips",
        link="https://example.com/none",
        source="TechCrunch",
        summary="How to make battery last longer.",
    )

    sources_lookup = {"TechCrunch": {"is_pure_ai": False}}
    filtered = filter_engine.filter_articles([art_none, art_low, art_high], sources_lookup=sources_lookup)

    assert len(filtered) == 2
    assert filtered[0].id == "10"  # Highest relevance first
    assert filtered[1].id == "11"
    assert filtered[0].relevance_score > filtered[1].relevance_score

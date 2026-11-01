import pytest
from unittest.mock import patch, MagicMock
from src.collector import NewsCollector, clean_html, normalize_url
from src.models import Article


def test_clean_html():
    raw = "<p>New breakthrough in <b>LLM reasoning</b> &amp; memory!</p>"
    cleaned = clean_html(raw)
    assert cleaned == "New breakthrough in LLM reasoning & memory!"
    assert clean_html(None) == ""


def test_normalize_url():
    # Strips tracking params while preserving search_query and paths
    url_with_tracking = "https://example.com/article?utm_source=twitter&utm_medium=social&search_query=cat:cs.AI"
    normalized = normalize_url(url_with_tracking)
    assert "utm_source" not in normalized
    assert "utm_medium" not in normalized
    assert "search_query=cat%3Acs.AI" in normalized or "search_query=cat:cs.AI" in normalized

    # Handles whitespace and trailing fragments cleanly
    assert normalize_url("  https://EXAMPLE.com/page/  ") == "https://example.com/page/"
    assert normalize_url("") == ""
    assert normalize_url(None) == ""


SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>AI News Feed</title>
    <link>https://example.com</link>
    <description>Sample feed</description>
    <item>
      <title>GPT-5 Architecture Leaks &amp; Analysis</title>
      <link>https://example.com/gpt-5-analysis?utm_source=rss</link>
      <guid>https://example.com/gpt-5-analysis</guid>
      <pubDate>Mon, 13 Oct 2026 09:00:00 GMT</pubDate>
      <description>&lt;p&gt;Researchers discover new multi-modal scaling laws.&lt;/p&gt;</description>
    </item>
    <!-- Duplicate entry with different utm parameter -->
    <item>
      <title>GPT-5 Architecture Leaks &amp; Analysis Duplicate</title>
      <link>https://example.com/gpt-5-analysis?utm_source=newsletter</link>
      <description>Duplicate item</description>
    </item>
    <!-- Item with missing title and description -->
    <item>
      <link>https://example.com/item-missing-fields</link>
    </item>
  </channel>
</rss>
"""


@patch("src.collector.requests.get")
def test_fetch_feed_success_and_deduplication(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = SAMPLE_RSS_XML.encode("utf-8")
    mock_get.return_value = mock_resp

    collector = NewsCollector()
    source_info = {"name": "Test Source", "url": "https://example.com/rss", "category": "releases"}
    articles = collector.fetch_feed(source_info, max_items=5)

    # Entry 1 is kept, Entry 2 (duplicate link after normalization) is ignored, Entry 3 is kept
    assert len(articles) == 2
    art1 = articles[0]
    assert art1.title == "GPT-5 Architecture Leaks & Analysis"
    assert art1.link == "https://example.com/gpt-5-analysis"
    assert art1.source == "Test Source"
    assert "Researchers discover new multi-modal scaling laws." in art1.summary

    art2 = articles[1]
    assert art2.title == "Untitled AI Update"
    assert art2.summary is None
    assert art2.link == "https://example.com/item-missing-fields"


@patch("src.collector.requests.get")
def test_fetch_feed_network_error(mock_get):
    import requests
    mock_get.side_effect = requests.RequestException("Connection timed out")

    collector = NewsCollector()
    source_info = {"name": "Test Source", "url": "https://example.com/rss"}
    articles = collector.fetch_feed(source_info)

    assert articles == []


@patch("src.collector.requests.get")
def test_fetch_all_skips_disabled_sources(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = SAMPLE_RSS_XML.encode("utf-8")
    mock_get.return_value = mock_resp

    collector = NewsCollector()
    sources = [
        {"name": "Enabled Source", "url": "https://example.com/rss", "enabled": True},
        {"name": "Disabled Source", "url": "https://disabled.com/rss", "enabled": False},
    ]

    articles = collector.fetch_all(sources, max_items_per_source=2)
    assert len(articles) == 2
    # Verify mock_get was called only once for the enabled source
    assert mock_get.call_count == 1


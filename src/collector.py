import html
import logging
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Set
import feedparser
import requests

from src.models import Article

logger = logging.getLogger(__name__)

TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
}


def clean_html(raw_html: Optional[str]) -> str:
    """Remove HTML tags and unescape HTML entities for clean plaintext snippets."""
    if not raw_html:
        return ""
    # Strip HTML tags
    clean = re.sub(r"<[^>]+>", " ", raw_html)
    # Decode HTML entities (e.g. &amp; -> &)
    clean = html.unescape(clean)
    # Collapse multiple whitespaces
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def normalize_url(raw_url: Optional[str]) -> str:
    """
    Safely normalize article URL:
    - Strips whitespace.
    - Standardizes scheme and host to lowercase.
    - Strips promotional/tracking query parameters (utm_*, fbclid, etc.)
      while strictly preserving meaningful parameters (e.g. search_query, id).
    """
    if not raw_url:
        return ""
    url = raw_url.strip()
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return url

        query_params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        filtered_params = [
            (k, v) for k, v in query_params if k.lower() not in TRACKING_QUERY_PARAMS
        ]
        clean_query = urllib.parse.urlencode(filtered_params)

        normalized = urllib.parse.urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path,
                parsed.params,
                clean_query,
                parsed.fragment,
            )
        )
        return normalized
    except Exception:
        return url


class NewsCollector:
    """Collects, parses, and normalizes news articles from RSS and Atom feeds."""

    def __init__(self, request_timeout: int = 10):
        self.request_timeout = request_timeout
        self.headers = {
            "User-Agent": "AIPulseAgent/2.0 (+https://github.com/aipulse-news-agent; learning project)"
        }

    def fetch_feed(
        self, source_info: Dict[str, Any], max_items: int = 5
    ) -> List[Article]:
        """
        Fetch and parse a single RSS/Atom feed.

        Args:
            source_info: Dictionary containing 'name', 'url', optional 'category', and 'enabled'.
            max_items: Maximum number of recent articles to parse.

        Returns:
            List of parsed Article objects.
        """
        name = source_info.get("name", "Unknown Source")
        url = source_info.get("url", "").strip()
        category = source_info.get("category", "general")

        if not url:
            logger.warning("Skipping source '%s': Missing URL", name)
            return []

        logger.info("Fetching feed: %s (%s)", name, url)

        try:
            response = requests.get(url, headers=self.headers, timeout=self.request_timeout)
            response.raise_for_status()
            parsed = feedparser.parse(response.content)
        except requests.exceptions.Timeout:
            logger.error("Timeout fetching %s (%ss limit reached)", name, self.request_timeout)
            return []
        except requests.RequestException as e:
            logger.error("Network error fetching %s: %s", name, e)
            return []
        except Exception as e:
            logger.error("Unexpected error parsing %s: %s", name, e)
            return []

        if parsed.bozo and not parsed.entries:
            logger.warning(
                "Feed '%s' returned bozo error with no entries: %s",
                name,
                parsed.bozo_exception,
            )
            return []

        articles: List[Article] = []
        seen_links_in_feed: Set[str] = set()

        for entry in parsed.entries:
            if len(articles) >= max_items:
                break

            raw_link = getattr(entry, "link", "") or ""
            link = normalize_url(raw_link)

            # Skip entries without valid link
            if not link:
                continue

            # Duplicate guard within the same feed
            if link in seen_links_in_feed:
                continue
            seen_links_in_feed.add(link)

            title_raw = getattr(entry, "title", "") or "Untitled AI Update"
            title = clean_html(title_raw)

            # Determine unique identifier (GUID, ID, or normalized link)
            article_id = (
                getattr(entry, "id", None)
                or getattr(entry, "guid", None)
                or link
            )

            # Extract publication timestamp string
            published = (
                getattr(entry, "published", None)
                or getattr(entry, "updated", None)
                or getattr(entry, "pubDate", None)
            )

            # Extract summary or snippet
            summary_raw = ""
            if hasattr(entry, "summary"):
                summary_raw = entry.summary
            elif hasattr(entry, "description"):
                summary_raw = entry.description
            elif hasattr(entry, "content") and entry.content:
                summary_raw = entry.content[0].get("value", "")

            summary_clean = clean_html(summary_raw)
            if len(summary_clean) > 350:
                summary_clean = summary_clean[:347] + "..."

            articles.append(
                Article(
                    id=str(article_id),
                    title=title,
                    link=link,
                    source=name,
                    published=published,
                    summary=summary_clean if summary_clean else None,
                    category=category,
                )
            )

        logger.info("Successfully parsed %d items from %s", len(articles), name)
        return articles

    def fetch_all(
        self, sources: List[Dict[str, Any]], max_items_per_source: int = 5
    ) -> List[Article]:
        """
        Fetch articles across all configured sources, skipping disabled sources
        and deduplicating identical links in this collection run.
        """
        all_articles: List[Article] = []
        seen_global_links: Set[str] = set()

        for source in sources:
            if not source.get("enabled", True):
                logger.info("Skipping disabled source: %s", source.get("name"))
                continue

            feed_articles = self.fetch_feed(source, max_items=max_items_per_source)
            for art in feed_articles:
                if art.link not in seen_global_links:
                    seen_global_links.add(art.link)
                    all_articles.append(art)
                else:
                    logger.debug("Duplicate link skipped across feeds: %s", art.link)

        return all_articles


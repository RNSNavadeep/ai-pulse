import html
import logging
from typing import List, Optional, Tuple
import requests

from src.models import Article

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Manages mobile alerts through the Telegram Bot API."""

    BASE_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, bot_token: str, chat_id: str, timeout: int = 10):
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Check if bot token and chat id are non-empty and non-default."""
        return bool(
            self.bot_token
            and self.chat_id
            and self.bot_token != "your_bot_token_here"
            and self.chat_id != "your_chat_id_here"
        )

    def _call_api(self, method: str, payload: Optional[dict] = None) -> Tuple[bool, dict]:
        """Make an authenticated HTTP POST request to Telegram Bot API."""
        if not self.bot_token:
            return False, {"description": "Bot token is not configured."}

        url = self.BASE_URL.format(token=self.bot_token, method=method)
        try:
            response = requests.post(url, json=payload or {}, timeout=self.timeout)
            data = response.json()
            if response.status_code == 200 and data.get("ok"):
                return True, data
            elif response.status_code == 429:
                retry_after = data.get("parameters", {}).get("retry_after", 2)
                logger.warning("Telegram rate limit reached (HTTP 429). Retry after %ds", retry_after)
                return False, {"description": f"Rate limit reached. Retry after {retry_after}s"}
            else:
                desc = data.get("description", f"HTTP {response.status_code}")
                logger.error("Telegram API '%s' failed: %s", method, desc)
                return False, {"description": desc}
        except requests.RequestException as e:
            logger.error("Network error connecting to Telegram API: %s", e)
            return False, {"description": str(e)}

    def verify_bot(self) -> Tuple[bool, str]:
        """Verify the bot token by calling 'getMe'."""
        ok, res = self._call_api("getMe")
        if ok:
            bot_info = res.get("result", {})
            username = bot_info.get("username", "UnknownBot")
            first_name = bot_info.get("first_name", "")
            return True, f"Connected to @{username} ({first_name})"
        return False, res.get("description", "Unknown error")

    def get_latest_chat_id(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Poll 'getUpdates' to discover the chat ID of a user who initiated contact with the bot.
        Returns: (chat_id, user_display_name) or (None, None)
        """
        ok, res = self._call_api("getUpdates")
        if not ok:
            return None, None
        updates = res.get("result", [])
        if not updates:
            return None, None

        for update in reversed(updates):
            msg = update.get("message") or update.get("channel_post") or update.get("my_chat_member")
            if msg:
                chat = msg.get("chat") or {}
                chat_id = str(chat.get("id", ""))
                user_name = chat.get("first_name") or chat.get("title") or "User"
                if chat_id:
                    return chat_id, user_name
        return None, None

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a text notification to the configured chat.
        Truncates messages exceeding Telegram's 4096 character limit.
        """
        if not self.is_configured:
            logger.warning("Telegram is not configured. Message skipped.")
            return False

        # Telegram hard limit: 4096 UTF-8 characters
        if len(text) > 4000:
            text = text[:3990] + "\n...[truncated]"

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": False,
        }

        ok, res = self._call_api("sendMessage", payload)
        if ok:
            logger.info("Telegram notification sent successfully to chat %s", self.chat_id)
            return True
        else:
            logger.error("Failed to send message: %s", res.get("description"))
            return False

    def format_article(self, article: Article) -> str:
        """
        Format an Article into a clean HTML card for Telegram.
        Displays AI category, importance/relevance scores, concise summary,
        student-focused 'Why it matters', and canonical source link.
        """
        safe_title = html.escape(article.title)
        safe_source = html.escape(article.source)

        # Category: prioritize AI categorized tag, fallback to feed category
        cat_display = article.ai_category or article.category or "NEWS"
        safe_category = html.escape(cat_display.upper())

        # Header banner
        lines = [
            f"⚡ <b>AI PULSE ALERT</b> | <code>{safe_category}</code>",
        ]

        # Importance & Relevance badge (if available)
        if article.ai_importance is not None and article.ai_relevance is not None:
            lines.append(
                f"⭐ <b>Importance:</b> {article.ai_importance}/10 | "
                f"🎯 <b>Relevance:</b> {article.ai_relevance}/10"
            )

        lines.append(f"<b>{safe_title}</b>")
        lines.append("")
        lines.append(f"📰 <b>Source:</b> {safe_source}")

        if article.published:
            lines.append(f"🕒 <b>Published:</b> {html.escape(article.published)}")

        if article.matched_keywords:
            tags = " ".join([f"#{kw.replace(' ', '_').replace('-', '_')}" for kw in article.matched_keywords[:4]])
            lines.append(f"🏷️ <b>Topics:</b> <code>{html.escape(tags)}</code>")

        # Summary section
        lines.append("")
        if article.ai_summary:
            summary_label = (
                "Summary (Rule-based Fallback)"
                if article.summarization_status == "fallback"
                else "Summary"
            )
            lines.append(f"📝 <b>{summary_label}:</b> {html.escape(article.ai_summary)}")
        elif article.summary:
            lines.append(f"📝 <b>Summary:</b> {html.escape(article.summary)}")

        # Why it matters to an AI engineering student
        if article.why_it_matters:
            lines.append("")
            lines.append(f"🎓 <b>Why it matters:</b> {html.escape(article.why_it_matters)}")

        # Suggested next action / takeaway
        if article.suggested_action:
            lines.append(f"💡 <b>Suggested action:</b> {html.escape(article.suggested_action)}")

        lines.append("")
        lines.append(f"🔗 <a href=\"{article.link}\">Read Full Article</a>")

        return "\n".join(lines)

    def send_article(self, article: Article) -> bool:
        """Format and send a single news article."""
        formatted_text = self.format_article(article)
        return self.send_message(formatted_text)

    def send_articles_batch(
        self,
        articles: List[Article],
        max_count: int = 5,
        delay_seconds: float = 1.0,
    ) -> int:
        """
        Send a capped batch of articles with rate-limiting pauses to prevent flood.
        Returns the count of successfully sent articles.
        """
        import time

        sent_count = 0
        batch = articles[:max_count]
        logger.info("Dispatching %d articles to Telegram (max cap: %d)...", len(batch), max_count)

        for idx, art in enumerate(batch):
            if self.send_article(art):
                sent_count += 1
            if idx < len(batch) - 1 and delay_seconds > 0:
                time.sleep(delay_seconds)

        return sent_count


import pytest
from unittest.mock import patch, MagicMock
from src.notifier import TelegramNotifier
from src.models import Article


def test_telegram_is_configured():
    # Blank/default configurations
    n1 = TelegramNotifier("", "")
    assert not n1.is_configured

    n2 = TelegramNotifier("your_bot_token_here", "your_chat_id_here")
    assert not n2.is_configured

    # Valid config
    n3 = TelegramNotifier("123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11", "987654321")
    assert n3.is_configured


def test_format_article():
    notifier = TelegramNotifier("fake_token", "fake_chat")
    article = Article(
        id="test-id",
        title="OpenAI announces <New Model> & Features",
        link="https://openai.com/news/test",
        source="OpenAI News",
        published="2026-10-13",
        summary="A major breakthrough in AI agent planning.",
        category="releases",
    )
    formatted = notifier.format_article(article)

    # Validate HTML escaping
    assert "&lt;New Model&gt;" in formatted
    assert "&amp;" in formatted
    assert "<b>Source:</b> OpenAI News" in formatted
    assert '<a href="https://openai.com/news/test">Read Full Article</a>' in formatted


@patch("src.notifier.requests.post")
def test_send_message_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True, "result": {"message_id": 42}}
    mock_post.return_value = mock_resp

    notifier = TelegramNotifier("123456:token", "987654")
    success = notifier.send_message("Test message")

    assert success is True
    mock_post.assert_called_once()


@patch("src.notifier.requests.post")
def test_send_message_failure(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {"ok": False, "description": "Bad Request: chat not found"}
    mock_post.return_value = mock_resp

    notifier = TelegramNotifier("123456:token", "wrong_chat")
    success = notifier.send_message("Test message")

    assert success is False


@patch("src.notifier.requests.post")
def test_get_latest_chat_id(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "ok": True,
        "result": [
            {
                "update_id": 100,
                "message": {
                    "chat": {"id": 12345678, "first_name": "Alice"},
                    "text": "/start",
                },
            }
        ],
    }
    mock_post.return_value = mock_resp

    notifier = TelegramNotifier("123456:token", "")
    chat_id, name = notifier.get_latest_chat_id()

    assert chat_id == "12345678"
    assert name == "Alice"


@patch("src.notifier.requests.post")
def test_rate_limit_429_handling(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.json.return_value = {
        "ok": False,
        "description": "Too Many Requests: retry after 5",
        "parameters": {"retry_after": 5},
    }
    mock_post.return_value = mock_resp

    notifier = TelegramNotifier("123456:token", "123456")
    success = notifier.send_message("Test message")
    assert success is False


@patch("src.notifier.TelegramNotifier.send_article")
def test_send_articles_batch(mock_send):
    mock_send.return_value = True

    articles = [
        Article(id=str(i), title=f"Title {i}", link=f"https://example.com/{i}", source="Test")
        for i in range(10)
    ]

    notifier = TelegramNotifier("123456:token", "123456")
    sent = notifier.send_articles_batch(articles, max_count=3, delay_seconds=0.0)

    assert sent == 3
    assert mock_send.call_count == 3



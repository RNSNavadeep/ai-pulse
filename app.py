import argparse
import logging
import sys
from config import config
from src.agent import AIPulseAgent
from src.collector import NewsCollector
from src.database import Database
from src.filter import TopicFilter
from src.models import Article
from src.notifier import TelegramNotifier
from src.summarizer import GeminiSummarizer

# Ensure UTF-8 output on Windows consoles to prevent charmap UnicodeEncodeErrors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AIPulseAgent")


def run_test_rss(collector: NewsCollector):
    """Test RSS feed collection from configured feeds."""
    print("\n" + "=" * 60)
    print("📡 [TEST] RSS News Collection (Preview)")
    print("=" * 60)

    sources = config.get_sources()
    articles = collector.fetch_all(sources, max_items_per_source=2)

    if not articles:
        print("❌ No articles fetched. Check network connection or feed URLs.")
        return []

    print(f"\n✅ Successfully fetched {len(articles)} articles across {len(sources)} configured feeds:\n")
    for idx, art in enumerate(articles, start=1):
        print(f"[{idx}] {art.title}")
        print(f"    Source:    {art.source} ({art.category})")
        print(f"    Published: {art.published or 'N/A'}")
        print(f"    Link:      {art.link}")
        if art.summary:
            print(f"    Snippet:   {art.summary[:120]}...")
        print("-" * 60)

    return articles


def run_test_telegram(notifier: TelegramNotifier):
    """Test Telegram bot connection and send a verification message."""
    print("\n" + "=" * 60)
    print("📱 [TEST] Telegram Bot Notification")
    print("=" * 60)

    if not notifier.is_configured:
        print("⚠️ Telegram is not yet configured in .env file.")
        print("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your local .env file.")
        print("See .env.example for instructions.")
        return False

    print("1. Verifying Bot Token with Telegram API...")
    ok, info = notifier.verify_bot()
    if not ok:
        print(f"❌ Bot verification failed: {info}")
        return False
    print(f"✅ Bot verified successfully: {info}")

    print(f"2. Sending test notification to Chat ID: {notifier.chat_id}...")
    test_msg = (
        "🚀 <b>AI Pulse Agent - Test Notification</b>\n\n"
        "Your AI Pulse notification pipeline is working!\n"
        "Phase 3 connection test passed successfully. 🎉"
    )
    sent = notifier.send_message(test_msg)
    if sent:
        print("✅ Test message delivered! Check your Telegram app.")
        return True
    else:
        print("❌ Failed to deliver message. Check Chat ID and bot permissions.")
        return False


def run_setup_bot(notifier: TelegramNotifier):
    """Interactive helper to verify bot and automatically detect Chat ID upon user interaction."""
    import time
    print("\n" + "=" * 60)
    print("🤖 [SETUP] Telegram Bot Auto-Discovery")
    print("=" * 60)

    ok, info = notifier.verify_bot()
    if not ok:
        print(f"❌ Failed to connect with TELEGRAM_BOT_TOKEN: {info}")
        print("Please check the token in your .env file.")
        return False

    print(f"✅ Bot Verified: {info}")
    print("\n👉 ACTION REQUIRED ON YOUR PHONE:")
    print("1. Open Telegram and search for your bot")
    print("2. Tap 'START' (or send any message like 'hello') to your bot.")
    print("\nListening for your message (checking for 45 seconds)...")

    detected_id = None
    user_name = "User"
    for _ in range(15):
        chat_id, name = notifier.get_latest_chat_id()
        if chat_id:
            detected_id = chat_id
            user_name = name or "User"
            break
        time.sleep(3)
        print(".", end="", flush=True)

    print()
    if not detected_id:
        print("⏳ No message received yet.")
        print("Make sure you opened your bot in Telegram and tapped START (or sent a message).")
        print("Then run: .\\.venv\\Scripts\\python.exe app.py --setup-bot")
        return False

    print(f"🎉 Success! Detected message from {user_name} (Chat ID: {detected_id})")

    env_path = config.BASE_DIR / ".env"
    lines = []
    found_chat_id = False
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("TELEGRAM_CHAT_ID="):
                    lines.append(f"TELEGRAM_CHAT_ID={detected_id}\n")
                    found_chat_id = True
                else:
                    lines.append(line)
        if not found_chat_id:
            lines.append(f"TELEGRAM_CHAT_ID={detected_id}\n")
    else:
        lines = [
            f"TELEGRAM_BOT_TOKEN={notifier.bot_token}\n",
            f"TELEGRAM_CHAT_ID={detected_id}\n",
            "POLL_INTERVAL_SECONDS=300\n",
            "LOG_LEVEL=INFO\n",
        ]

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("💾 Saved TELEGRAM_CHAT_ID to your local .env file.")

    notifier.chat_id = detected_id
    test_msg = (
        "🚀 <b>AI Pulse Agent Connected!</b>\n\n"
        f"Hello {user_name}! Your Telegram alert pipeline is officially verified.\n"
        "You will receive curated AI engineering updates straight to this chat. 🎉"
    )
    print("📤 Sending welcome notification to your phone...")
    if notifier.send_message(test_msg):
        print("✅ Welcome notification delivered! Check your Telegram app.")
        return True
    else:
        print("❌ Could not deliver test message.")
        return False


def run_test_gemini(summarizer: GeminiSummarizer):
    """Test Gemini API connection, prompt structure, or fallback mode."""
    print("\n" + "=" * 60)
    print("🧠 [TEST] Phase 4: Gemini LLM Integration")
    print("=" * 60)
    print(f"Model:                {summarizer.model_name}")
    print(f"Per-Cycle Cap:        {summarizer.max_requests_per_cycle} calls")
    print(f"Daily Cap:            {summarizer.daily_limit} calls")
    print(f"API Key Configured:   {'✅ Yes' if summarizer.is_configured else '❌ No (Rule-based fallback active)'}")
    print("-" * 60)

    sample_article = Article(
        id="test-gemini-sample",
        title="OpenAI Releases GPT-5 with Autonomous Agent Reasoning",
        link="https://openai.com/news/gpt-5",
        source="OpenAI News",
        published="2026-10-19T10:00:00Z",
        summary="A major model release featuring autonomous multi-step tool use, code generation, and evaluation on real-world engineering benchmarks.",
    )

    if not summarizer.is_configured:
        print("ℹ️ Note: GEMINI_API_KEY is not configured in your .env file.")
        print("Demonstrating deterministic rule-based fallback analysis:\n")
        fallback = summarizer.create_fallback_analysis(sample_article)
        print(f"📌 Category:        {fallback.category}")
        print(f"⭐ Importance:      {fallback.importance_score}/10")
        print(f"🎯 Relevance:       {fallback.relevance_score}/10")
        print(f"📝 Summary:         {fallback.summary}")
        print(f"🎓 Why It Matters:  {fallback.why_it_matters}")
        print(f"💡 Action:          {fallback.suggested_action}")
        print(f"📋 Score Reason:    {fallback.score_explanation}")
        print("\nTo enable live Gemini AI summarization:")
        print("1. Get a free API key at https://aistudio.google.com/app/apikey")
        print("2. Add GEMINI_API_KEY=your_key to your .env file.")
        return True

    print("Sending sample article metadata to Gemini for structured analysis...")
    analysis, status = summarizer.summarize_article(sample_article)
    if analysis:
        print("✅ Gemini structured analysis received successfully:\n")
        print(f"📌 Category:        {analysis.category}")
        print(f"⭐ Importance:      {analysis.importance_score}/10")
        print(f"🎯 Relevance:       {analysis.relevance_score}/10")
        print(f"📝 Summary:         {analysis.summary}")
        print(f"🎓 Why It Matters:  {analysis.why_it_matters}")
        print(f"💡 Action:          {analysis.suggested_action}")
        print(f"📋 Score Reason:    {analysis.score_explanation}")
        return True
    else:
        print(f"❌ Gemini analysis failed: {status}")
        return False


def print_status(db: Database):
    """Print database statistics."""
    stats = db.get_stats()
    print("\n" + "=" * 60)
    print("📊 AI Pulse Agent - Database Status")
    print("=" * 60)
    print(f"📁 Path:        {db.db_path}")
    print(f"📚 Total:       {stats['total']} articles recorded")
    print(f"🆕 Discovered:  {stats['discovered']} (awaiting alert)")
    print(f"⏳ Pending:     {stats['pending']} (in progress)")
    print(f"✅ Sent:        {stats['sent']} (delivered to Telegram)")
    print(f"❌ Failed:      {stats['failed']} (retry attempts remaining)")
    print(f"🚫 Ignored:     {stats['ignored']} (excluded from all pipelines)")
    print("-" * 60)
    print("🧠 Phase 4: Gemini LLM Summaries")
    print(f"   • Completed (AI Summaries): {stats.get('summaries_completed', 0)}")
    print(f"   • Fallback (Rule-Based):    {stats.get('summaries_fallback', 0)}")
    print(f"   • Insufficient Context:     {stats.get('summaries_insufficient', 0)}")
    print(f"   • Failed Summarization:     {stats.get('summaries_failed', 0)}")
    print("   ─ Pending breakdown (will NOT be re-processed) ─────────")
    print(f"   • Actionable (next cycle):  {stats.get('summaries_pending_actionable', 0)}"
          "  ← agent will summarize these")
    print(f"   • Historic (pre-Phase 4):   {stats.get('summaries_pending_historic', 0)}"
          "  ← already sent, never re-summarized")
    print(f"   • Ignored (backlog-skipped):{stats.get('summaries_pending_ignored', 0)}"
          "  ← permanently excluded, never processed")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AI Pulse Agent - Core Controller")
    parser.add_argument("--run-cycle", action="store_true", help="Run one full news monitoring & alert cycle")
    parser.add_argument("--poll", action="store_true", help="Run continuous autonomous polling loop")
    parser.add_argument("--interval", type=int, default=config.POLL_INTERVAL_SECONDS, help="Polling interval in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Collect, filter, and persist without sending Telegram alerts")
    parser.add_argument("--limit", type=int, default=config.MAX_NOTIFICATIONS_PER_RUN, help="Maximum notifications to dispatch per cycle")
    parser.add_argument("--send-initial-alerts", action="store_true", help="Send alerts on first-run instead of establishing quiet baseline")
    parser.add_argument("--status", action="store_true", help="View SQLite database article counts and notification statuses")
    parser.add_argument("--reset-db", action="store_true", help="Reset local SQLite database schema and wipe records")
    parser.add_argument("--test-rss", action="store_true", help="Fetch and print sample articles from RSS feeds")
    parser.add_argument("--test-telegram", action="store_true", help="Send a test message to verify Telegram credentials")
    parser.add_argument("--test-gemini", action="store_true", help="Test Gemini AI summarization or fallback mode")
    parser.add_argument("--setup-bot", action="store_true", help="Auto-detect Chat ID and verify Telegram bot connection")
    parser.add_argument("--run-once", action="store_true", help="Legacy alias for --run-cycle")

    args = parser.parse_args()

    collector = NewsCollector()
    topic_filter = TopicFilter()
    db = Database()
    notifier = TelegramNotifier(
        bot_token=config.TELEGRAM_BOT_TOKEN,
        chat_id=config.TELEGRAM_CHAT_ID,
    )
    summarizer = GeminiSummarizer()

    agent = AIPulseAgent(
        collector=collector,
        topic_filter=topic_filter,
        db=db,
        notifier=notifier,
        summarizer=summarizer,
        poll_interval_seconds=args.interval,
        max_notifications_per_cycle=args.limit,
        initial_run_send_alerts=args.send_initial_alerts or config.INITIAL_RUN_SEND_ALERTS,
    )

    if args.status:
        print_status(db)
    elif args.reset_db:
        db.reset()
        print("✅ Local database table reset successfully.")
    elif args.poll:
        agent.start_polling(poll_interval=args.interval)
    elif args.run_cycle or args.run_once:
        print("\n" + "=" * 60)
        print("⚡ [PHASE 4] Executing Single Monitoring & Intelligence Cycle")
        print("=" * 60)
        stats = agent.run_cycle(dry_run=args.dry_run)
        print("\n" + "-" * 60)
        print(f"📊 Cycle Summary:")
        print(f"   • Feeds Queried:       {stats['sources_queried']}")
        print(f"   • Articles Ingested:   {stats['articles_fetched']}")
        print(f"   • Relevant AI:         {stats['relevant_ai_articles']}")
        print(f"   • New Saved:           {stats['new_articles_saved']}")
        print(f"   • Duplicates:          {stats['duplicates_skipped']}")
        print(f"   • AI Summaries:        {stats.get('summaries_generated', 0)}")
        print(f"   • Cached Summaries:    {stats.get('summaries_cached', 0)}")
        print(f"   • Fallback Summaries:  {stats.get('summaries_fallback', 0)}")
        print(f"   • Alerts Sent:         {stats['notifications_sent']}")
        print(f"   • Alerts Failed:       {stats['notifications_failed']}")
        print(f"   • Execution Time:      {stats['duration_seconds']}s")
        if stats["first_run_seeded"]:
            print("   🛡️ (Safe first-run baseline established without alert spam)")
        print("-" * 60 + "\n")
    elif args.setup_bot:
        run_setup_bot(notifier)
    elif args.test_rss:
        run_test_rss(collector)
    elif args.test_telegram:
        run_test_telegram(notifier)
    elif args.test_gemini:
        run_test_gemini(summarizer)
    else:
        print("\n⚡ AI Pulse Agent (Phase 4)")
        print("=" * 60)
        print("💡 Use --run-cycle          to execute one news collection and alert run.")
        print("💡 Use --poll               to start continuous background monitoring.")
        print("💡 Use --status             to view database article metrics & LLM status.")
        print("💡 Use --run-cycle --dry-run to test without sending Telegram alerts.")
        print("💡 Use --test-gemini        to test Gemini AI summarization / fallback.")
        print("💡 Use --test-telegram      to verify Telegram bot delivery.")
        print("=" * 60 + "\n")
        print_status(db)


if __name__ == "__main__":
    main()




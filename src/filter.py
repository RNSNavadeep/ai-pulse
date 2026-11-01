import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from src.models import Article

logger = logging.getLogger(__name__)

# Curated AI topics and terminology categorized for clear configuration
DEFAULT_AI_TOPICS: Dict[str, List[str]] = {
    "llm_and_models": [
        "llm",
        "llms",
        "large language model",
        "foundation model",
        "open weights",
        "open-weights",
        "fine-tuning",
        "finetuning",
        "pretraining",
        "gpt",
        "claude",
        "gemini",
        "llama",
        "mistral",
        "deepseek",
        "qwen",
    ],
    "agents_and_autonomy": [
        "ai agent",
        "ai agents",
        "agentic",
        "agentic ai",
        "autonomous agent",
        "multi-agent",
        "tool use",
        "function calling",
    ],
    "generative_ai": [
        "generative ai",
        "genai",
        "text-to-image",
        "diffusion model",
        "multimodal",
        "vision-language",
    ],
    "machine_learning_core": [
        "machine learning",
        "deep learning",
        "neural network",
        "transformer architecture",
        "reinforcement learning",
        "rlhf",
    ],
    "rag_and_knowledge": [
        "retrieval-augmented generation",
        "rag",
        "vector database",
        "embeddings",
        "semantic search",
    ],
    "developer_tools_and_coding": [
        "coding assistant",
        "ai coding",
        "copilot",
        "code generation",
        "langchain",
        "llamaindex",
        "vllm",
        "ollama",
        "hugging face",
        "huggingface",
    ],
    "research_and_benchmarks": [
        "benchmark",
        "benchmarks",
        "evals",
        "evaluation",
        "reasoning model",
        "chain-of-thought",
        "scaling laws",
    ],
    "robotics": [
        "robotics",
        "humanoid",
        "embodied ai",
    ],
}


class TopicFilter:
    """Intelligent keyword-based topic filter for AI news articles."""

    def __init__(
        self,
        custom_topics: Optional[Dict[str, List[str]]] = None,
        require_match_for_pure_ai: bool = False,
    ):
        """
        Args:
            custom_topics: Custom dictionary of topic names to keyword lists.
            require_match_for_pure_ai: If False, articles from pure AI sources
                                       pass by default even without explicit keywords.
        """
        self.topics = custom_topics or DEFAULT_AI_TOPICS
        self.require_match_for_pure_ai = require_match_for_pure_ai
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> List[Tuple[str, re.Pattern]]:
        """
        Compile keywords into regular expressions with safe word boundaries.
        Uses \b word boundaries for short acronyms (e.g. 'rag', 'llm') to prevent
        false positives like 'dragon' or 'parallelism'.
        """
        patterns = []
        for category, keywords in self.topics.items():
            for kw in keywords:
                # Use word boundaries for short acronyms / single words
                escaped = re.escape(kw.strip())
                if len(kw.strip()) <= 4 or " " not in kw:
                    pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
                else:
                    pattern = re.compile(escaped, re.IGNORECASE)
                patterns.append((kw.strip(), pattern))
        return patterns

    def score_article(
        self, article: Article, is_pure_ai_source: bool = False
    ) -> Tuple[bool, List[str], float]:
        """
        Evaluate an article for AI relevance.

        Args:
            article: Article to evaluate.
            is_pure_ai_source: True if the source is exclusively an AI publication.

        Returns:
            Tuple of (is_relevant, matched_keywords, relevance_score [0.0 - 1.0]).
        """
        matched_keywords_set: Set[str] = set()
        title_matches = 0
        summary_matches = 0

        title_text = article.title or ""
        summary_text = article.summary or ""

        for kw, pattern in self._compiled_patterns:
            in_title = bool(pattern.search(title_text))
            in_summary = bool(pattern.search(summary_text))

            if in_title:
                matched_keywords_set.add(kw)
                title_matches += 1
            if in_summary:
                matched_keywords_set.add(kw)
                summary_matches += 1

        matched_list = sorted(list(matched_keywords_set))

        # Calculate a normalized relevance score
        # Title matches are weighted heavier than summary matches
        raw_score = (title_matches * 0.3) + (summary_matches * 0.15)

        if is_pure_ai_source:
            # Baseline score for official AI sources even with zero explicit keywords
            base_score = 0.5
            final_score = min(1.0, round(base_score + raw_score, 2))
            is_relevant = True if not self.require_match_for_pure_ai else bool(matched_list)
            if not matched_list:
                matched_list = ["ai announcement"]
            return is_relevant, matched_list, final_score

        # For general tech/industry sources (e.g. TechCrunch, VentureBeat)
        # Must match at least one keyword
        is_relevant = len(matched_list) > 0
        final_score = min(1.0, round(raw_score, 2)) if is_relevant else 0.0

        return is_relevant, matched_list, final_score

    def filter_articles(
        self, articles: List[Article], sources_lookup: Optional[Dict[str, dict]] = None
    ) -> List[Article]:
        """
        Filter and annotate a list of articles with relevance scores and keywords.

        Args:
            articles: List of articles to filter.
            sources_lookup: Optional dict mapping source name -> source config dict.

        Returns:
            List of relevant articles sorted by relevance score descending.
        """
        sources_lookup = sources_lookup or {}
        relevant_articles: List[Article] = []

        for article in articles:
            source_cfg = sources_lookup.get(article.source, {})
            is_pure_ai = source_cfg.get("is_pure_ai", False)

            is_rel, matched_kws, score = self.score_article(
                article, is_pure_ai_source=is_pure_ai
            )

            if is_rel:
                article.matched_keywords = matched_kws
                article.relevance_score = score
                relevant_articles.append(article)
            else:
                logger.debug(
                    "Filtered out non-AI article: '%s' from %s",
                    article.title,
                    article.source,
                )

        # Sort highest relevance score first
        relevant_articles.sort(key=lambda a: a.relevance_score or 0.0, reverse=True)
        logger.info(
            "Topic filter kept %d of %d total articles.",
            len(relevant_articles),
            len(articles),
        )
        return relevant_articles

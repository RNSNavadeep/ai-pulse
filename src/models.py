from typing import List, Optional
from pydantic import BaseModel, Field

VALID_AI_CATEGORIES = [

    "LLMs and model releases",
    "AI agents and agentic AI",
    "Machine learning and deep learning",
    "Generative AI",
    "RAG and vector databases",
    "AI research and benchmarks",
    "AI developer tools and coding assistants",
    "Robotics",
    "Industry announcements",
    "Other AI news",
]


class ArticleAnalysis(BaseModel):
    """Structured response from Gemini API for an AI news item."""

    summary: str = Field(..., description="Concise summary in beginner-friendly English")
    main_development: str = Field(..., description="Core announcement or technical breakthrough")
    why_it_matters: str = Field(..., description="Why this development matters to an AI engineering student")
    suggested_action: Optional[str] = Field(None, description="Suggested next action or practical takeaway")
    category: str = Field("Other AI news", description="One of the predefined AI categories")
    importance_score: int = Field(5, ge=1, le=10, description="Global impact score from 1 to 10")
    relevance_score: int = Field(5, ge=1, le=10, description="Relevance to AI engineering from 1 to 10")
    score_explanation: str = Field("", description="Short justification of the scores")
    is_insufficient_info: bool = Field(False, description="True if RSS content was insufficient to analyze")


class Article(BaseModel):
    """Data model representing a discovered AI news article or paper."""

    id: str = Field(..., description="Unique identifier (URL, GUID, or hash)")
    title: str = Field(..., description="Headline of the article")
    link: str = Field(..., description="Canonical URL to the full article")
    source: str = Field(..., description="Name of the publisher/feed")
    published: Optional[str] = Field(None, description="Publication timestamp string")
    summary: Optional[str] = Field(None, description="Snippet, summary, or abstract")
    category: Optional[str] = Field("general", description="Category tag e.g. research, releases")
    relevance_score: Optional[float] = Field(None, description="Relevance score (0.0 to 1.0)")
    matched_keywords: List[str] = Field(default_factory=list, description="Keywords matched during filtering")

    # Phase 3: Persistence and tracking metadata
    normalized_url: Optional[str] = Field(None, description="Cleaned URL without tracking parameters")
    status: str = Field("discovered", description="Notification status: discovered, pending, sent, failed")
    retry_count: int = Field(0, description="Number of attempted notification retries")
    error_message: Optional[str] = Field(None, description="Last recorded processing error")
    discovered_at: Optional[str] = Field(None, description="ISO timestamp when first discovered")
    updated_at: Optional[str] = Field(None, description="ISO timestamp when last processed")

    # Phase 4: LLM Analysis metadata
    ai_summary: Optional[str] = Field(None, description="LLM-generated student-friendly summary")
    ai_category: Optional[str] = Field(None, description="LLM-assigned category")
    ai_importance: Optional[int] = Field(None, description="LLM importance score (1-10)")
    ai_relevance: Optional[int] = Field(None, description="LLM relevance score (1-10)")
    why_it_matters: Optional[str] = Field(None, description="Why this matters to an AI engineering student")
    suggested_action: Optional[str] = Field(None, description="Suggested next action or practical experiment")
    score_explanation: Optional[str] = Field(None, description="Reasoning for assigned scores")
    summarization_status: str = Field("pending", description="Status: pending, completed, fallback, failed")
    llm_model: Optional[str] = Field(None, description="Identifier of the LLM model that produced the analysis")
    llm_processed_at: Optional[str] = Field(None, description="ISO timestamp when summarized")




"""
Structured representation of the job description's requirements.

This is deliberately NOT "parse the JD text at runtime with an LLM." The JD
is fixed and known at development time, so we encode its actual requirements
- including the explicit disqualifiers and "how to read between the lines"
section - as data here. This is auditable, fast (no parsing cost in the
ranking step), and directly defensible in a Stage 5 interview: every number
below traces to a specific sentence in job_description.md.

If a different JD were swapped in, this module is exactly what would need to
change - everything downstream (scoring.py, etc.) consumes this structured
form, not the raw JD text. That separation is itself part of the design:
recruiters re-use the same ranking engine across many JDs.
"""
from __future__ import annotations
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Target seniority / experience band
# ---------------------------------------------------------------------------
# JD: "Experience Required: 5-9 years... this is a range, not a requirement...
# we'll seriously consider candidates outside the band if other signals are
# strong." So this is a soft band with graceful falloff, not a hard cutoff.
EXPERIENCE_BAND = (5.0, 9.0)
EXPERIENCE_SOFT_MARGIN = 2.5  # years of falloff tolerance outside the band

# ---------------------------------------------------------------------------
# Title / role-function matching
# ---------------------------------------------------------------------------
# Exact or near-exact title matches (highest title-fit tier)
CORE_TITLE_TERMS = [
    "ai engineer", "machine learning engineer", "ml engineer",
    "applied ml engineer", "research engineer",
]

# Adjacent functions that plausibly do the same *work* even without the
# literal title - this is exactly the "Tier 5 plain language" case the JD
# warns we must not miss (e.g. "Search Engineer" at Google doing ranking).
ADJACENT_TITLE_TERMS = [
    "search engineer", "recommendation", "ranking", "nlp engineer",
    "data scientist", "applied scientist", "research scientist",
    "ai specialist", "ai research", "information retrieval",
]

# Titles that are explicitly OFF-target even if skills look AI-ish -
# these correspond to the keyword-stuffer trap category.
OFF_TARGET_TITLES = [
    "hr manager", "content writer", "accountant", "sales executive",
    "customer support", "graphic designer", "marketing manager",
    "business analyst", "operations manager", "project manager",
    "mechanical engineer", "civil engineer",
]

# Seniority modifiers in the title - "senior" alone isn't sufficient (the
# JD is explicit that someone who stopped writing code 18mo ago to become
# "tech lead"/"architect" is a soft disqualifier), but seniority language in
# general supports fit.
SENIOR_TITLE_MARKERS = ["senior", "staff", "lead", "principal"]
ARCHITECTURE_TITLE_MARKERS = ["architect", "tech lead", "engineering manager", "head of"]

# ---------------------------------------------------------------------------
# Required technical surface area
# ---------------------------------------------------------------------------
# "Things you absolutely need" - each maps to a family of acceptable skill
# names/phrases, since the JD explicitly says "we don't care which model/tech,
# we care about the operational experience."
MUST_HAVE_SKILL_FAMILIES = {
    "embeddings_retrieval": [
        "embeddings", "sentence-transformers", "sentence transformers",
        "openai embeddings", "bge", "e5", "dense retrieval", "semantic search",
        "vector search", "retrieval", "information retrieval", "vector representations",
        "text encoders",
        # RAG orchestration frameworks that inherently involve embedding models
        # and retrieval pipelines — not just infrastructure. Haystack (deepset)
        # and LlamaIndex both require selecting, calling, and managing embedding
        # models as a core workflow. Qdrant is intentionally kept out of this
        # family (it is a vector DB / infra layer; using it does not necessarily
        # imply generating embeddings yourself).
        "haystack", "llamaindex",
    ],
    "vector_or_hybrid_search_infra": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
        "elasticsearch", "faiss", "hybrid search", "vector database",
        "vector db", "pgvector", "haystack", "llamaindex", "bm25",
        "search backend", "search infrastructure", "search & discovery",
        "indexing algorithms",
    ],
    "python": ["python"],
    "ranking_evaluation": [
        "ndcg", "mrr", "map", "a/b testing", "ab testing", "offline evaluation",
        "learning to rank", "ranking evaluation", "ir evaluation",
        "ranking systems", "recommendation systems", "content matching",
    ],
}

# "Things we'd like but won't reject for" - bonus, not required
NICE_TO_HAVE_SKILL_FAMILIES = {
    "llm_finetuning": ["lora", "qlora", "peft", "fine-tuning llms", "fine-tuning",
                        "llm fine-tuning"],
    "learning_to_rank_models": ["xgboost", "learning to rank", "neural ranking"],
    "hrtech_background": [],  # detected via industry/company text, not skills
    "distributed_systems": ["distributed systems", "large-scale inference",
                             "kafka", "spark", "kubernetes"],
    "open_source": [],  # detected via github_activity_score
}

# Buzzword-only skills that, when present WITHOUT supporting career-history
# substance, are themselves a weak negative signal (keyword-stuffer pattern).
# These are legitimate skills - the issue is never their presence alone, only
# presence-without-substance, which is why this list feeds the *trust*
# modifier (trust.py) rather than the base relevance score.
TRENDY_BUZZWORDS = [
    "rag", "llm", "pinecone", "langchain", "vector database", "prompt engineering",
    "fine-tuning llms", "transformers", "gpt", "embeddings",
]

# ---------------------------------------------------------------------------
# Disqualifiers ("we will not move forward" / "probably not move forward")
# ---------------------------------------------------------------------------
PURE_CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini",
]

NON_NLP_DEEP_TECH_FIELDS = ["computer vision", "speech recognition", "robotics"]
# Note: these are explicitly fine if there's "significant NLP/IR exposure" too
# (JD: "without significant NLP/IR exposure") - so this disqualifier only
# applies when it's the *primary* and *sole* expertise, checked in scoring.py.

# ---------------------------------------------------------------------------
# Location / logistics
# ---------------------------------------------------------------------------
PREFERRED_LOCATIONS = ["pune", "noida"]
ACCEPTABLE_INDIA_LOCATIONS = ["hyderabad", "pune", "mumbai", "delhi", "noida",
                               "gurugram", "gurgaon", "ncr",
                               "bangalore", "bengaluru"]  # both spellings in active use
TARGET_COUNTRY = "india"
NO_VISA_SPONSORSHIP = True  # outside-India candidates are case-by-case, no sponsorship

# Notice period: "we'd love sub-30-day notice... 30+ day candidates are still
# in scope but the bar gets higher" - soft penalty, not a cutoff.
IDEAL_NOTICE_DAYS = 30
NOTICE_PENALTY_PER_EXTRA_30D = 0.05  # fractional score penalty per 30 extra days

# ---------------------------------------------------------------------------
# "Ideal candidate" composite (read-between-the-lines section)
# ---------------------------------------------------------------------------
IDEAL_TOTAL_EXPERIENCE_BAND = (6.0, 8.0)
IDEAL_APPLIED_ML_YEARS = (4.0, 5.0)  # of which N years in applied ML/AI roles

CAREER_TRAJECTORY_HOPPING_THRESHOLD_MONTHS = 18  # "switching every 1.5 years"
CAREER_TRAJECTORY_HOPPING_MIN_INSTANCES = 3  # pattern, not one data point


@dataclass
class JDRequirements:
    """Bundled accessor so scoring.py imports one object, not 15 constants."""
    experience_band: tuple = EXPERIENCE_BAND
    experience_soft_margin: float = EXPERIENCE_SOFT_MARGIN
    core_title_terms: list = field(default_factory=lambda: CORE_TITLE_TERMS)
    adjacent_title_terms: list = field(default_factory=lambda: ADJACENT_TITLE_TERMS)
    off_target_titles: list = field(default_factory=lambda: OFF_TARGET_TITLES)
    senior_title_markers: list = field(default_factory=lambda: SENIOR_TITLE_MARKERS)
    architecture_title_markers: list = field(default_factory=lambda: ARCHITECTURE_TITLE_MARKERS)
    must_have_skill_families: dict = field(default_factory=lambda: MUST_HAVE_SKILL_FAMILIES)
    nice_to_have_skill_families: dict = field(default_factory=lambda: NICE_TO_HAVE_SKILL_FAMILIES)
    trendy_buzzwords: list = field(default_factory=lambda: TRENDY_BUZZWORDS)
    pure_consulting_firms: list = field(default_factory=lambda: PURE_CONSULTING_FIRMS)
    non_nlp_deep_tech_fields: list = field(default_factory=lambda: NON_NLP_DEEP_TECH_FIELDS)
    preferred_locations: list = field(default_factory=lambda: PREFERRED_LOCATIONS)
    acceptable_india_locations: list = field(default_factory=lambda: ACCEPTABLE_INDIA_LOCATIONS)
    target_country: str = TARGET_COUNTRY
    ideal_notice_days: int = IDEAL_NOTICE_DAYS
    notice_penalty_per_extra_30d: float = NOTICE_PENALTY_PER_EXTRA_30D
    ideal_total_experience_band: tuple = IDEAL_TOTAL_EXPERIENCE_BAND
    career_hopping_threshold_months: int = CAREER_TRAJECTORY_HOPPING_THRESHOLD_MONTHS
    career_hopping_min_instances: int = CAREER_TRAJECTORY_HOPPING_MIN_INSTANCES

    # Free-text requirements summary used for the semantic-similarity layer
    # (semantic.py). This is the JD's substance translated into a dense
    # paragraph of the *work*, deliberately phrased to match how a strong
    # candidate's career_history descriptions would read - i.e. function over
    # buzzword, matching the JD's own framing - so a Tier-5 plain-language
    # candidate scores well on semantic similarity even without using AI
    # jargon themselves.
    semantic_profile_text: str = (
        "Own the intelligence layer of a recruiting product: ranking, retrieval, "
        "and matching systems that decide what recruiters see when they search "
        "for candidates. Audit and improve an existing BM25 plus rule-based "
        "scoring system. Ship a v2 ranking system using embeddings, hybrid "
        "retrieval, and LLM-based re-ranking, deployed to real users at "
        "meaningful scale, with demonstrable improvement to recruiter "
        "engagement metrics. Build evaluation infrastructure: offline "
        "benchmarks, online A/B testing, recruiter feedback loops. Production "
        "experience handling embedding drift, index refresh, and retrieval "
        "quality regression. Operational experience with vector databases or "
        "hybrid search infrastructure. Hands-on design of evaluation "
        "frameworks for ranking systems using NDCG, MRR, MAP, and "
        "offline-to-online correlation. Strong Python and systems thinking, "
        "not just framework usage. Has shipped an end-to-end ranking, search, "
        "or recommendation system to real users at meaningful scale."
    )


JD = JDRequirements()

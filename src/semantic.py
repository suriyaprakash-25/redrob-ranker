"""
Semantic similarity layer.

Purpose: catch candidates whose CAREER HISTORY shows real substance matching
the JD's requirements, even when their listed skills don't use the JD's
exact vocabulary. This is the direct countermeasure to the "Tier 5
plain-language" trap and the paraphrased-skill-vocabulary pattern found
during validation (see VALIDATION_LOG.md) - structured scoring alone
under-ranks these candidates because it does exact/substring matching on a
fixed skill-family vocabulary; this layer instead compares the *meaning* of
free text.

Design choice: TF-IDF + truncated SVD (LSA), not a pretrained transformer
embedding model. Why:
  - Zero external downloads. Builds and runs anywhere `scikit-learn` is
    installed - no HuggingFace weights, no internet access required during
    either pre-computation or ranking. This matters because the hackathon's
    own sandboxed reproduction environment may not have network access even
    during pre-computation, and definitely not during the timed ranking step.
  - Fully deterministic and auditable: the vocabulary and component loadings
    can be inspected directly, which is a real advantage when defending this
    in the Stage 5 interview ("show me why this candidate scored 0.81") -
    you can point at actual shared n-grams, not just a black-box cosine
    score from an opaque 400M-parameter model.
  - The corpus here (career_history descriptions + JD requirements text) is
    domain-specific tech/recruiting vocabulary with meaningful term overlap
    between genuine fits and the JD - exactly the regime where TF-IDF does
    well, since the "signal" is largely lexical-but-paraphrased (synonyms,
    not deep semantic restructuring).

Optional upgrade path (documented in README, not required to run): swap in
sentence-transformers/BGE embeddings computed OFFLINE on a machine with
internet access, cache to the same artifact format consumed here. The
`encode_corpus` / `similarity_to_jd` interface is intentionally agnostic to
which vector backend produced the embeddings, so this is a drop-in swap, not
a rewrite. See `embeddings_upgrade.py` for the optional sentence-transformers
implementation of the same interface.
"""
from __future__ import annotations
from dataclasses import dataclass
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from jd_requirements import JD

N_SVD_COMPONENTS = 128
MIN_DF = 3            # ignore terms appearing in fewer than 3 candidate documents (noise)
MAX_DF = 0.6          # ignore terms appearing in >60% of documents (uninformative boilerplate)
NGRAM_RANGE = (1, 2)  # unigrams + bigrams catch phrases like "vector search", "production deployment"


def _candidate_text(candidate: dict) -> str:
    """
    Build the free-text document for a candidate. Deliberately uses
    career_history descriptions + profile summary/headline - NOT the
    structured skills list - since the skills list is exactly what's
    already covered (and gamed/paraphrased) in scoring.py. This text
    captures what the candidate actually *did*, in their own words.
    """
    parts = [
        candidate["profile"].get("headline", ""),
        candidate["profile"].get("summary", ""),
    ]
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    return " ".join(parts)


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class SemanticIndex:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    candidate_ids: list
    candidate_vectors: np.ndarray  # (n_candidates, N_SVD_COMPONENTS), L2-normalized
    jd_vector: np.ndarray          # (N_SVD_COMPONENTS,), L2-normalized


def build_semantic_index(candidates: list[dict]) -> SemanticIndex:
    """
    Offline pre-computation step. For 100K candidates this is the expensive
    part (fit TF-IDF + SVD), but it's a ONE-TIME cost done before the timed
    ranking step - the ranking step only does a cosine-similarity lookup
    against this pre-built index, which is fast (a few seconds for 100K
    rows of dense 128-dim vectors).
    """
    docs = [_clean(_candidate_text(c)) for c in candidates]
    jd_doc = _clean(JD.semantic_profile_text)

    vectorizer = TfidfVectorizer(
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        max_features=20000,
    )
    # Fit on candidate docs + the JD doc together so the JD's vocabulary is
    # guaranteed to be represented in the fitted vocabulary even if some of
    # its terms are rare across candidates.
    tfidf_matrix = vectorizer.fit_transform(docs + [jd_doc])

    n_components = min(N_SVD_COMPONENTS, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(tfidf_matrix)
    reduced = normalize(reduced, axis=1)

    candidate_vectors = reduced[:-1]
    jd_vector = reduced[-1]

    return SemanticIndex(
        vectorizer=vectorizer,
        svd=svd,
        candidate_ids=[c["candidate_id"] for c in candidates],
        candidate_vectors=candidate_vectors,
        jd_vector=jd_vector,
    )


def semantic_scores(index: SemanticIndex) -> dict[str, float]:
    """
    Cosine similarity (vectors are already L2-normalized, so this is just a
    dot product) between every candidate and the JD's semantic profile.
    Returned as a dict candidate_id -> raw cosine similarity in [-1, 1]
    (in practice these are all positive for this corpus, but we don't clip
    here - normalization to [0,1] for blending happens in final.py so each
    layer's raw output stays inspectable).
    """
    sims = index.candidate_vectors @ index.jd_vector
    return dict(zip(index.candidate_ids, sims.tolist()))

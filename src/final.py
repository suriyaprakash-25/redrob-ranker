"""
Final score combination.

Combines the four independent lenses into one ranking score:
  - structured_score   (scoring.py): does the structured profile match the
                         JD's explicit requirements (title, experience,
                         skills, location, disqualifiers)?
  - semantic_score      (semantic.py): does the candidate's own description
                         of their work resemble what the JD actually needs,
                         independent of exact skill-name vocabulary?
  - trust_multiplier    (trust.py): should we believe the claimed skills?
  - availability_mult   (behavioral.py): can we actually reach/hire them
                         right now?
  - honeypot_score      (honeypot.py): is this profile internally
                         consistent at all?

Combination design:
  - honeypot_score is a HARD GATE, not a blended input - validated against
    the spec's explicit rule (>10% honeypot rate in top 100 = Stage 3
    disqualification regardless of composite score). A candidate with
    honeypot_score >= 0.5 is forced to the bottom of the ranking,
    full stop, mirroring the ground truth's "forced to relevance tier 0."
  - structured_score and semantic_score are blended additively (weighted
    average) since they're complementary, not contradictory, views of the
    same underlying "JD fit" question - exactly the gap each one is
    designed to cover for the other (structured catches exact
    keyword/title signal fast; semantic catches paraphrased substance).
  - trust_multiplier and availability_multiplier are applied
    MULTIPLICATIVELY on top, not blended in - they represent "how much do
    we discount an otherwise-good fit score," not "fit" itself. A
    candidate with perfect fit but 0 trust should end up near 0, not
    averaged down to ~50%, which a weighted blend would do.

Weighting between structured (0.55) and semantic (0.45) reflects that
structured scoring directly encodes the JD's EXPLICIT, hard-stated rules
(title, experience band, disqualifiers) which the JD says are non-
negotiable in several places ("we will not move forward"), while semantic
similarity is the safety net for paraphrase/vocabulary gaps - important,
but secondary to explicit JD rules when they're both available and don't
conflict.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from scoring import score_structured
from trust import evaluate_trust
from behavioral import evaluate_behavioral
from honeypot import evaluate_honeypot
from jd_requirements import JD

STRUCTURED_WEIGHT = 0.55
SEMANTIC_WEIGHT = 0.45
HONEYPOT_GATE_THRESHOLD = 0.5


@dataclass
class FinalScoreResult:
    candidate_id: str
    final_score: float
    is_honeypot: bool
    structured_score: float
    semantic_score: float
    trust_multiplier: float
    availability_multiplier: float
    honeypot_score: float
    structured_notes: list = field(default_factory=list)
    trust_notes: list = field(default_factory=list)
    behavioral_notes: list = field(default_factory=list)
    honeypot_notes: list = field(default_factory=list)


def _normalize_semantic(raw_cosine: float, lo: float = 0.3, hi: float = 0.95) -> float:
    """
    Raw cosine similarities from the TF-IDF/SVD space cluster in a narrow
    band (empirically ~0.3-0.95 on this corpus rather than the full [-1,1]
    range) because every candidate document shares a lot of generic
    tech/career vocabulary with the JD profile text. Rescale to [0,1] using
    the empirical band so the semantic score actually spreads out and
    contributes meaningfully to the blend, rather than every candidate
    landing in a compressed 0.3-0.95 sliver that the structured score would
    dominate by default.
    """
    return max(0.0, min(1.0, (raw_cosine - lo) / (hi - lo)))


def combine_scores(
    candidate: dict,
    semantic_raw_cosine: float,
) -> FinalScoreResult:
    structured = score_structured(candidate)
    trust = evaluate_trust(candidate, JD.trendy_buzzwords)
    behavioral = evaluate_behavioral(candidate)
    honeypot = evaluate_honeypot(candidate)

    semantic_norm = _normalize_semantic(semantic_raw_cosine)

    fit_score = STRUCTURED_WEIGHT * structured.structured_score + SEMANTIC_WEIGHT * semantic_norm
    final = fit_score * trust.trust_multiplier * behavioral.availability_multiplier

    is_hp = honeypot.honeypot_score >= HONEYPOT_GATE_THRESHOLD
    if is_hp:
        # Hard gate: forced below every non-honeypot candidate regardless of
        # how good the rest of the profile looks, matching ground truth's
        # "forced to relevance tier 0." We keep a tiny ordering signal
        # (1 - honeypot_score, scaled into a sub-zero range) purely so
        # multiple honeypots don't all tie at literally 0 -- score ties are
        # allowed by the spec but we'd rather keep a stable, explicable order.
        final = -1.0 + (1.0 - honeypot.honeypot_score) * 0.01

    return FinalScoreResult(
        candidate_id=candidate["candidate_id"],
        final_score=final,
        is_honeypot=is_hp,
        structured_score=structured.structured_score,
        semantic_score=semantic_norm,
        trust_multiplier=trust.trust_multiplier,
        availability_multiplier=behavioral.availability_multiplier,
        honeypot_score=honeypot.honeypot_score,
        structured_notes=structured.notes,
        trust_notes=trust.notes,
        behavioral_notes=behavioral.notes,
        honeypot_notes=honeypot.reasons,
    )

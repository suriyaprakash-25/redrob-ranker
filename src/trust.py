"""
Verified-vs-claimed trust modifier.

Directly operationalizes the JD's framing: "A candidate who has all the AI
keywords listed as skills but whose title is 'Marketing Manager' is not a
fit, no matter how perfect their skill list looks." More generally, ANY
candidate's self-reported skill list can be inflated; this module asks
"how much should we actually believe this skill list?" using signals that
are expensive/impossible to fake cheaply:

  1. skill_assessment_scores: a tested score, independent of self-report.
     Validated at full-dataset scale (see VALIDATION_LOG.md / earlier
     analysis): 34.3% of self-reported proficiency claims diverge from the
     tested score by more than 30 points - this gap is common enough to be
     a real, generalizable signal, not noise.
  2. github_activity_score: real, externally-observable contribution signal
     (-1 if absent, which we treat as neutral/unknown, not penalized -
     plenty of strong engineers don't have public GitHub activity tied to
     proprietary work).
  3. endorsements vs duration_months: someone "expert" in a skill for 80
     months with 0 endorsements is a softer flag than someone claiming the
     same with thin usage; we use this as a secondary, low-weight signal
     since endorsement counts on a synthetic platform are a weaker proxy
     than the tested assessment score.
  4. Raw count of AI/ML buzzwords as listed SKILLS, with no corroborating
     career_history substance - this is the literal keyword-stuffer pattern
     confirmed at scale: 2,719 candidates with off-target titles (HR
     Manager, Content Writer, etc.) list 5+ ML-relevant skills.

Output: a `trust_multiplier` in roughly [0.5, 1.05] applied multiplicatively
to the combined structured+semantic score in final.py. We deliberately keep
this as a multiplier (not an additive penalty) so it scales proportionally -
a borderline candidate with inflated skills gets knocked down more in
absolute terms than a clearly-strong candidate with the same inflation
pattern, which matches how a real recruiter would weigh "I have some doubts
about your claimed skills" against an otherwise excellent profile.
"""
from __future__ import annotations
from dataclasses import dataclass, field

PROFICIENCY_EXPECTED_SCORE = {
    "beginner": 25, "intermediate": 50, "advanced": 75, "expert": 95,
}


@dataclass
class TrustResult:
    candidate_id: str
    trust_multiplier: float
    notes: list = field(default_factory=list)


def _claimed_vs_tested_gap(candidate: dict) -> tuple[float, list[str]]:
    """
    Returns a penalty fraction in [0, 0.35] based on how much the
    candidate's self-reported proficiency overstates their tested
    assessment scores, averaged across skills that have both.
    """
    scores = candidate["redrob_signals"].get("skill_assessment_scores", {})
    if not scores:
        return 0.0, ["no skill assessment data available to verify claims"]

    skill_map = {s["name"]: s for s in candidate.get("skills", [])}
    gaps = []
    overstated_examples = []

    for skill_name, tested in scores.items():
        if skill_name not in skill_map:
            continue
        prof = skill_map[skill_name]["proficiency"]
        expected = PROFICIENCY_EXPECTED_SCORE[prof]
        gap = expected - tested  # positive = overstated
        gaps.append(gap)
        if gap > 30:
            overstated_examples.append(f"{skill_name} (claimed {prof}, tested {tested:.0f}/100)")

    if not gaps:
        return 0.0, ["no overlapping assessed skills to verify claims"]

    avg_gap = sum(gaps) / len(gaps)
    # avg_gap of 0 = perfectly calibrated; 40+ = systematically overstated.
    penalty = max(0.0, min(0.35, avg_gap / 40.0 * 0.35))

    notes = []
    if penalty > 0.08:
        notes.append(
            f"self-reported skill levels run ahead of tested assessment scores "
            f"(avg gap {avg_gap:.0f}pts" + (f"; e.g. {', '.join(overstated_examples[:2])}" if overstated_examples else "") + ")"
        )
    elif avg_gap < -10:
        notes.append(f"tested assessment scores exceed self-reported proficiency (avg +{-avg_gap:.0f}pts) - likely understating own skill")

    return penalty, notes


def _keyword_stuffing_without_substance(candidate: dict, trendy_buzzwords: list[str]) -> tuple[float, list[str]]:
    """
    Counts trendy AI/ML buzzwords present as SKILLS but absent from the
    candidate's own career_history descriptions (i.e. claimed but never
    actually described as work done). A handful of such gaps is normal
    (people list aspirational/learning skills, and career-history text is
    free-form prose that won't always restate every listed skill verbatim);
    a large count combined with an off-target title is the keyword-stuffer
    signature confirmed at scale in the dataset (2,719 such candidates
    found during validation, almost all off-target-titled with 4+
    unsubstantiated buzzwords).

    Minimum-sample guard: with only 1-2 matched buzzwords, a single
    unsubstantiated term produces a 100% "unsupported ratio" by pure
    arithmetic, which incorrectly fired at max penalty on a validated gold
    candidate (CAND_0006567, one buzzword - "Prompt Engineering" - not
    restated verbatim in paraphrased career-history prose). Require at
    least MIN_BUZZWORDS_FOR_PENALTY matched terms before this check can
    apply any penalty at all; below that, a single non-restated skill is
    noise, not signal.
    """
    MIN_BUZZWORDS_FOR_PENALTY = 4
    NOTE_RATIO_THRESHOLD = 0.6

    skill_names = {s["name"].lower() for s in candidate.get("skills", [])}
    matched_buzzwords = [b for b in trendy_buzzwords if b in skill_names]
    if len(matched_buzzwords) < MIN_BUZZWORDS_FOR_PENALTY:
        return 0.0, []

    history_blob = " ".join(
        r.get("description", "").lower() for r in candidate.get("career_history", [])
    )
    unsupported = [b for b in matched_buzzwords if b not in history_blob]

    if not unsupported:
        return 0.0, []

    ratio = len(unsupported) / len(matched_buzzwords)
    if ratio <= NOTE_RATIO_THRESHOLD:
        return 0.0, []

    penalty = min(0.25, ratio * 0.25)
    notes = [
        f"lists {len(matched_buzzwords)} AI/ML-trendy skills but career history doesn't "
        f"substantiate {len(unsupported)} of them - possible keyword stuffing"
    ]
    return penalty, notes


def _github_signal(candidate: dict) -> tuple[float, list[str]]:
    """Small bonus for verified external activity; neutral (not penalized) if absent."""
    gh = candidate["redrob_signals"].get("github_activity_score", -1)
    if gh is None or gh < 0:
        return 0.0, []
    if gh >= 60:
        return 0.05, [f"strong verified GitHub activity (score {gh:.0f}/100) corroborates claimed skills"]
    return 0.0, []


def evaluate_trust(candidate: dict, trendy_buzzwords: list[str]) -> TrustResult:
    gap_penalty, n1 = _claimed_vs_tested_gap(candidate)
    stuffing_penalty, n2 = _keyword_stuffing_without_substance(candidate, trendy_buzzwords)
    github_bonus, n3 = _github_signal(candidate)

    total_penalty = min(0.5, gap_penalty + stuffing_penalty)
    multiplier = max(0.5, 1.0 - total_penalty) + github_bonus
    multiplier = min(1.05, multiplier)

    return TrustResult(
        candidate_id=candidate["candidate_id"],
        trust_multiplier=multiplier,
        notes=n1 + n2 + n3,
    )

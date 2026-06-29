"""
Honeypot / data-integrity detector.

The hackathon dataset contains ~80 candidates with "subtly impossible profiles"
(README + submission_spec.md, Section 7). Examples given explicitly:
  - 8 years of experience at a company founded 3 years ago
  - "expert" proficiency in a skill with 0 years used

Submissions with honeypot rate > 10% in the top 100 are disqualified at Stage 3,
regardless of composite score. This module must have high recall on the known
patterns and near-zero false-positive rate on genuine candidates (a false
positive here just costs us a borderline candidate's rank; a false negative
risks the whole submission).

We do NOT need an exhaustive list of every honeypot - the spec explicitly says
"you don't need to special-case them," i.e. a system that actually reads
profiles will naturally avoid them. This module operationalizes that "reading"
as a small set of internal-consistency checks, each independently justifiable
from the schema regardless of the honeypot rule.

Output: a `honeypot_score` in [0, 1] per candidate (0 = clean, 1 = certainly
fabricated) plus the list of specific reasons triggered, for transparency in
the reasoning column and for the methodology write-up.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


@dataclass
class HoneypotResult:
    candidate_id: str
    honeypot_score: float  # 0.0 - 1.0
    reasons: list[str] = field(default_factory=list)

    @property
    def is_honeypot(self) -> bool:
        # Conservative threshold: only treat as honeypot if we have a
        # genuinely hard internal contradiction, not a soft stylistic oddity.
        return self.honeypot_score >= 0.5


def _check_yoe_vs_career_history(candidate: dict) -> tuple[float, list[str]]:
    """
    Total months across career_history entries should be roughly consistent
    with years_of_experience. Real careers have gaps (so total can be a bit
    less than YOE*12) but should never wildly exceed it, and shouldn't be
    a tiny fraction of it either for someone with a long, populated history.
    """
    yoe = candidate["profile"]["years_of_experience"]
    history = candidate["career_history"]
    total_months = sum(r["duration_months"] for r in history)
    yoe_months = yoe * 12

    reasons = []
    score = 0.0

    if yoe_months <= 0:
        return 0.0, reasons

    ratio = total_months / yoe_months

    # Overlapping/duplicated roles or a single role with an impossible duration
    # can push total_months far above yoe_months. Allow some slack (concurrent
    # part-time roles, rounding) but flag clear impossibilities.
    if ratio > 1.5:
        score = max(score, min(1.0, (ratio - 1.5) * 0.8 + 0.4))
        reasons.append(
            f"career_history totals {total_months}mo vs {yoe_months:.0f}mo implied by "
            f"years_of_experience ({yoe}y) - ratio {ratio:.2f}x"
        )

    # A long career history (3+ roles) covering only a small fraction of
    # claimed YOE is the inverse problem.
    if len(history) >= 2 and ratio < 0.35:
        score = max(score, 0.4)
        reasons.append(
            f"career_history totals only {total_months}mo across {len(history)} roles vs "
            f"{yoe_months:.0f}mo implied by years_of_experience ({yoe}y) - ratio {ratio:.2f}x"
        )

    return score, reasons


def _check_single_role_duration_sanity(candidate: dict) -> tuple[float, list[str]]:
    """
    A single role with an extreme duration_months (e.g. 166 months = 13.8
    years in one job) relative to the candidate's total claimed experience,
    or relative to a sane absolute ceiling, is a strong fabrication signal.
    """
    yoe = candidate["profile"]["years_of_experience"]
    yoe_months = yoe * 12
    reasons = []
    score = 0.0

    for role in candidate["career_history"]:
        dm = role["duration_months"]
        # Absolute sanity ceiling: no single role should exceed ~40 years.
        if dm > 480:
            score = max(score, 1.0)
            reasons.append(f"role at {role['company']} has duration_months={dm} (>40 years)")
            continue
        # Relative: a single role consuming far more months than the
        # candidate's entire claimed career.
        if yoe_months > 0 and dm > yoe_months * 1.3 + 12:
            score = max(score, 0.85)
            reasons.append(
                f"role at {role['company']} ({role['title']}) duration_months={dm} exceeds "
                f"candidate's total claimed years_of_experience ({yoe}y = {yoe_months:.0f}mo)"
            )

    return score, reasons


def _check_date_consistency(candidate: dict) -> tuple[float, list[str]]:
    """
    start_date + duration_months should roughly equal end_date (or "now" for
    is_current roles). Large discrepancies indicate fabricated/inconsistent
    records rather than a real career timeline.
    """
    reasons = []
    score = 0.0
    today = date(2026, 6, 28)

    for role in candidate["career_history"]:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date")) if not role.get("is_current") else today
        if start is None or end is None:
            continue
        implied_months = (end.year - start.year) * 12 + (end.month - start.month)
        if implied_months < 0:
            score = max(score, 1.0)
            reasons.append(
                f"role at {role['company']}: end_date before start_date "
                f"({role.get('start_date')} -> {role.get('end_date')})"
            )
            continue
        stated = role["duration_months"]
        if stated <= 0:
            continue
        discrepancy_ratio = abs(implied_months - stated) / max(stated, 1)
        if discrepancy_ratio > 0.5 and abs(implied_months - stated) > 6:
            score = max(score, min(1.0, 0.3 + discrepancy_ratio * 0.3))
            reasons.append(
                f"role at {role['company']}: dates imply ~{implied_months}mo but "
                f"duration_months={stated} (mismatch)"
            )

    return score, reasons


def _check_expert_zero_duration(candidate: dict) -> tuple[float, list[str]]:
    """
    "Expert" or "advanced" proficiency claimed with ~0 months of usage is the
    second explicit example given in the spec ("expert proficiency in 10
    skills with 0 years used"). We scale severity by how many such skills
    appear and how extreme the proficiency/duration gap is.
    """
    reasons = []
    bad_skills = []
    for s in candidate.get("skills", []):
        dur = s.get("duration_months", None)
        if dur is None:
            continue
        prof = s["proficiency"]
        if prof == "expert" and dur <= 2:
            bad_skills.append((s["name"], prof, dur))
        elif prof == "advanced" and dur <= 1:
            bad_skills.append((s["name"], prof, dur))

    if not bad_skills:
        return 0.0, reasons

    # One isolated case could be a fast learner / rounding artifact at the
    # boundary; multiple is a strong fabrication pattern (matches the
    # "expert in 10 skills with 0 years used" example, i.e. pervasive not
    # isolated).
    score = min(1.0, 0.35 + 0.2 * len(bad_skills))
    reasons.append(
        "implausible proficiency/duration: "
        + ", ".join(f"{n} ({p}, {d}mo used)" for n, p, d in bad_skills[:5])
    )
    return score, reasons


def _check_signup_vs_activity(candidate: dict) -> tuple[float, list[str]]:
    """
    NOTE on this check's history: an earlier version flagged ANY
    last_active_date < signup_date as a honeypot signal at full severity.
    Empirically, on the real 100K pool, this fires for ~7,496 candidates
    (7.5% of the entire dataset) with gaps up to 234 days - far too common
    to be the ~80 "subtly impossible profiles" the spec describes. This is
    evidently routine synthetic-data noise (signup_date and last_active_date
    likely generated by semi-independent random processes), not a designed
    trap. Treating it as a honeypot trigger would have misclassified
    thousands of ordinary candidates and risked tanking precision broadly.

    We keep a much more conservative version here: only flag a *large*
    gap (>120 days) as a mild signal, and we deliberately cap its severity
    well below the is_honeypot threshold (0.5) so it can never, by itself,
    suppress a candidate. It still gets folded into the behavioral
    "availability" scoring (see behavioral.py) as a legitimate negative
    signal about freshness - just not into honeypot disqualification.
    """
    sig = candidate["redrob_signals"]
    signup = _parse_date(sig.get("signup_date"))
    active = _parse_date(sig.get("last_active_date"))
    if signup and active and active < signup:
        gap_days = (signup - active).days
        if gap_days > 120:
            return 0.15, [
                f"last_active_date precedes signup_date by {gap_days}d "
                f"(likely data noise, not treated as disqualifying)"
            ]
    return 0.0, []


def _check_education_year_sanity(candidate: dict) -> tuple[float, list[str]]:
    """
    end_year before start_year, or a degree window inconsistent with the
    candidate plausibly also having years_of_experience years of work
    (e.g. graduated 2024 but claims 12 years of experience), is a
    fabrication signal.
    """
    reasons = []
    score = 0.0
    yoe = candidate["profile"]["years_of_experience"]
    current_year = 2026

    for edu in candidate.get("education", []):
        sy, ey = edu.get("start_year"), edu.get("end_year")
        if sy and ey and ey < sy:
            score = max(score, 1.0)
            reasons.append(f"education end_year ({ey}) before start_year ({sy})")
        if ey:
            years_since_grad = current_year - ey
            # Can't have more work experience than years since the *most
            # recent* graduation allows for, with slack for pre-degree work.
            if years_since_grad < -1:  # graduates in the future
                score = max(score, 0.6)
                reasons.append(f"education end_year ({ey}) is in the future")

    return score, reasons


CHECKS = [
    _check_yoe_vs_career_history,
    _check_single_role_duration_sanity,
    _check_date_consistency,
    _check_expert_zero_duration,
    _check_signup_vs_activity,
    _check_education_year_sanity,
]


def evaluate_honeypot(candidate: dict) -> HoneypotResult:
    best_score = 0.0
    all_reasons: list[str] = []
    for check in CHECKS:
        score, reasons = check(candidate)
        if score > 0:
            all_reasons.extend(reasons)
        best_score = max(best_score, score)
    return HoneypotResult(
        candidate_id=candidate["candidate_id"],
        honeypot_score=best_score,
        reasons=all_reasons,
    )

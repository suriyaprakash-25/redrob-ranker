"""
Behavioral availability modifier.

Direct implementation of the JD's explicit instruction: "a perfect-on-paper
candidate who hasn't logged in for 6 months and has a 5% recruiter response
rate is, for hiring purposes, not actually available. Down-weight them
appropriately."

This is intentionally scored as a SEPARATE multiplier from trust.py, even
though both consume redrob_signals, because they answer different
questions:
  - trust.py: "should we believe what this profile claims about skills?"
  - behavioral.py: "even if everything is true, can we actually reach and
    hire this person right now?"

A candidate can be 100% truthful and skilled but completely unreachable
(low multiplier here, high in trust.py), or reachable but skill-inflated
(opposite pattern). Conflating them would hide which problem a low score
is pointing at - bad for the reasoning column, bad for defending design
choices at Stage 5.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime

REFERENCE_DATE = date(2026, 6, 28)  # dataset's "now," used for recency calcs


def _days_since(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (REFERENCE_DATE - d).days


@dataclass
class BehavioralResult:
    candidate_id: str
    availability_multiplier: float
    notes: list = field(default_factory=list)


def _recency_factor(sig: dict) -> tuple[float, list[str]]:
    days = _days_since(sig.get("last_active_date"))
    if days is None:
        return 1.0, []
    if days <= 30:
        return 1.0, []
    if days <= 90:
        return 0.9, [f"last active {days}d ago (1-3 months)"]
    if days <= 180:
        return 0.7, [f"last active {days}d ago (3-6 months) - engagement may have cooled"]
    return 0.45, [f"last active {days}d ago (6+ months) - likely not actively job-seeking despite profile"]


def _response_rate_factor(sig: dict) -> tuple[float, list[str]]:
    rate = sig.get("recruiter_response_rate", 0.5)
    if rate >= 0.5:
        return 1.0, []
    if rate >= 0.25:
        return 0.85, [f"recruiter response rate {rate:.0%} is below average"]
    return 0.6, [f"recruiter response rate {rate:.0%} is low - may be hard to actually reach"]


def _open_to_work_factor(sig: dict) -> tuple[float, list[str]]:
    if sig.get("open_to_work_flag"):
        return 1.05, ["marked open to work"]
    if sig.get("applications_submitted_30d", 0) > 0:
        return 1.0, []
    return 0.92, []


def _interview_reliability_factor(sig: dict) -> tuple[float, list[str]]:
    rate = sig.get("interview_completion_rate", 1.0)
    if rate >= 0.7:
        return 1.0, []
    if rate >= 0.4:
        return 0.9, [f"interview completion rate {rate:.0%} - some history of no-shows/drop-off"]
    return 0.75, [f"interview completion rate {rate:.0%} is low - meaningful risk of drop-off in process"]


def _offer_acceptance_factor(sig: dict) -> tuple[float, list[str]]:
    rate = sig.get("offer_acceptance_rate", -1)
    if rate is None or rate < 0:
        return 1.0, []  # no offer history - neutral, not penalized
    if rate >= 0.5:
        return 1.0, []
    return 0.9, [f"offer acceptance rate {rate:.0%} historically - may be evaluating multiple options or hard to close"]


def _verification_factor(sig: dict) -> tuple[float, list[str]]:
    if sig.get("verified_email") and sig.get("verified_phone"):
        return 1.0, []
    missing = []
    if not sig.get("verified_email"):
        missing.append("email")
    if not sig.get("verified_phone"):
        missing.append("phone")
    return 0.95, [f"unverified contact info ({', '.join(missing)})"] if missing else (1.0, [])


def _stale_signup_active_factor(sig: dict) -> tuple[float, list[str]]:
    """
    Soft signal carried over from honeypot.py's investigation: a large
    last_active < signup gap is common dataset noise (affects 7.5% of all
    candidates - NOT a fabrication signal at that frequency), but it's
    still legitimately informative here as a freshness/availability
    signal, just at low weight.
    """
    signup = sig.get("signup_date")
    active = sig.get("last_active_date")
    if not signup or not active:
        return 1.0, []
    try:
        s = datetime.strptime(signup, "%Y-%m-%d").date()
        a = datetime.strptime(active, "%Y-%m-%d").date()
    except ValueError:
        return 1.0, []
    if a < s and (s - a).days > 120:
        return 0.95, []
    return 1.0, []


def evaluate_behavioral(candidate: dict) -> BehavioralResult:
    sig = candidate["redrob_signals"]
    factors_and_notes = [
        _recency_factor(sig),
        _response_rate_factor(sig),
        _open_to_work_factor(sig),
        _interview_reliability_factor(sig),
        _offer_acceptance_factor(sig),
        _verification_factor(sig),
        _stale_signup_active_factor(sig),
    ]
    multiplier = 1.0
    notes = []
    for factor, note_list in factors_and_notes:
        multiplier *= factor
        notes.extend(note_list)

    multiplier = max(0.3, min(1.08, multiplier))

    return BehavioralResult(
        candidate_id=candidate["candidate_id"],
        availability_multiplier=multiplier,
        notes=notes,
    )

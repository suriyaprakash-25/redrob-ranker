"""
Structured JD-fit scoring.

This module computes the "does this candidate's *structured* profile match
what the JD actually asks for" score - title/function, experience band,
required skill families, disqualifiers, location/logistics, and the
"ideal candidate" composite from the JD's "how to read between the lines"
section.

Deliberately separate from:
  - semantic.py (free-text similarity on career_history descriptions - catches
    candidates who did the work without using the JD's exact words)
  - trust.py (claimed-vs-verified skill discounting, behavioral availability)
  - honeypot.py (data-integrity / fabrication detection)

Each of those is an independent lens; final.py combines them. Keeping them
separate means each one is auditable and individually testable against the
real data, and it's a much better story in the Stage 5 interview than one
monolithic black-box score.

All scores in this module are in [0, 1] unless noted otherwise.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date

from jd_requirements import JD


@dataclass
class StructuredScoreResult:
    candidate_id: str
    title_fit: float
    experience_fit: float
    must_have_skill_coverage: float
    nice_to_have_bonus: float
    disqualifier_penalty: float  # 0 = no penalty, up to 1 = hard disqualify
    location_fit: float
    notice_fit: float
    ideal_candidate_bonus: float
    structured_score: float  # combined, in [0, 1]
    notes: list[str] = field(default_factory=list)


def _title_fit(candidate: dict) -> tuple[float, list[str]]:
    title = candidate["profile"]["current_title"].lower()
    notes = []

    if any(t in title for t in JD.off_target_titles):
        return 0.05, [f"current title '{candidate['profile']['current_title']}' is off-target for an AI engineering role"]

    core_hit = any(t in title for t in JD.core_title_terms)
    adjacent_hit = any(t in title for t in JD.adjacent_title_terms)

    base = 0.95 if core_hit else (0.7 if adjacent_hit else 0.25)

    # Seniority language is a mild positive, but "architect"/"tech lead" is a
    # JD-explicit soft disqualifier (checked again in _disqualifiers, this
    # is just the title-level partial signal).
    if any(m in title for m in JD.senior_title_markers):
        base = min(1.0, base + 0.05)
        notes.append("title carries senior-level language")
    if any(m in title for m in JD.architecture_title_markers):
        base = max(0.0, base - 0.15)
        notes.append("title suggests architecture/management track, not hands-on coding")

    if core_hit:
        notes.append(f"title '{candidate['profile']['current_title']}' is a direct match for the target role")
    elif adjacent_hit:
        notes.append(f"title '{candidate['profile']['current_title']}' is functionally adjacent (search/ranking/recsys/NLP/DS)")
    else:
        notes.append(f"title '{candidate['profile']['current_title']}' has no clear AI/ML/search function signal")

    return base, notes


def _experience_fit(candidate: dict) -> tuple[float, list[str]]:
    yoe = candidate["profile"]["years_of_experience"]
    lo, hi = JD.experience_band
    margin = JD.experience_soft_margin

    if lo <= yoe <= hi:
        return 1.0, [f"{yoe}y experience is within the target 5-9y band"]

    if yoe < lo:
        gap = lo - yoe
    else:
        gap = yoe - hi

    # Graceful falloff per the JD's own framing ("we'll seriously consider
    # candidates outside the band if other signals are strong") - linear
    # decay to 0 at margin, not a cliff.
    score = max(0.0, 1.0 - gap / margin)
    direction = "below" if yoe < lo else "above"
    return score, [f"{yoe}y experience is {direction} the 5-9y target band by {gap:.1f}y"]


def _skill_family_hit(skill_names: set[str], family_terms: list[str]) -> bool:
    return any(term in skill_names for term in family_terms) or any(
        any(term in sn for term in family_terms) for sn in skill_names
    )


FAMILY_DISPLAY_NAMES = {
    "embeddings_retrieval": "embeddings/semantic retrieval",
    "vector_or_hybrid_search_infra": "vector database or hybrid search infrastructure",
    "python": "Python",
    "ranking_evaluation": "ranking evaluation methodology (NDCG/MRR/A-B testing)",
}


def _must_have_skill_coverage(candidate: dict) -> tuple[float, list[str]]:
    skill_names = {s["name"].lower() for s in candidate.get("skills", [])}
    notes = []
    hits = 0
    total = len(JD.must_have_skill_families)
    missing = []

    for family_name, terms in JD.must_have_skill_families.items():
        if _skill_family_hit(skill_names, terms):
            hits += 1
        else:
            missing.append(family_name)

    coverage = hits / total
    if missing:
        readable = [FAMILY_DISPLAY_NAMES.get(m, m) for m in missing]
        if len(readable) == 1:
            notes.append(f"no listed experience with {readable[0]}")
        else:
            notes.append(f"no listed experience with {', '.join(readable[:-1])} or {readable[-1]}")
    else:
        notes.append("covers all must-have skill families (embeddings/retrieval, vector/hybrid search, Python, ranking evaluation)")

    return coverage, notes


def _nice_to_have_bonus(candidate: dict) -> tuple[float, list[str]]:
    skill_names = {s["name"].lower() for s in candidate.get("skills", [])}
    notes = []
    hits = 0
    checked = 0

    for family_name, terms in JD.nice_to_have_skill_families.items():
        if not terms:
            continue  # handled via other signals (github score, industry text)
        checked += 1
        if _skill_family_hit(skill_names, terms):
            hits += 1
            notes.append(f"nice-to-have: {family_name.replace('_', ' ')}")

    gh = candidate["redrob_signals"].get("github_activity_score", -1)
    if gh is not None and gh >= 40:
        hits += 1
        checked += 1
        notes.append(f"active open-source contributor (GitHub activity score {gh:.0f})")
    elif gh is not None and gh >= 0:
        checked += 1

    bonus = (hits / checked) * 0.15 if checked else 0.0  # capped small bonus, never decisive alone
    return min(bonus, 0.15), notes


def _is_pure_consulting_career(candidate: dict) -> bool:
    employers = [r["company"].lower() for r in candidate["career_history"]]
    current = candidate["profile"]["current_company"].lower()
    all_companies = set(employers + [current])
    # "Only worked at consulting firms in their ENTIRE career" - must be ALL
    # employers matching, and current company must also be one (JD explicitly
    # exempts people CURRENTLY at one of these IF they have prior product co
    # experience).
    consulting_hits = [c for c in all_companies if any(f in c for f in JD.pure_consulting_firms)]
    return len(consulting_hits) > 0 and len(consulting_hits) == len(all_companies)


_SENIORITY_RANK = {
    "junior": 0, "": 1, "senior": 2, "staff": 3, "lead": 3,
    "principal": 4, "director": 5, "head": 5,
}


def _seniority_level(title: str) -> int:
    title = title.lower()
    for marker in ("principal", "director", "head", "staff", "lead", "senior", "junior"):
        if marker in title:
            return _SENIORITY_RANK[marker]
    return 1  # unmarked / mid-level


def _is_career_title_chaser(candidate: dict) -> bool:
    """
    JD: "If your career trajectory shows you optimizing for 'Senior' ->
    'Staff' -> 'Principal' titles by switching companies every 1.5 years."

    The defining feature is the COMBINATION of (a) short stints and (b) a
    monotonically escalating seniority ladder across those stints - not
    short tenure alone. Plenty of genuinely strong engineers have a couple
    of short stints (layoffs, acqui-hires, bad team fit) without being
    title-chasers; the JD's complaint is specifically about the escalation
    pattern. An earlier version of this check only looked at tenure length
    and incorrectly flagged a gold-standard candidate (alternating
    Lead/Senior titles at Adobe/Google/Locobuzz/Glance, no escalation) -
    validated and fixed against the real dataset.
    """
    history = sorted(
        candidate["career_history"],
        key=lambda r: r.get("start_date") or "",
    )
    if len(history) < JD.career_hopping_min_instances:
        return False

    short_stints = [r for r in history if r["duration_months"] <= JD.career_hopping_threshold_months]
    if len(short_stints) < JD.career_hopping_min_instances:
        return False

    levels = [_seniority_level(r["title"]) for r in history]
    # Strictly non-decreasing across the whole history, with at least one
    # real step up, is the "chasing the next title" signature.
    non_decreasing = all(b >= a for a, b in zip(levels, levels[1:]))
    has_real_increase = levels[-1] > levels[0]
    return non_decreasing and has_real_increase


def _is_pure_research_no_production(candidate: dict) -> bool:
    blob = " ".join(r.get("description", "").lower() for r in candidate["career_history"])
    blob += " " + candidate["profile"].get("summary", "").lower()
    research_markers = ["research scientist", "academic", "phd research", "research-only", "research lab"]
    production_markers = ["production", "deployed", "shipped", "real users", "scale", "live system"]
    has_research = any(m in blob for m in research_markers)
    has_production = any(m in blob for m in production_markers)
    return has_research and not has_production


def _is_recent_langchain_only(candidate: dict) -> bool:
    """JD: 'AI experience' = <12mo of LangChain+OpenAI calls, without pre-LLM ML production exposure."""
    yoe = candidate["profile"]["years_of_experience"]
    skill_names = {s["name"].lower() for s in candidate.get("skills", [])}
    has_langchain = "langchain" in skill_names
    if not has_langchain:
        return False
    # find duration_months for langchain-adjacent skills
    llm_skills = [s for s in candidate.get("skills", []) if s["name"].lower() in
                  ("langchain", "prompt engineering", "rag", "llm", "gpt")]
    max_llm_duration = max((s.get("duration_months", 0) for s in llm_skills), default=0)
    pre_llm_production_years = yoe - (max_llm_duration / 12.0)
    return max_llm_duration <= 12 and pre_llm_production_years < 2


def _is_cv_speech_robotics_primary(candidate: dict) -> bool:
    blob = " ".join(r.get("description", "").lower() for r in candidate["career_history"])
    blob += " " + candidate["profile"].get("headline", "").lower() + " " + candidate["profile"].get("summary", "").lower()
    nlp_ir_markers = ["nlp", "retrieval", "ranking", "search", "recommendation", "information retrieval", "embeddings"]
    cv_markers = ["computer vision", "image classification", "object detection", "speech recognition", "robotics", "tts", "text-to-speech"]
    has_cv = any(m in blob for m in cv_markers)
    has_nlp_ir = any(m in blob for m in nlp_ir_markers)
    return has_cv and not has_nlp_ir


def _disqualifiers(candidate: dict) -> tuple[float, list[str]]:
    """
    Returns a penalty in [0, 1]. We deliberately use STRONG penalties (not
    hard zero-out) for the "probably not move forward" cases, and a near-
    total penalty only for the explicit "we will not move forward" case
    (pure research, no production) - matching the JD's own language
    distinction between "will not" and "will probably not."

    Off-target title is also gated HERE (not left to the weighted-average
    title_fit term alone). Reasoning: a weighted average lets a high
    skill_cov / location / "ideal experience band" bonus partially
    compensate for a near-zero title_fit, which let an off-target-titled
    candidate (CAND_0079114, "Sales Executive" with several ML skills
    listed) reach structured_score=0.743 - found during validation against
    the confirmed 2,719-candidate keyword-stuffer trap set. The JD is
    explicit and absolute on this point ("not a fit, no matter how perfect
    their skill list looks"), so it must act as a gate, not a weighted
    input that other dimensions can outvote.
    """
    penalty = 0.0
    notes = []

    if any(t in candidate["profile"]["current_title"].lower() for t in JD.off_target_titles):
        penalty = max(penalty, 0.9)
        notes.append(
            f"title '{candidate['profile']['current_title']}' is fundamentally off-target for an "
            f"AI engineering role - skills/experience cannot compensate for this per the JD's explicit framing"
        )

    if _is_pure_research_no_production(candidate):
        penalty = max(penalty, 0.85)
        notes.append("appears to be pure-research background without production deployment evidence (explicit JD disqualifier)")

    if _is_recent_langchain_only(candidate):
        penalty = max(penalty, 0.6)
        notes.append("AI experience reads as recent LangChain/OpenAI usage without pre-LLM-era production ML evidence")

    if any(m in candidate["profile"]["current_title"].lower() for m in JD.architecture_title_markers):
        penalty = max(penalty, 0.35)
        notes.append("current title suggests architecture/management track rather than hands-on coding")

    if _is_pure_consulting_career(candidate):
        penalty = max(penalty, 0.7)
        notes.append("entire career appears to be at consulting firms with no product-company experience")

    if _is_cv_speech_robotics_primary(candidate):
        penalty = max(penalty, 0.6)
        notes.append("primary expertise reads as computer vision/speech/robotics without significant NLP/IR exposure")

    if _is_career_title_chaser(candidate):
        penalty = max(penalty, 0.4)
        notes.append("career history shows multiple short (<=18mo) stints, consistent with title-chasing pattern the JD flags")

    return penalty, notes


def _location_fit(candidate: dict) -> tuple[float, list[str]]:
    location = candidate["profile"]["location"].lower()
    country = candidate["profile"]["country"].lower()
    willing_to_relocate = candidate["redrob_signals"].get("willing_to_relocate", False)
    notes = []

    if any(p in location for p in JD.preferred_locations):
        return 1.0, [f"based in {candidate['profile']['location']}, a JD-preferred location"]

    if country == JD.target_country:
        if any(loc in location for loc in JD.acceptable_india_locations):
            return 0.85, [f"based in {candidate['profile']['location']}, an explicitly welcomed India location"]
        if willing_to_relocate:
            return 0.65, [f"based in {candidate['profile']['location']}, India, and open to relocation"]
        return 0.45, [f"based in {candidate['profile']['location']}, India, but relocation preference unclear"]

    # Outside India: JD says case-by-case, no visa sponsorship
    if willing_to_relocate:
        notes.append(f"based outside India ({candidate['profile']['country']}); JD offers no visa sponsorship but candidate is open to relocating")
        return 0.3, notes
    notes.append(f"based outside India ({candidate['profile']['country']}); JD does not sponsor visas and candidate is not flagged as willing to relocate")
    return 0.1, notes


def _notice_fit(candidate: dict) -> tuple[float, list[str]]:
    notice = candidate["redrob_signals"].get("notice_period_days", 90)
    if notice <= JD.ideal_notice_days:
        return 1.0, [f"notice period {notice}d is within the preferred sub-30-day window"]
    extra_blocks = (notice - JD.ideal_notice_days) / 30.0
    penalty = min(0.6, extra_blocks * JD.notice_penalty_per_extra_30d * 4)
    score = max(0.4, 1.0 - penalty)
    return score, [f"notice period {notice}d exceeds the preferred 30-day window (bar gets higher per JD)"]


def _ideal_candidate_bonus(candidate: dict) -> tuple[float, list[str]]:
    """Small additive bonus for matching the JD's explicit 'ideal candidate' composite."""
    yoe = candidate["profile"]["years_of_experience"]
    lo, hi = JD.ideal_total_experience_band
    bonus = 0.0
    notes = []

    if lo <= yoe <= hi:
        bonus += 0.05
        notes.append(f"{yoe}y total experience sits inside the JD's explicit 6-8y 'ideal candidate' window")

    blob = " ".join(r.get("description", "").lower() for r in candidate["career_history"])
    shipped_markers = ["ranking system", "search system", "recommendation system", "recsys",
                        "retrieval system", "matching system", "ranking pipeline", "search ranking"]
    scale_markers = ["scale", "production", "real users", "millions", "live"]
    if any(m in blob for m in shipped_markers) and any(m in blob for m in scale_markers):
        bonus += 0.08
        notes.append("career history shows evidence of shipping a ranking/search/recsys system at scale (JD's explicit ideal-candidate marker)")

    sig = candidate["redrob_signals"]
    if sig.get("open_to_work_flag") or (sig.get("applications_submitted_30d", 0) > 0):
        bonus += 0.02
        notes.append("clear signal of being active in the job market")

    return min(bonus, 0.15), notes


def score_structured(candidate: dict) -> StructuredScoreResult:
    title_fit, n1 = _title_fit(candidate)
    exp_fit, n2 = _experience_fit(candidate)
    skill_cov, n3 = _must_have_skill_coverage(candidate)
    nice_bonus, n4 = _nice_to_have_bonus(candidate)
    disq_penalty, n5 = _disqualifiers(candidate)
    loc_fit, n6 = _location_fit(candidate)
    notice_fit, n7 = _notice_fit(candidate)
    ideal_bonus, n8 = _ideal_candidate_bonus(candidate)

    # Weighted combination of the core fit dimensions (pre-disqualifier,
    # pre-bonus). Weights reflect the JD's own emphasis: title/function and
    # must-have skills matter most; experience band is explicitly soft;
    # location/notice are real but secondary (JD says "bar gets higher," not
    # "instant reject").
    base = (
        0.32 * title_fit
        + 0.30 * skill_cov
        + 0.18 * exp_fit
        + 0.12 * loc_fit
        + 0.08 * notice_fit
    )

    base = base + nice_bonus + ideal_bonus
    base = base * (1.0 - disq_penalty)
    final = max(0.0, min(1.0, base))

    all_notes = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8

    return StructuredScoreResult(
        candidate_id=candidate["candidate_id"],
        title_fit=title_fit,
        experience_fit=exp_fit,
        must_have_skill_coverage=skill_cov,
        nice_to_have_bonus=nice_bonus,
        disqualifier_penalty=disq_penalty,
        location_fit=loc_fit,
        notice_fit=notice_fit,
        ideal_candidate_bonus=ideal_bonus,
        structured_score=final,
        notes=all_notes,
    )

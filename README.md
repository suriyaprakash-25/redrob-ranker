# Redrob Hackathon — Candidate Ranking System

## What this is

A multi-stage ranking pipeline for the Redrob "Intelligent Candidate
Discovery & Ranking Challenge" hackathon. Ranks the top 100 candidates from
a 100,000-candidate pool against the Senior AI Engineer JD.

**Built and validated in two phases, both complete:**

1. **Phase 1 — core pipeline:** honeypot detection, structured JD-fit
   scoring, semantic similarity, trust/verified-vs-claimed modifier,
   behavioral availability modifier, deterministic reasoning generation.
   Every component was validated against the real `candidates.jsonl`
   (100K rows), and several real bugs were found and fixed — **read
   `VALIDATION_LOG.md` first**, it's the most important file for
   understanding *why* the code looks the way it does, and it's good
   material for the Stage 5 "defend your work" interview.

2. **Phase 2 — submission packaging:** git history built through genuine
   iteration, HF Spaces sandbox live, submission artifacts complete.
   Local LLM re-ranking (Ollama) was evaluated and deliberately not used —
   100K candidates × 100ms–2s per call = 2.8–55 hours, far outside the
   300s budget; the deterministic pipeline already met all required checks
   within budget.

## Quick start

```bash
pip install -r requirements.txt
python rank.py --candidates candidates.jsonl --out submission.csv
python validate_submission.py submission.csv
```

Validated end-to-end runtime on a single CPU core: **~64 seconds** for the
full 100K-candidate pool (well under the 5-minute / 16GB / CPU-only / no-
network constraint in `submission_spec.md`). No GPU, no network calls, no
hosted LLM API calls anywhere in this path.

## Architecture

```
candidates.jsonl (100K)
        │
        ▼
┌─────────────────────┐
│   honeypot.py        │  internal-consistency checks → honeypot_score
│   (hard gate)         │  (forces ~59 fabricated profiles to the bottom)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐     ┌─────────────────────┐
│   scoring.py          │     │   semantic.py          │
│   structured JD-fit    │     │   TF-IDF/SVD similarity│
│   (title, exp band,    │     │   on career_history     │
│   skills, location,    │     │   descriptions vs JD    │
│   disqualifiers)        │     │   profile text          │
└─────────────────────┘     └─────────────────────┘
        │                              │
        └──────────────┬───────────────┘
                        ▼
              0.55×structured + 0.45×semantic
                        │
                        ▼
        × trust.py (verified-vs-claimed multiplier)
                        │
                        ▼
        × behavioral.py (availability multiplier)
                        │
                        ▼
                  final_score, sorted
                        │
                        ▼
              top 100 + reasoning.py
              (deterministic, zero LLM calls,
               built entirely from real extracted facts)
                        │
                        ▼
                 submission.csv
```

## Why this architecture (short version — see job_description.md +
VALIDATION_LOG.md for the long version)

- **The JD explicitly states the trap**: "the right answer is not 'find
  candidates whose skills section contains the most AI keywords.'" Pure
  keyword/embedding matching alone reproduces exactly the failure mode the
  JD criticizes. We confirmed 2,719 deliberate keyword-stuffer candidates
  exist in the real dataset (off-target titles like HR Manager / Content
  Writer with 5+ AI skills listed) and validated that our combined scoring
  suppresses all of them below 0.08 while genuine candidates score 0.70+.
- **Structured scoring alone isn't enough either.** We found 8 genuine
  "gold-standard" candidates (Senior/Staff/Lead AI titles at Meta, Adobe,
  LinkedIn, Salesforce, etc.) whose skill lists use rare paraphrased terms
  ("Vector Representations" instead of "embeddings", etc.) that a fixed
  keyword vocabulary misses. Structured-only scoring ranked one of them
  #4,151 out of 100,000. The semantic similarity layer (built on
  career-history *prose*, not the skills list) recovers all 8 into the top
  175, most into the top 50.
- **Verified-vs-claimed matters at scale**: 34.3% of self-reported skill
  proficiency claims diverge from the platform's own tested
  `skill_assessment_scores` by more than 30 points. This is core, not
  cosmetic.
- **Honeypots are a hard gate, not a blended score**, matching the ground
  truth's "forced to relevance tier 0" and the >10%-in-top-100
  disqualification rule.
- **No per-candidate LLM calls on the full pool** — would blow the 5-minute
  CPU budget by 150-1000x at any realistic per-call latency. Any LLM use is
  architecturally bounded to a small shortlist.

## Files

| File | Purpose |
|---|---|
| `rank.py` | Single-command entry point. Produces `submission.csv`. |
| `src/honeypot.py` | Internal-consistency / fabrication detection. |
| `src/jd_requirements.py` | JD requirements encoded as structured, auditable data. |
| `src/scoring.py` | Structured JD-fit scoring (title, experience, skills, disqualifiers, location, notice). |
| `src/semantic.py` | TF-IDF + SVD semantic similarity layer. |
| `src/trust.py` | Verified-vs-claimed skill trust modifier. |
| `src/behavioral.py` | Availability/reachability modifier from `redrob_signals`. |
| `src/final.py` | Combines all of the above into `final_score`. |
| `src/reasoning.py` | Deterministic, fact-grounded reasoning text generator. |
| `VALIDATION_LOG.md` | Real findings + bugs caught/fixed against the actual dataset. **Read this.** |
| `PLAN.md` | Original implementation plan (completed — documents build phases and architectural decisions). |
| `data/gold_candidates.json` | 8 manually-discovered high-confidence "ideal candidate" profiles, used throughout as a validation set. |
| `hackathon_bundle_docs/` | Markdown copies of the original hackathon docs (JD, submission spec, signals reference, schema) for quick reference without re-opening the original .docx files. |

## Compute environment this was developed/validated against

Single CPU core, ~4GB RAM available, no internet access to huggingface.co
(only PyPI-class package registries reachable). This is why the semantic
layer defaults to TF-IDF/SVD (zero model-weight downloads) rather than
sentence-transformers — see `src/semantic.py` docstring for the full
reasoning and the documented upgrade path.

# PLAN.md — Remaining work, for Claude Code on Suriya's laptop

Context for whichever Claude Code instance picks this up: Phase 1 (this
bundle's `src/` and `rank.py`) was built and validated against the real
100K-candidate dataset already — `rank.py` runs end-to-end in ~76s on a
single CPU core and passes `validate_submission.py`. **Do not start over.**
The job now is to (a) add the optional Ollama re-ranking stage, (b) turn
this into a real git-tracked repo with genuine iterative history, (c) stand
up the sandbox, and (d) finish the submission package. Read
`VALIDATION_LOG.md` before touching the scoring code — it documents real
bugs already found and fixed; don't reintroduce them.

## Step 0 — Get oriented (15 min)

1. Read `README.md`, then `VALIDATION_LOG.md` in full.
2. Run the existing pipeline once, untouched, to confirm it still works on
   this machine:
   ```bash
   pip install -r requirements.txt
   python rank.py --candidates /path/to/candidates.jsonl --out submission_baseline.csv
   python validate_submission.py submission_baseline.csv
   ```
3. Skim `src/*.py` docstrings — each module explains its own design
   rationale and links back to specific JD clauses or validation findings.
   `data/gold_candidates.json` has 8 known-good candidates worth spot-
   checking after any change you make.

## Step 1 — Initialize git properly (don't skip this)

Stage 4 manual review explicitly checks "git history authenticity (real
iteration vs single dump)" and treats a flat history as a rejection
signal. So:

```bash
git init
git add README.md VALIDATION_LOG.md PLAN.md .gitignore
git commit -m "Initial scaffold: README, validation log, plan"

git add src/jd_requirements.py src/honeypot.py
git commit -m "JD requirements model + honeypot/data-integrity detector"

git add src/scoring.py
git commit -m "Structured JD-fit scoring engine"

git add src/semantic.py
git commit -m "TF-IDF/SVD semantic similarity layer"

git add src/trust.py src/behavioral.py
git commit -m "Verified-vs-claimed trust modifier + behavioral availability modifier"

git add src/final.py src/reasoning.py rank.py
git commit -m "Score combination + deterministic reasoning generator + entry point"

git add validate_submission.py data/
git commit -m "Bundle validator + gold-candidate validation set"
```

Then keep committing for real as you make the changes in the steps below —
don't batch everything into one final commit. Several small, honest commits
covering bug fixes, the Ollama addition, and config tweaks is exactly what
"real iteration" looks like.

## Step 2 — Decide: do you actually need the LLM stage?

Be honest with yourself here. The current pipeline already:
- Passes the validator
- Runs in ~76s, way under budget
- Separates 2,719 known keyword-stuffer traps from genuine candidates with
  a wide score margin (traps max 0.07, gold candidates 0.70+)
- Recovers paraphrased-vocabulary "Tier 5" candidates via the semantic layer
- Produces fact-grounded, varied, honest reasoning text

Adding Ollama is valuable mainly for: (a) directly answering the JD's own
"probably some LLM-based re-ranking" framing in the interview, and (b)
potentially sharpening reasoning quality / catching subtle re-ranking
errors the deterministic layers miss. It is NOT free — it adds a real
dependency (Ollama must be installed + model pulled in the reproduction
environment) and a new failure mode to defend at Stage 5. If you're tight
on the 4-day clock, it's reasonable to deprioritize this and spend the time
on Steps 4-6 instead (sandbox, git hygiene, deck) which are unconditionally
required. If you do build it, keep it strictly optional/fallback-safe (see
below) so a missing Ollama install never breaks `rank.py`.

If proceeding:

### 2a. Install Ollama and pull the model

```bash
# macOS
brew install ollama
ollama serve &              # starts local server on localhost:11434
ollama pull qwen2.5:3b-instruct
# or qwen2.5:1.5b-instruct if RAM-constrained
```

### 2b. Build `src/llm_rerank.py`

Interface contract:
- Input: the top ~150-200 candidates by `final_score` from the existing
  pipeline (NOT all 100K — this is the bounded-cost design from the
  original architecture discussion).
- For each, send a compact prompt (candidate facts + JD summary, NOT the
  full raw JSON) to `http://localhost:11434/api/generate` or
  `/api/chat`, asking for either (a) a re-rank nudge score in some range,
  or (b) a refined reasoning sentence.
- **Critical: wrap every call in try/except.** If Ollama isn't running, or
  the request times out, or returns malformed output — log a warning to
  stderr and fall through to the existing deterministic reasoning /
  ranking for that candidate. `rank.py` must ALWAYS produce a complete,
  valid submission even with zero Ollama availability. This is not just
  robustness for its own sake — the Stage 3 reproduction environment may
  not have Ollama installed at all, so the script must degrade gracefully,
  not crash.
- **No network calls** — `localhost` HTTP to a model you're running
  yourself is not the same thing as calling a hosted LLM API, and the spec
  forbids "OpenAI, Anthropic, Cohere, Gemini, or any hosted LLM service,"
  not local inference. State this explicitly in `submission_metadata.yaml`
  → `methodology_summary` and be ready to explain the distinction
  cleanly at Stage 5 if asked — don't leave it ambiguous.
- Time-box it: re-ranking ~200 candidates at a few seconds each on CPU
  should land well inside the remaining time budget (current pipeline uses
  ~76s of the 300s budget, leaving ~220s of margin) but MEASURE it on your
  actual machine and log the timing. If it's going to bust the 5-minute
  ceiling, cut the shortlist size (e.g. top 100 instead of 200) rather than
  skipping validation of the timing.

### 2c. Validate before trusting it

Re-run against `data/gold_candidates.json` — confirm the LLM stage doesn't
demote any of the 8 known-good candidates, and spot check that it isn't
hallucinating skills/companies not in the source profile (the Stage 4
"no hallucination" check applies to LLM-touched reasoning just as much as
the deterministic version — arguably more, since LLM output is exactly
where hallucination risk concentrates). If the LLM-generated reasoning
introduces ANY fact not present in the candidate's actual profile, that's
a Stage 4 red flag — consider keeping the deterministic reasoning as the
default and using the LLM only for the numeric re-rank nudge, not the
reasoning text, if hallucination shows up in testing.

## Step 3 — Re-run full validation after any scoring changes

Any time `src/*.py` changes, re-run the checks already established:

```bash
python -c "
import sys, json
sys.path.insert(0, 'src')
from final import combine_scores
from semantic import build_semantic_index, semantic_scores

gold_ids = {'CAND_0005538','CAND_0006567','CAND_0030468','CAND_0037980',
            'CAND_0061257','CAND_0068351','CAND_0080766','CAND_0093193'}
candidates = [json.loads(l) for l in open('/path/to/candidates.jsonl')]
index = build_semantic_index(candidates)
sem = semantic_scores(index)
results = sorted(
    [combine_scores(c, sem[c['candidate_id']]) for c in candidates],
    key=lambda r: -r.final_score,
)
rank_of = {r.candidate_id: i+1 for i, r in enumerate(results)}
for cid in gold_ids:
    print(cid, rank_of[cid])
"
```

All 8 should stay well inside the top 200 (currently top 175 worst-case,
most in top 50). If a change pushes any of them out, that's a regression —
figure out why before moving on.

Also re-run the trap-suppression check (see VALIDATION_LOG.md's exact
methodology) to confirm the 2,719 keyword-stuffer candidates are still
suppressed below ~0.1 after your changes.

## Step 4 — Final submission CSV

```bash
python rank.py --candidates /path/to/candidates.jsonl --out submission.csv
python validate_submission.py submission.csv
```

Must print "Submission is valid." with no errors. Rename to your registered
participant ID per the spec (`team_xxx.csv`) before upload.

## Step 5 — Sandbox link (HuggingFace Spaces)

Requirement: a small-sample, end-to-end-runnable demo, not the full 100K
pipeline. Suggested approach:

1. Create a new HF Space (Gradio or Streamlit SDK, CPU-basic free tier).
2. Bundle `src/`, `rank.py`, and a small sample file (e.g. the provided
   `sample_candidates.json`, trimmed to ~50 rows) into the Space.
3. Build a minimal `app.py`: a button that runs `rank.py`-equivalent logic
   against the sample file and displays the resulting top-N with reasoning
   in a table. This does NOT need to be the full 100K run — it just needs
   to prove the code executes and produces sane output.
4. Push the Space, copy its URL into `submission_metadata.yaml` →
   `sandbox_link`.

## Step 6 — Fill in `submission_metadata.yaml`

Copy `submission_metadata_template.yaml` from the original hackathon bundle
to the repo root as `submission_metadata.yaml`. Fill in real team info,
GitHub repo URL, sandbox link, compute environment (your actual laptop
specs), and AI tools declaration — be honest, the rules state this isn't
penalized but contradicting it at interview is a much worse signal than
the AI use itself. For `methodology_summary`, draw from this bundle's
README "Why this architecture" section, condensed to ≤200 words, and
explicitly mention the Ollama/local-LLM distinction from Step 2 if you
built it.

## Step 7 — The methodology deck (submission requirement #2)

A deck/PPT converted to PDF explaining what you built, why, and how it
works. Suggested structure (keep it tight, this doesn't need to be long):

1. **The trap, and why naive approaches fail** — show the
   `sample_submission.csv` example (HR Manager ranked #1 by raw skill
   count) as the explicit "what NOT to do," validated against the real
   2,719-candidate keyword-stuffer pool.
2. **Architecture diagram** — the pipeline diagram from this README.
3. **The vocabulary-gap finding** — the 8 gold candidates with paraphrased
   skill terms, and the before/after rank improvement from adding the
   semantic layer. This is a genuinely good, concrete story.
4. **Trust + behavioral modifiers** — the 34.3% claimed-vs-tested gap
   statistic, and the honeypot hard-gate design.
5. **Compute constraints satisfied** — the 76s timing breakdown, no GPU,
   no network.
6. **(If built) the Ollama stage** — what it adds and why it's bounded/
   optional rather than load-bearing.
7. **Honest limitations** — pick 2-3 real ones (e.g. TF-IDF/SVD is lexical-
   paraphrase-robust but not deep-semantic-robust; the rule-based
   disqualifiers are calibrated to this specific JD's stated language and
   would need rework for a different JD). Naming real limitations
   yourself is a stronger Stage 5 signal than pretending there are none.

Use the `pptx` skill (`/mnt/skills/public/pptx/SKILL.md`) if building this
with Claude Code's document tools, then export/convert to PDF per the
submission requirement.

## Step 8 — Final pre-submission checklist

- [ ] `python rank.py --candidates ... --out submission.csv` runs in under
      5 minutes on this machine, measured fresh (not from a warm cache)
- [ ] `python validate_submission.py submission.csv` passes
- [ ] Honeypot count in top 100 is 0 (or very low; must be ≤10)
- [ ] `requirements.txt` lists exact versions used
- [ ] `submission_metadata.yaml` is filled in and matches portal entries
      you're about to submit
- [ ] GitHub repo is pushed, public or org-access-granted, with real
      multi-commit history
- [ ] Sandbox link is live and actually loads
- [ ] README.md has the single reproduce command spelled out clearly
- [ ] You (the human) can explain every scoring component well enough to
      defend it live — re-read VALIDATION_LOG.md the night before if a
      Stage 5 interview gets scheduled

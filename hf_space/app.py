"""
Redrob Intelligent Candidate Ranking — Gradio demo.

Runs the full pipeline (honeypot detection, structured JD-fit scoring,
TF-IDF/SVD semantic similarity, trust modifier, behavioral modifier) on a
50-candidate sample and displays the ranked results with reasoning.

The src/ modules here are identical to the production rank.py pipeline —
this demo exists to show the code runs end-to-end, not as a separate
implementation.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import gradio as gr
from final import combine_scores
from semantic import build_semantic_index, semantic_scores
from reasoning import generate_reasoning

_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'sample_candidates.json')
with open(_DATA_PATH, encoding='utf-8') as f:
    CANDIDATES = json.load(f)


def run_pipeline():
    index = build_semantic_index(CANDIDATES)
    sem = semantic_scores(index)

    results = sorted(
        [combine_scores(c, sem[c['candidate_id']]) for c in CANDIDATES],
        key=lambda r: -r.final_score,
    )

    rows = []
    display_rank = 1
    for r in results:
        c = next(x for x in CANDIDATES if x['candidate_id'] == r.candidate_id)
        p = c['profile']
        reasoning_text = generate_reasoning(c, r)

        if r.is_honeypot:
            rank_display = "EXCLUDED"
            score_display = f"{r.final_score:.3f} (honeypot)"
        else:
            rank_display = str(display_rank)
            score_display = f"{r.final_score:.3f}"
            display_rank += 1

        rows.append([
            rank_display,
            r.candidate_id,
            p['current_title'],
            p['current_company'],
            round(p['years_of_experience'], 1),
            score_display,
            reasoning_text,
        ])

    return rows


with gr.Blocks(title="Redrob Candidate Ranker") as demo:
    gr.Markdown(
        """
        ## Redrob Intelligent Candidate Discovery & Ranking — Demo

        Ranks a 50-candidate sample for the **Senior AI Engineer** role.

        **Pipeline:** honeypot hard-gate → structured JD-fit scoring (title, experience,
        must-have skills, location, notice) + TF-IDF/SVD semantic similarity on career
        descriptions → trust multiplier (verified-vs-claimed skills) → behavioral
        availability modifier → deterministic fact-grounded reasoning.

        Honeypot candidates are excluded from ranking and shown at the bottom with
        `EXCLUDED` rank. Keyword-stuffer traps (off-target titles with AI skills) score
        below 0.08 and do not appear in the top results.

        Full methodology and source: [GitHub](https://github.com/suriyaprakash-25/redrob-ranker)

        > **Note on scores:** absolute scores are lower here (~0.55 max) than in the
        > full 100K run (~0.99 max) because the TF-IDF semantic layer's normalization
        > constants are calibrated to a large corpus; with 50 candidates the raw cosines
        > compress. Relative ordering and trap/honeypot suppression are unaffected.
        """
    )

    run_btn = gr.Button("Run Pipeline on Sample Candidates", variant="primary", size="lg")

    output = gr.Dataframe(
        headers=["Rank", "Candidate ID", "Title", "Company", "YoE", "Score", "Reasoning"],
        datatype=["str", "str", "str", "str", "number", "str", "str"],
        wrap=True,
        column_widths=["60px", "120px", "180px", "160px", "60px", "140px", None],
    )

    run_btn.click(fn=run_pipeline, inputs=[], outputs=output)

demo.launch()

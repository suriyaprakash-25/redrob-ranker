---
title: Redrob Ranker
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# Redrob Intelligent Candidate Discovery & Ranking

Demo for the Redrob "Intelligent Candidate Discovery & Ranking Challenge".

Ranks a 50-candidate sample for a **Senior AI Engineer** role using a
multi-stage pipeline: honeypot hard-gate → structured JD-fit scoring →
TF-IDF/SVD semantic similarity → trust & behavioral modifiers → deterministic
fact-grounded reasoning.

Click **Run Pipeline on Sample Candidates** to execute the full pipeline.

Full source and methodology: https://github.com/suriyaprakash-25/redrob-ranker

# TowerMind MVP Controlled Benchmark (Seed 101)

**Date:** 2026-11-06T13:24:28.299845+00:00  
**Baseline Experiment ID:** `baseline_seed_101`  
**Guided Experiment ID:** `guided_seed_101`  
**Random Seed:** 101  
**Training Timesteps:** 512  
**Evaluation Episodes:** 2  

---

## 1. Metric Performance Comparison

| Metric | Baseline PPO | LLM-Guided PPO | Delta | Outcome |
|---|---|---|---|---|
| **Win Rate** | 0.0% | 0.0% | 0.0% | Parity |
| **Mean Waves Survived** | 10.00 | 10.00 | 0.00 | Parity |
| **Mean Final Lives** | 20.00 | 20.00 | 0.00 | Parity |
| **Mean Episode Return** | 17.85 | 26.70 | +8.85 | Improvement |
| **Action Legality Rate** | 99.1% | 99.1% | 0.0% | Healthy |

---

## 2. Meta-Controller Segment Review & Decisions

| Segment | Timesteps | Active Profile | Proposed Candidate | Decision | Rationale & Hypothesis |
|---|---|---|---|---|---|
| Seg 1 | 256 | `llm-cand-001_1` | `llm-cand-001_1` | **PROMOTED** | Penalize enemy leaks more heavily to encourage front-line defenses. (Hypothesis: Higher leak penalty will reduce average leaks per wave.) |
| Seg 2 | 512 | `llm-cand-001_2` | `llm-cand-001_2` | **PROMOTED** | Penalize enemy leaks more heavily to encourage front-line defenses. (Hypothesis: Higher leak penalty will reduce average leaks per wave.) |

---

## 3. LLM Gateway Telemetry & Compute Budget

- **Total LLM Requests:** 2
- **Total Fallbacks Triggered:** 0
- **Tokens Consumed:** 460
- **Estimated API Cost (USD):** $0.0010

---

## 4. Scientific Findings & Honest Disclosure

> **Note on Statistical Rigor:** This report reflects a preliminary, controlled MVP experiment run. Differences observed in a single matched-seed run serve as exploratory evidence of the meta-controller's workflow and safety guarantees, rather than definitive proof of universal policy superiority across all maps.

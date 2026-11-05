# TowerMind Map Generalization Benchmark Report

**Evaluation Episodes per Map:** 1  
**Base Seed:** 42  

## 1. Summary Transfer Performance

| Partition | Mean Win Rate | Mean Waves Survived |
|---|---|---|
| **Training Maps** (s_curve, zigzag) | 0.0% | 3.50 |
| **Held-Out Maps** (chokepoint, spiral) | 0.0% | 3.50 |
| **Generalization Transfer Gap** | +0.0% | +0.00 waves |

## 2. Per-Map Breakdown

| Map | Type | Win Rate | Mean Waves | Legality Rate |
|---|---|---|---|---|
| `s_curve` | training | 0.0% | 4.00 | 100.0% |
| `zigzag` | training | 0.0% | 3.00 | 100.0% |
| `chokepoint` | heldout | 0.0% | 3.00 | 100.0% |
| `spiral` | heldout | 0.0% | 4.00 | 100.0% |

---

> **Methodology:** Unseen held-out maps evaluate whether the agent learned generalized path-coverage heuristics rather than memorizing coordinate patterns on specific training geometries.
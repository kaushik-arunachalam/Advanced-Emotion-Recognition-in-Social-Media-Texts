# Phase 3C — room-for-improvement search (DONE — see Capstone_Phase3C_Improvements.ipynb)

**Rule:** every candidate is judged on the *held-out pool* (gold-labelled pool posts that are **not** among the 500 training labels). The 80% test split is used only once, for the final chosen version.

## Done
| Step | Result (held-out pool, total accuracy / F1, average of 3 datasets) |
|---|---|
| v1 DA-HGNN-F (graph student ⊕ calibrated teacher) | .678 / .675 (5 seeds) |
| + supervised MLP as a 3rd equal vote | **.691 / .684** (5 seeds); better on all 3 datasets, TEC +2.5 acc |
| Correct & Smooth with gold labels | .610 / .618: **hurts**, dropped |
| Oracle (any of teacher/student/MLP right) | .779: large complementary headroom |
| Fine-tuned RoBERTa-large (LoRA) alone, seed 0 | .683 / .680: strong baseline, best single model on TEC (.553) |
| 4-way vote (+ fine-tuned RoBERTa), seed 0 | .693 / .685 |

Cached predictions: `results/improved_v2/runs/` holds the fine-tuned DistilRoBERTa and RoBERTa-LoRA for all 5 label draws × 3 datasets (30 files).

## Next (not started / interrupted)
1. `probe_st.py`: self-training (Noisy Student). RoBERTa-LoRA trained on gold + the 5-way ensemble's confident soft labels; seeds 0–1. Interrupted before any output.
2. Evaluate 3/4/5-way ensembles over all 5 seeds on the held-out pool; pick the winner with a rule fixed in advance (best mean accuracy + F1 over datasets).
3. Build `Capstone_Phase3C_Improvements.ipynb`: headroom → candidates → selection → **one** confirmatory test-split evaluation (5 seeds, t-tests vs v1 and the best single model) → update `Phase3_Review_Record.md`.

## How to resume
Run from the `Capstone Project` folder with the capstone venv, `MPLBACKEND=Agg`, inside a retry loop (the RTX 2060 throws sporadic CUDA/cuBLAS faults; everything is cached, so retries resume):
```
C:/Users/kaush/venvs/capstone/Scripts/python.exe -u dev/phase3c_wip/probe_st.py
```

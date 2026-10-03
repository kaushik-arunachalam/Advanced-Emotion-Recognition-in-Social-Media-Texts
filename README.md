# Advanced Emotion Recognition in Social Media Texts

Replication and improvement of the base paper:

> Maazallahi, A., Asadpour, M., & Bazmi, P. (2025). *Advancing emotion recognition in social media: A novel integration of heterogeneous neural networks with fine-tuned language models.* Information Processing & Management, 62, 103974.

## The problem
Two strong fine-tuned emotion models, DistilRoBERTa and RoBERTa-large, give **different labels to 27–33% of social-media posts**, and on those posts each model is right only 14–46% of the time. The base paper trains a heterogeneous graph neural network (Tweet–Phrase–Emotion) on the posts where the two models agree and uses it to label the posts where they disagree.

## What this repository contains
| Phase | Notebook | Summary |
|---|---|---|
| **3A — Replication** | `Capstone_Phase3A_Paper_Replication.ipynb` | Full re-implementation of the paper (all 180 grid configurations, 3 datasets). The baselines and the paper's HGNN-Max reproduce. The paper's best result (HGNN-Min) does not: that model family is seed-unstable (0.344 ± 0.228 accuracy on disagreed posts). |
| **3B — Improved model** | `Capstone_Phase3B_Improved_Model.ipynb` | **DA-HGNN-F**, a disagreement-aware heterogeneous GNN (multi-view features, semantic tweet–tweet relation, soft distillation + 500 gold labels) fused with a calibrated LM teacher. Includes ablations, a label-efficiency study and significance tests. |
| **3C — Room for improvement** | `Capstone_Phase3C_Improvements.ipynb` | **DA-HGNN-E**, an equal-vote ensemble that adds LoRA-fine-tuned LMs. It was selected on held-out data with a pre-registered rule, and the test split was used once. |

Also included:
- `Phase3_Review_Record.md`: literature survey, research gaps, data handling, design log, results and viva Q&A.
- `Capstone_Phase3_Slides.pptx`: the 6-slide review deck.

## Results
Untouched 80% test split · 500 gold labels per dataset · mean of 5 seeds. Total accuracy / weighted F1:

| Dataset | Paper's best (reported) | v1 DA-HGNN-F | Fine-tuned RoBERTa-large (LoRA) | **v2 DA-HGNN-E** | **Gain vs paper (acc / F1, points)** |
|---|---|---|---|---|---|
| GoEmotions | .737 / .738 | .788 / .785 | .771 / .774 | **.801 / .795** | **+6.4 / +5.7** |
| Friends | .535 / .578 | .719 / .717 | .713 / .708 | **.732 / .724** | **+19.7 / +14.6** |
| TEC (unseen) | .436 / .443 | .517 / .518 | .546 / .541 | **.552 / .546** | **+11.6 / +10.4** |

* Every gain over the paper's HGNN (our replication) and the raw LMs is significant (p < 0.001; McNemar p < 10⁻¹⁰).
* The largest single lever is calibrating the LM teacher with the few gold labels (−3.6 points if removed).
* Honest limits:
  * the method needs a small labelled set; label-free, it only matches the LM average;
  * inside v2, the graph student contributes within ±0.3 points; it helps most when labels are scarce (≤ 500) and on the unseen TEC.

## Repository structure
```
├── Capstone_Phase3A_Paper_Replication.ipynb   # 3A: replication of the paper
├── Capstone_Phase3B_Improved_Model.ipynb      # 3B: DA-HGNN-F
├── Capstone_Phase3C_Improvements.ipynb        # 3C: DA-HGNN-E
├── Phase3_Review_Record.md                    # rubric-aligned review record
├── Capstone_Phase3_Slides.pptx                # 6-slide review deck
├── requirements.txt
├── data/README.md                             # dataset sources (data itself not included)
├── results/
│   ├── replication/                           # 3A tables + figures (paper vs ours)
│   ├── improved/                              # 3B tables, ablations, label budget, significance, figures
│   └── improved_v2/                           # 3C held-out selection, test results, figure
└── dev/
    ├── notebook_generators/                   # scripts that generate the notebooks
    ├── slides/                                # script that builds the slide deck
    └── phase3c_wip/                           # 3C exploration scripts (held-out analyses)
```

## How to run
1. `pip install -r requirements.txt` (install a CUDA build of PyTorch for GPU), or open the notebooks in Google Colab with a GPU runtime.
2. Run the notebooks **in order: 3A → 3B → 3C**, from the repository root. Each one reuses the previous one's cached outputs.
3. **3A downloads the datasets automatically** into `data/raw/` (see `data/README.md`).

Trained-model predictions are cached under `results/*/runs/` (not committed), so interrupted runs resume where they stopped. A full run from scratch takes a few hours on a single consumer GPU (RTX 2060, 6 GB).

## Datasets
GoEmotions (Demszky et al., 2020), EmotionLines / Friends (Chen et al., 2018) and the Twitter Emotion Corpus (Mohammad, 2012). The data is **not redistributed** here; see `data/README.md` for sources and licences.

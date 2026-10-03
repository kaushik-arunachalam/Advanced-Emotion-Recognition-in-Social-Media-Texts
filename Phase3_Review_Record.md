# Phase 3 (End-Sem) Review Record — PSID 181

**M A Kaushik · CH.SC.U4CSE24123**
**Base paper:** Maazallahi, Asadpour & Bazmi (2025). *Advancing emotion recognition in social media: A novel integration of heterogeneous neural networks with fine-tuned language models.* Information Processing & Management, 62, 103974.

This file records every component the Phase 3 rubric asks for, with the evidence behind it. It is the source material for the end-sem PPT. All results are final: Phase 3A replication and Phase 3B improved model.

**One-line summary:** we reproduced the base paper and found why its best results are unstable. We then built **DA-HGNN-F (v1)** and the ensemble **DA-HGNN-E (v2)**. v2 beats the paper on all three datasets by **+6.4 (GoEmotions), +19.7 (Friends) and +11.6 (TEC) accuracy points** (p < 0.001), using 500 labelled posts per dataset.

## Rubric map

| Rubric component (marks) | What "Excellent" asks for | Where it is covered | Evidence |
|---|---|---|---|
| Problem identification & literature survey (40) | In-depth understanding of the problem; gaps in existing literature; future research | §1 | Replication notebook (3A), literature table §1.5, gaps §1.6 |
| Handling datasets & data preprocessing (40) | Clear data sources; inconsistencies addressed; justified features | §2 | `data/raw/`, Table 1 replication, §2.2 inconsistency log |
| Algorithmic design & approach (40) | Strong knowledge of the algorithm; convincing results on various test cases | §3 | 3A notebook (paper reproduced), 3B notebook (improved model, ablations, seeds, t-tests) |
| Presentation & interaction (30) | Clear oral and visual delivery | §4 | Slide outline, figure list, viva Q&A |

---

## 1. Problem identification & literature survey (40 marks)

### 1.1 Problem statement
Social-media posts are short and informal, and full of sarcasm, slang and implicit emotion. Fine-tuned language models (LMs) such as DistilRoBERTa and RoBERTa-large often give **different emotion labels to the same post** (*label non-compliance*). For about 27–33% of posts we cannot tell which label to trust. **Goal:** decide the correct emotion (anger, disgust, fear, joy, neutral, sadness, surprise) for every post, *including the ones the models disagree on*, without collecting new human labels.

### 1.2 Why it matters
* Public-mood tracking, crisis/distress detection, brand monitoring: wrong labels on the hard posts are the most costly ones.
* Single models give no warning when they are wrong. Disagreement between models is a free, label-free signal of uncertainty.

### 1.3 Base paper in one slide
1. Two LMs (DistilRoBERTa, RoBERTa-large) label every post. Posts where they agree form the **compliance** set, used as training data. The rest form the **non-compliance** set.
2. A heterogeneous graph is built: **Tweets – Phrases (TextRank) – Emotions**. A phrase is linked to an emotion if its mean score is > 0.5 and its std is < 0.1.
3. A heterogeneous GNN (GraphSAGE-style messages, HAN-style semantic attention) is trained on agreed tweets and predicts every tweet.
4. The paper reports gains on GoEmotions, Friends and TEC (Tables 5–7).

### 1.4 What our replication (Phase 3A) found — evidence-based gaps in the base paper
Notebook: `Capstone_Phase3A_Paper_Replication.ipynb`. All 180 configurations of the paper's grid were trained on 3 datasets.

| Check | Paper | Ours | Verdict |
|---|---|---|---|
| Compliance split (GoEmotions / Friends / TEC) | 74 / 70 / 65 % | 73 / 72 / 67 % | ✅ reproduced |
| LM baselines (RoBERTa, DistilRoBERTa, DistilRoBERTa-v2) | Tables 5–6 | within ≈ 0.02 on almost every cell | ✅ reproduced |
| HGNN-Max on GoEmotions, accuracy (agreed / disagreed / total) | .720 / .530 / .671 | .713 / .536 / .665 | ✅ reproduced (single run) |
| HGNN-Min, the paper's best (disagreed accuracy) | .677 | .397 | ❌ not reproduced |
| Attention > mean aggregation (Table 3) | .56 vs .50 | +0.010, p = 0.70 | ❌ no difference |
| Significant gains over LMs (Table 7) | p < 0.05 | none over 10 seeds | ❌ not reproduced |
| Stability of the reproduced HGNN-Max | not reported | accuracy on disagreed posts **0.344 ± 0.228** (0.05–0.61) over 10 seeds | ⚠️ the matching run is a lucky draw |
| Trivial rule "disagree → neutral" (no training) | not reported | GoEmotions disagreed accuracy **0.759**, total F1 **0.740** | ⚠️ matches/beats the paper's best HGNN (0.677 / 0.738) |

**Root causes identified**
* Disagreed tweets are a **different population**: 76% neutral on GoEmotions vs 68% in the agreed set. The model is trained only on agreed tweets.
* The paper's "zero features for disagreed tweets" option is **never seen in training**, so predictions on those tweets are arbitrary. Validation on agreed tweets cannot detect this.
* Models are selected by **similarity to RoBERTa**, using test results, with a single run and no seeds.

### 1.5 Literature survey

| # | Area | Work | What it does | Strength | Limitation / gap for our problem | How we use it |
|---|---|---|---|---|---|---|
| 1 | Datasets | Demszky et al. 2020 — **GoEmotions** (ACL) | 58k Reddit comments, 27 emotions + neutral, multiple raters | Large, rater-level data | Multi-label, highly imbalanced (70% neutral in our subset) | Main dataset |
| 2 | Datasets | Mohammad 2012 — **TEC** (*SEM) | 21k tweets labelled by emotion hashtags | Not used to train any of the LMs, so a fair test | No neutral class; hashtag-based labels are noisy | Unseen test set |
| 3 | Datasets | Chen et al. 2018 — **EmotionLines/Friends** (LREC) | 14.5k TV-dialogue utterances, 5 annotators each | Conversational, majority-vote labels | Utterances are short; we ignore dialogue context | Second test set |
| 4 | Fine-tuned LMs | Hartmann et al. 2023 — *More than a feeling* (IJRM) | DistilRoBERTa / RoBERTa-large emotion models trained on 6 datasets | Strong off-the-shelf 7-class emotion models | Black boxes; disagree with each other; GoEmotions is in their training data | The two "annotators" |
| 5 | Fine-tuned LMs | Liu et al. 2019 **RoBERTa**; Sanh et al. 2019 **DistilBERT** | Pre-trained Transformers | Context-aware representations | Only the 7 output scores are used in the paper | We also use their **hidden-state embeddings** |
| 6 | Emotion models | Alhuzali & Ananiadou 2021 — **SpanEmo** (EACL) | Emotion classification as span prediction with label–label correlations | SOTA on SemEval-2018 multi-label | Needs gold training labels | Related-work context |
| 7 | Emotion models | Dong et al. 2026 — multi-agent LLMs (PLOS ONE 21(2) e0342053) | Several LoRA-tuned LLaMA agents; a meta-agent resolves conflicts with confidence-weighted, entropy-calibrated fusion | Explicit disagreement resolution | 70B-parameter models; expensive; supervised | Supports **confidence/entropy-aware fusion** |
| 8 | Combining annotators | Dawid & Skene 1979 (JRSS-C) | EM estimate of each annotator's confusion matrix and the true label | Classic, label-free | Weakly identifiable with only 2 annotators | Motivates treating LMs as noisy annotators |
| 9 | Weak supervision | Ratner et al. 2017 — **Snorkel** (VLDB) | Label model combines noisy labelling functions into probabilistic labels | Produces soft training labels without gold | Assumes many conditionally independent sources | Motivates **probabilistic (soft) pseudo-labels** |
| 10 | Learning from disagreement | Uma et al. 2021 — survey (JAIR 72) | Reviews methods that learn from annotator disagreement | Shows that disagreement carries information | Focuses on human annotators | Disagreement is a **signal**, not noise |
| 11 | Learning from disagreement | Fornaciari et al. 2021 — *Beyond Black & White* (NAACL) | Soft-label auxiliary loss from annotator distributions | Improves accuracy and calibration | Needs annotator distributions | **Soft-target (KL) training** |
| 12 | Learning from disagreement | Plank 2022 — human label variation (EMNLP) | Argues label variation is a property of data, models and evaluation | Clear conceptual framing | Position paper | Framing of the problem |
| 13 | Learning from disagreement | Xu, Theune & Braun 2024 (arXiv 2409.17577) | Compares multi-label, ensemble and instruction-tuning ways to use disagreement | Up-to-date comparison | Hate-speech domain | Supports keeping disagreement information as features |
| 14 | Noisy labels | Northcutt et al. 2021 — **Confident Learning** (JAIR 70) | Estimates label noise; prunes or weights noisy examples | Principled noise handling | Needs a trained classifier's probabilities | **Confidence-weighting** of pseudo-labels |
| 15 | Distillation / self-training | Hinton et al. 2015 (KD); Lee 2013 (pseudo-label); Xie et al. 2020 (**Noisy Student**, CVPR) | Train a student on a teacher's soft or hard predictions, with noise | Students can beat teachers | Can copy teacher bias | Student GNN learns from **both** LM teachers; **input noise** (masking) |
| 16 | Keyword graphs | Mihalcea & Tarau 2004 — **TextRank** (EMNLP) | PageRank over a word co-occurrence graph | Unsupervised key phrases | Sparse for short texts | Phrase nodes (as in the paper) |
| 17 | GNNs | Hamilton et al. 2017 **GraphSAGE**; Wang et al. 2019 **HAN** | Neighbour aggregation; semantic attention over relations | Inductive; relation-aware | HAN attention is useless when a node has one relation | Base layer (paper) |
| 18 | Heterogeneous GNNs | Schlichtkrull et al. 2018 **R-GCN**; Hu et al. 2020 **HGT** (WWW) | Relation-specific weights; type-aware transformer attention | Strong on multi-relational graphs | Heavier | Related work / design choices |
| 19 | Heterogeneous GNNs | Lv et al. 2021 — *Are we really making much progress?* (**SimpleHGN**, KDD) | Shows many HGNN gains vanish under fair tuning; simple models are strong | Rigorous benchmarking | — | Motivates **honest baselines, seeds, ablations** |
| 20 | Text graphs | Yao et al. 2019 **TextGCN** (AAAI); Lin et al. 2021 **BertGCN** (Findings ACL) | Document–word graphs; BERT embeddings as GCN node features, with predictions interpolated | Graph + LM features beat either alone | Need gold labels | **LM embeddings as node features**; doc–doc edges |
| 21 | Label propagation | Huang et al. 2021 — **Correct & Smooth** (ICLR) | Simple model + propagate errors + smooth predictions over the graph | Matches GNNs at a fraction of the cost | Needs homophily | Strong **graph baseline** |
| 22 | Masked training | Shi et al. 2021 — **UniMP** (IJCAI); Rong et al. 2020 **DropEdge** (ICLR) | Mask labels/features during training so the model learns to infer them from neighbours | Removes the train/test mismatch | — | **Disagreement-simulation (score masking) training** |
| 23 | Calibration | Guo et al. 2017 (ICML) | Temperature / vector scaling fitted on validation data | Simple, effective | Needs a validation set | **Disagreement-conditioned calibration** |
| 24 | Label shift | Saerens et al. 2002 (Neural Computation); Lipton et al. 2018 (ICML) | Re-estimate class priors on a new population and adjust posteriors | Principled prior correction | Assumes p(x\|y) is unchanged | Disagreed tweets have a different class prior |
| 25 | Imbalance | Menon et al. 2021 — logit adjustment (ICLR) | Add class-prior offsets to logits | Simple, theoretically grounded | — | Prior-aware decision rule |
| 26 | Ensembles | Wolpert 1992 (stacking); hybrid stacked ensembles of Transformers for emotion detection | Meta-learner over base models | Usually beats single models | Needs labelled data for the meta-learner | **Score-average / stacking baselines; equal-vote ensemble (v2)** |
| 27 | Parameter-efficient fine-tuning | Hu et al. 2022 — **LoRA** (ICLR) | Trains low-rank adapters instead of all weights | Fine-tunes large LMs on small GPUs with few labels | Still needs labelled data | **Fine-tuning RoBERTa-large on 500 labels on a 6 GB GPU (v2 member)** |

### 1.6 Research gaps (what nobody, including the base paper, addresses)
* **G1 — Disagreement is used only as a filter.** The paper throws away *how* the models disagree: their confidence, entropy, and the divergence between them. The learning-from-disagreement literature (#10–13) shows this is useful information.
* **G2 — Thin features.** Each tweet has only 7 DistilRoBERTa scores. 23% of phrase nodes on GoEmotions (8,736 of 38,746) have an all-zero feature vector. BertGCN-style work (#20) shows full LM embeddings are much richer.
* **G3 — Train/test mismatch.** Training uses only agreed tweets; testing focuses on disagreed tweets, a different population (more neutral). Zero-feature tweets never appear in training (3A: 0.344 ± 0.228).
* **G4 — Hard, noisy pseudo-labels.** Agreed labels are only 71% (GoEmotions), 72% (Friends) and 43% (TEC) correct, yet they are used as hard targets.
* **G5 — Weak evaluation protocol.** Selection by similarity to RoBERTa on test data, single runs, no trivial baselines. Lv et al. 2021 (#19) show such gains often vanish.
* **G6 — Label-space mismatch.** The LMs always allow *neutral*, but TEC has no neutral class: 19–27% of LM predictions on TEC are wrong by construction.
* **G7 — Sparse graph.** Tweets connect only through exact phrase matches, and attention has a single relation to weigh at tweet nodes (hence attention ≈ mean).

### 1.7 Areas for future research (for the conclusion slide)
* Use LLM rationales/explanations as extra node text.
* Multi-label emotion output instead of a single label.
* Add conversation/thread context (Friends dialogues, Reddit threads).
* Active learning: send only the most uncertain disagreed posts to humans.
* Cross-lingual social media (code-mixed Indian-language tweets).

---

## 2. Handling datasets & data preprocessing (40 marks)

### 2.1 Data sources
| Dataset | Domain | Posts used | Labels | Source | Role |
|---|---|---|---|---|---|
| GoEmotions (Demszky et al., 2020) | Reddit comments | 31,621 (paper: 32,291) | 7 (Ekman + neutral) | Google Research raw rater-level release (`goemotions_{1,2,3}.csv`) | Main; models partly trained on it (in-domain) |
| Friends / EmotionLines (Chen et al., 2018) | TV dialogue | 11,731 (= paper) | 7 | Official EmotionX release (`friends.json`) | Second test set; DistilRoBERTa-v2 was trained on it (leaky for that baseline only) |
| TEC (Mohammad, 2012) | Twitter | 21,051 (paper: 21,047) | 6 (no neutral) | `Jan9-2012-tweets-clean.txt` | Unseen gold test set |

| Statistic | GoEmotions | Friends | TEC |
|---|---|---|---|
| Median words/post | 12 | 6 | 15 |
| Neutral share — all / agreed / disagreed | 70% / 68% / 76% | 56% / 56% / 56% | 0% |
| Agreed-label accuracy (pseudo-label noise) | 0.713 | 0.724 | 0.430 |
| Share of LM predictions that are neutral (Distil / RoBERTa) | 46% / 43% | 40% / 42% | 27% / 19% (all wrong) |

### 2.2 Inconsistencies found and how they were handled
| # | Dataset | Inconsistency | Evidence | Handling |
|---|---|---|---|---|
| 1 | GoEmotions | Data is per-rater (211,225 rows for 58,011 comments), not per comment | raw CSVs | Aggregate per comment id |
| 2 | GoEmotions | 27 fine emotions + neutral vs 7 target classes; multi-label | rater rows | Keep comments where exactly one of the 7 Ekman names was chosen. This reconstructs the paper's subset to within 2% (31,621 vs 32,291) and within 10% for every class |
| 3 | GoEmotions | Anonymisation placeholders `[NAME]`, `[RELIGION]` | text | Removed |
| 4 | Friends | 2,772 utterances labelled `non-neutral` (no majority emotion) | label counts | Removed, which gives exactly the paper's 11,731 |
| 5 | Friends | Windows-1252 mojibake (`\x92` instead of ’) | e.g. "he\x92s lost it" | Re-decoded latin-1 → cp1252 |
| 6 | Friends / GoEmotions | Empty utterances after cleaning (e.g. "…"), 8 + 12 posts | counts | Kept with a placeholder so every post gets a prediction |
| 7 | TEC | 42 duplicate tweet IDs (multi-label tweets listed twice, sometimes with different labels) | id check | Kept, as the corpus defines them; documented as a 4-row difference from the paper |
| 8 | TEC | Residual emotion hashtags (`#JOY`) in 150 tweets; HTML entities (`&lt;`, `&#xA;`) | regex | `#` stripped (the word is kept); entities unescaped |
| 9 | TEC | **No neutral class**, but both LMs predict neutral 19–27% of the time | label sets | Phase 3B: **label-space alignment** (restrict predictions to the dataset's own label set), applied equally to all baselines |
| 10 | All | URLs, @mentions, control characters, extra whitespace | regex | Removed |
| 11 | All | Severe class imbalance (neutral 56–70%; TEC joy 39%) | distributions | Weighted metrics as in the paper, plus macro-F1; prior-aware calibration in 3B |
| 12 | All | Distribution shift between agreed and disagreed sets (neutral 68% → 76% on GoEmotions) | §2.1 | Phase 3B: disagreement-simulation training and disagreement-conditioned calibration |
| 13 | All | Leakage risk: GoEmotions is in the LMs' training data; Friends is in DistilRoBERTa-v2's | model cards | TEC used as the clean test; leakage noted when interpreting results |

### 2.3 Preprocessing pipeline
Raw text → HTML unescape → remove URLs, @mentions, placeholders and control characters → strip `#` → whitespace normalisation → (LM tokenizer for scores and embeddings) / (lower-case, stop-word filter, TextRank for phrases).

### 2.4 Features and why they were chosen
| Feature | Paper | Phase 3B | Justification |
|---|---|---|---|
| Tweet: LM emotion scores | DistilRoBERTa only (7-d) | DistilRoBERTa ⊕ RoBERTa-large (14-d) | Both annotators' opinions; the paper discarded RoBERTa's scores (gap G1) |
| Tweet: disagreement descriptors | — | confidence of each LM, entropy, Jensen–Shannon divergence between the two | Encodes *how* the models disagree (G1; refs #7, #10, #14) |
| Tweet: LM sentence embeddings | — | mean-pooled hidden states of both LMs | Much richer semantics than 7 scores (G2; ref #20) |
| Phrase: statistics | mean ⊕ std of agreed tweets' scores (zero if none) | same + phrase-text embedding | No phrase is left with an empty vector (G2) |
| Emotion: one-hot | ✓ | ✓ | Keeps the classes distinct |
| Edges | tweet–phrase, phrase–emotion (thresholds) | + tweet–tweet semantic k-NN | Denser propagation; attention now chooses between relations (G7) |

### 2.5 Splits and leakage control (Phase 3B)
* Each dataset is split once (stratified): a **20% pool**, from which gold labels may be drawn, and an **80% test** portion. **No design decision, hyperparameter or calibration ever touched the test portion.**
* **Setting A (label-free, the paper's setting):** no gold labels at all. Training targets come only from the LMs.
* **Setting B (few labels, main):** N = 500 gold labels drawn from the pool (GoEmotions 1.6%, Friends 4.3%, TEC 2.4% of the data). **Every baseline gets the same N labels.**
* All design choices were made by prototyping on held-out *pool* data (§3.2a).
* Every result is reported over **5 seeds**, each with its own label draw (mean ± std). Significance: one-sample and paired t-tests over seeds, plus McNemar tests on test posts.

---

## 3. Algorithmic design & approach (40 marks)

### 3.1 Base algorithm (paper, reproduced in 3A)
Eq. 1: $h_u[m] = W[m]\cdot\text{CONCAT}(W_U[m] h_u,\ W_T[m]\cdot\text{mean}_{t\in N_m(u)} h_t)$. Eq. 3–5: mean or semantic attention over message types. The model is trained with cross-entropy on agreed tweets; features are DistilRoBERTa scores.

### 3.2 Proposed model — **DA-HGNN-F: Disagreement-Aware Heterogeneous GNN with teacher Fusion**
Notebook: `Capstone_Phase3B_Improved_Model.ipynb`.

**Pipeline (one slide):**
1. Two LMs (DistilRoBERTa, RoBERTa-large) give scores and sentence embeddings for every post. Agreement defines the agreed and disagreed groups, as in the paper.
2. **Heterogeneous graph:** tweets, phrases (TextRank) and emotions, as in the paper, **plus tweet–tweet semantic k-NN edges**.
3. **DA-HGNN student:** the paper's Eq. 1 convolution and Eq. 4–5 semantic attention, with multi-view features. It is trained **semi-supervised**:
   * confidence-weighted **soft-label distillation** from both LMs on agreed posts (label-free);
   * cross-entropy on a **small gold set (N = 500)**;
   * **score-masking** noise.
4. **Teacher:** the average of the two LMs with **disagreement-conditioned calibration** (separate temperature and bias for agreed vs disagreed posts, fitted on the same N labels), then **graph smoothing** over semantic neighbours.
5. **Fusion:** log p = ½ log p(student) + ½ log p(teacher), then **label-space alignment** (only the classes the dataset defines).

| Component | Fixes gap | Idea | Literature basis |
|---|---|---|---|
| **C1 Multi-view node features** | G1, G2 | Tweet = both LMs' scores + disagreement descriptors (confidence, entropy, JS divergence) + both LMs' sentence embeddings. Phrase = statistics + contextual embedding | BertGCN, TextGCN, Dong 2026 |
| **C2 Soft, confidence-weighted distillation** | G4 | KL to the product of the two LMs' distributions, weighted by agreement confidence | KD (Hinton), Fornaciari 2021, Confident Learning, Snorkel |
| **C3 Disagreement simulation (score masking)** | G3 | Hide the LM scores of 30% of agreed training posts, so the model must use text and graph, as it must on disagreed posts | UniMP, DropEdge, Noisy Student |
| **C4 Semantic tweet–tweet relation** | G7 | k = 10 nearest neighbours on embeddings; attention chooses between phrase→tweet and tweet→tweet | TextGCN/BertGCN, HAN |
| **C5 Disagreement-conditioned calibration** | G3 | Vector scaling fitted separately per group on N gold labels | Guo 2017, Saerens 2002, Menon 2021 |
| **C6 Label-space alignment** | G6 | No neutral on TEC; applied to all baselines too | Task definition |
| **C7 Small gold set (semi-supervised)** | G4 | N = 500 labels (1.6–4.3% of the data) in the loss; all baselines get the same labels | Lee 2013, Noisy Student, C&S |
| **C8 Student–teacher fusion + graph smoothing** | G7 | Combine the GNN (strong where the LMs are weak) with the calibrated teacher (strong in-domain) | C&S (Huang 2021), BertGCN interpolation |
| **C9 Honest protocol** | G5 | 20% pool / 80% untouched test, design fixed on the pool, 5 seeds, ablations, t-tests + McNemar | Lv et al. 2021 |

### 3.2a Design log: how prototyping on dev data shaped the model (a good "approach" slide)
All numbers are on the held-out half of each dataset's dev pool, never the test split.

| Round | Question | Finding | Decision |
|---|---|---|---|
| 1 | Do richer features or soft labels make a label-free student beat the LMs? (GoEmotions) | No: every student ≈ score average (0.64 total accuracy). **Calibration** lifts every model to 0.80 | Calibration is essential; a student that only imitates the LMs cannot beat them |
| 2 | Calibrated teacher vs students vs graph smoothing (3 datasets) | All calibrated variants within ±0.01. Graph smoothing gives the best F1 on GoEmotions and Friends | Keep graph smoothing |
| 3 | Student/teacher blend; smoothing strength (3 seeds) | Differences ≤ 0.005; α = 0.5 best on average | α = 0.5 |
| 4 | **Semi-supervised** DA-HGNN (distillation + gold) vs calibration vs supervised MLP, N = 200/500/all | The graph beats the same model without a graph by +3–4 accuracy points at N = 200. DA-HGNN beats the calibrated teacher by **+3–7 points on TEC** (unseen domain) | Train DA-HGNN semi-supervised |
| 5 | Logistic-regression "resolver" on graph features | Weaker than calibration at N = 200; competitive only at N ≥ 500 | Dropped |
| 6 | Fuse DA-HGNN with the calibrated teacher (w = 0.3 / 0.5 / 0.7) | w = 0.5 best on average at N = 200 and 500 (acc .684/.687, F1 .677/.680 vs teacher .665/.677) | **Final: DA-HGNN-F, w = 0.5** |

### 3.3 Test cases (experimental design)
* **Datasets:** GoEmotions (in-domain for the LMs), Friends (conversational; DistilRoBERTa-v2 was trained on it), TEC (unseen, 6 classes). Each is evaluated on its untouched 80% test split.
* **Subsets:** compliance (C), non-compliance (NC) and total (T), as in the paper. Metrics: accuracy, weighted F1 (paper's metric) and macro-F1.
* **Baselines (16):**
  * raw: DistilRoBERTa, RoBERTa-large, DistilRoBERTa-v2, score average, "disagree → neutral" rule;
  * the **paper's HGNN-Max and HGNN-Min, from our 3A replication**;
  * label-free: aligned, smoothed and student variants;
  * few-label, all with the same N labels: calibrated RoBERTa, calibrated score average (± smoothing), supervised MLP, MLP semi (no graph), DA-HGNN semi.
* **Ablations:** 15 variants × 3 seeds × 3 datasets. **Label budget:** N = 50 … 2000 × 3 draws. **Seeds:** 5. **Tests:** t-tests and McNemar.

### 3.4 Results (from `Capstone_Phase3B_Improved_Model.ipynb`, figures in `results/improved/`)

**Main result.** Setting B, N = 500 gold labels, mean of 5 seeds, untouched 80% test split. Total accuracy / weighted F1:

| Dataset | Paper best HGNN (reported) | Paper HGNN-Max (our replication, same split) | RoBERTa-large | Calibrated LM ensemble (strongest simple baseline) | **DA-HGNN-F (ours)** | **Gain vs paper** |
|---|---|---|---|---|---|---|
| GoEmotions | .737 / .738 | .666 / .696 | .616 / .652 | .784 / .784 | **.788 / .785** | **+5.1 / +4.7** |
| Friends | .535 / .578 | .635 / .644 | .650 / .663 | .716 / .709 | **.719 / .717** | **+18.4 / +13.9** |
| TEC (unseen) | .436 / .443 | .405 / .432 | .392 / .437 | .482 / .468 | **.517 / .518** | **+8.1 / +7.5** |

* **Disagreed posts (NC), accuracy:** GoEmotions .755 (RoBERTa .356; paper replication .538), Friends .638 (RoBERTa .464), TEC .459 (RoBERTa .318).
* **Significance:** vs the paper's HGNN and both raw LMs, p < 0.001 on every dataset (t-test, 5 seeds); McNemar p < 10⁻¹⁰ for every seed. Vs the calibrated LM ensemble: significant on TEC (+3.4 / +5.1, p < 0.001), on GoEmotions accuracy (+0.5, p = 0.006) and on Friends F1 (+0.7, p = 0.01).

**Ablation.** Mean change in total accuracy / F1 when a component is removed (3 datasets × 3 seeds):

| Removed component | Δ acc | Δ F1 | Note |
|---|---|---|---|
| Calibration of the LM teacher (fitted on the 500 labels) | −3.6 | −2.3 | largest effect; per-group vs global ≈ 0 |
| Multi-view features → paper's DistilRoBERTa-only features | −1.6 | −2.3 | |
| DA-HGNN student (teacher only) | −1.2 | −1.8 | TEC −3.2 / −4.8 |
| LM sentence embeddings | −1.4 | −1.2 | |
| Fusion (student alone) | −0.8 | −0.5 | |
| Whole graph (MLP instead) | −0.4 | −0.5 | TEC −1.1 / −1.4 |
| Soft targets | −0.3 | −0.3 | |
| k-NN relation alone, phrase relations alone, score masking, confidence weighting, per-group calibration, smoothing | ≈ 0 | ≈ 0 | relations are interchangeable |
| Distillation from LMs | +0.8 | +0.2 | helps GoEmotions (−0.9 when removed), **hurts TEC** (+2.4 when removed) |

**Label efficiency** (total accuracy, DA-HGNN-F vs supervised MLP on the same labels):

| N labels | GoEmotions | Friends | TEC |
|---|---|---|---|
| 50 | **.770** vs .711 | **.704** vs .684 | **.475** vs .446 |
| 200 | **.779** vs .749 | **.706** vs .690 | **.507** vs .497 |
| 500 | **.788** vs .770 | **.717** vs .714 | .515 vs **.534** |
| 2000 | .788 vs **.801** | .727 vs **.738** | .536 vs **.584** |

**Interpretability.**
* First-layer semantic attention at tweet nodes is decisive: Friends puts 0.999 on tweet→tweet, GoEmotions 0.84, TEC 0.999 on phrase→tweet (the paper's attention was ≈ 0.5 / 0.5).
* Example (TEC): "Another early morning tomorrow and a long drive home". DistilRoBERTa = neutral, RoBERTa = fear, **DA-HGNN-F = joy (gold)**.

### 3.4b Phase 3C — room for improvement → **DA-HGNN-E (v2)**
Notebook: `Capstone_Phase3C_Improvements.ipynb`. Results: `results/improved_v2/`.

**Protocol.** Candidates were judged only on the *held-out pool* (gold-labelled pool posts not among the 500 training labels). A selection rule was fixed in advance: best mean of (accuracy + F1)/2 over the datasets and 5 seeds. The test split was then used **once**.

**What we tried** (held-out pool, mean of (acc + F1)/2 over the 3 datasets):

| Candidate | Score | Outcome |
|---|---|---|
| v1 DA-HGNN-F (graph student ⊕ calibrated teacher) | .677 | starting point |
| + supervised MLP as a third equal vote | .687 | ✅ helps on all datasets |
| **Fine-tune the LMs on the 500 labels** (RoBERTa-large with LoRA; DistilRoBERTa fully) | .681 (RoBERTa alone) | ✅ strong new baseline and ensemble member |
| **5-way equal vote: graph student + teacher + supervised MLP + both fine-tuned LMs** | **.694** | ✅ **selected** |
| Same without the graph student and DistilRoBERTa (teacher + MLP + FT-RoBERTa) | .690 | the graph adds little inside the ensemble |
| Correct & Smooth with gold labels | .617 | ❌ hurts |
| Self-training (Noisy Student), GoEmotions + Friends, 1 seed | +0.3 to +1.2 over the 5-way | ⚠️ promising, but too slow and crash-prone on a 6 GB GPU → future work |
| Oracle (any of the 5 members right) | .822 accuracy | large remaining headroom |

**Confirmatory test result** (5 seeds; total accuracy / F1):

| Dataset | Paper best (reported) | v1 DA-HGNN-F | Fine-tuned RoBERTa-large (LoRA) | **v2 DA-HGNN-E** | **v2 − paper** |
|---|---|---|---|---|---|
| GoEmotions | .737 / .738 | .788 / .785 | .771 / .774 | **.801 / .795** | **+6.4 / +5.7** |
| Friends | .535 / .578 | .719 / .717 | .713 / .708 | **.732 / .724** | **+19.7 / +14.6** |
| TEC | .436 / .443 | .517 / .518 | .546 / .541 | **.552 / .546** | **+11.6 / +10.4** |

* **v2 vs v1:** +1.3 / +1.3 / +3.6 accuracy points (p ≤ .002 on all three datasets).
* **v2 vs fine-tuned RoBERTa:** significantly better on GoEmotions and Friends; a tie on TEC (p = .36).
* **No selection overfitting:** held-out score .694 vs test .692.
* **Leave-one-member-out on test:** the gain comes from the fine-tuned LMs (−0.3 to −0.5 points when removed) and the supervised MLP (−0.3). Inside v2, the graph student and the calibrated teacher are within noise (±0.3).

### 3.5 Discussion (what to say honestly in the viva)
* **The paper is beaten on every dataset**, by 5–18 accuracy points, with significance tests and a protocol that never touched the test split.
* **Calibrating the LMs with the few gold labels** is the single biggest lever (−3.6 points if removed). Disagreed posts are a different population, which motivated fitting it per agreement group; inside the fused model, though, per-group vs one global calibration makes no measurable difference.
* **The graph student matters most where the LMs are weakest:** the unseen TEC, and small label budgets. On GoEmotions, which the LMs were trained on, it only ties the calibrated ensemble.
* **Limitations:**
  * With ≥ 1,000 labels, a plain supervised classifier overtakes DA-HGNN-F.
  * On TEC at N = 500, distillation from noisy LM labels (43% correct) holds it back.
  * Label-free, we do not beat simple averaging except via label-space alignment.
* **Phase 3C lesson:** with 500 labels, **fine-tuning the LMs themselves** (LoRA) is a stronger lever than the graph architecture. The best system (v2) is an equal-vote ensemble of diverse few-label members. Inside it the graph student is within noise. Its value lies in the label-free and very-few-label regimes, and in robustness on the unseen domain (v1).
* **Future work:** a learned combiner (oracle headroom 0.82 vs 0.69), self-training on stronger hardware, a domain-adaptive distillation strength, active learning of which disagreed posts to label, and multi-label output.

---

## 4. Presentation & interaction (30 marks)

### 4.1 Slide deck (built): `Capstone_Phase3_Slides.pptx` — 6 slides (slide limit), same visual rules as the Phase 2 deck
1. Title (CAPSTONE PROJECT — PHASE 3 · PSID - 181)
2. Problem Identification & Literature Survey: problem in numbers + 6-area survey table (works / offers / gap) + example post
3. Gaps in the Base Paper — Replication (3A): check / paper / ours / gap revealed (G1–G7) + root cause
4. Handling Datasets and Data Preprocessing: dataset table, inconsistencies → handling, features extracted, protocol
5. Algorithmic Design and Approach: 4-step pipeline, evidence (ablation, prototyping), room for improvement (3C)
6. Results — Beating the Base Paper: chart (paper / replication / fine-tuned RoBERTa / v1 / v2) + key results + take-away

Speaker notes on every slide carry the extra detail (and the 16-slide draft generator is kept in `dev/slides/`).

### 4.2 Figures available
* `results/replication/fig6-8_scatter_goemotions.png`: 180 configurations; on disagreed posts they scatter from 0.05 to 0.56 → instability
* `results/replication/fig5_hyperparameters.png`: hidden size × layers
* `results/replication/fig_phrase_emotion_edges.png`: graph statistics
* `results/improved/fig_main_comparison.png`: paper vs replication vs ours, 3 datasets (**key slide**)
* `results/improved/fig_label_budget.png`: accuracy/F1 vs number of labels (**key slide**)
* `results/improved/fig_ablation.png`: contribution of each component
* `results/improved/fig_confusion_disagreed.png`: confusion matrices on disagreed posts
* `results/improved/fig_attention.png`: learned relation attention at tweet nodes
* Tables: `results/improved/comparison_with_paper.csv`, `main_results_table.csv`, `ablation_table.csv`, `significance_tests.csv`, `headline_gains.csv`

### 4.3 Key one-liners
* "Two strong models disagree on a third of all posts. That disagreement is information, not noise."
* "We reproduced the paper's baselines within 0.02. Its best HGNN result, however, depends on a lucky seed."
* "A one-line rule matches the paper's best model, so any new model must beat that rule."

### 4.4 Anticipated viva questions
| Question | Answer |
|---|---|
| Why not simply fine-tune a new model on gold labels? | The setting assumes no new annotation; we use only the LMs' agreement. Gold labels are used only for evaluation and a small dev set. |
| Why is TEC important? | Neither LM was trained on it, so it is the only fair test of generalisation. |
| Isn't removing neutral on TEC cheating? | The label set is part of the task definition, not of the labels. We apply it to every baseline too, and report the results both with and without it. |
| Why did the paper's HGNN-Min not reproduce? | The paper does not specify its grid or seeds. Our 10-seed study shows that configuration family varies between 0.05 and 0.61. |
| What does the graph add over an MLP? | Measured cleanly by the ablation: inside the final fusion, replacing the graph student with an MLP of the same features and labels costs 0.4 / 0.5 points on average and 1.1 / 1.4 on TEC. In prototyping at N = 200, the graph student beat the no-graph student by 3–4 accuracy points. Its two relation types (phrase graph, semantic neighbours) are interchangeable. Against the no-graph MLP semi baseline, the full DA-HGNN-F is ahead by +3.5 / +3.7 on GoEmotions and +1.8 accuracy on TEC (paired t, p < 0.05). |
| Isn't using 500 gold labels unfair to the paper? | We report both settings. Every baseline gets the same 500 labels, and 500 is only 1.6–4.3% of each dataset. The label-budget curve shows gains even at N = 50. The paper also tuned on GoEmotions test results, which we never do. |
| Why does a supervised MLP win on TEC at N = 500? | On TEC the LMs' agreed labels are only 43% correct, so distilling from them adds noise. The ablation confirms that removing distillation helps on TEC (+2.4) but hurts on GoEmotions (−0.9). A domain-adaptive distillation weight is our future work. |
| Why is calibration so powerful? | Disagreed posts have a different label distribution (GoEmotions: 76% neutral vs 68% in agreed posts). Fitting a temperature and per-class bias on the few labels corrects the LMs' systematic under-prediction of neutral (they predict neutral for only 43–46% of posts against 70% in the gold labels). |
| Is the improvement statistically significant? | Yes, against the paper's HGNN and both LMs on all datasets: p < 0.001 (5 seeds), McNemar p < 10⁻¹⁰. |
| Why not just fine-tune RoBERTa on the 500 labels? | We did (LoRA). It is a strong baseline that already beats the paper. v2 is significantly better than it on GoEmotions (+3.0) and Friends (+1.8), and ties it on TEC. |
| How did you avoid overfitting while searching for improvements? | Every candidate was scored on held-out pool data, never the test split. The selection rule was written down before the evaluation, and the test split was used once. Held-out .694 vs test .692 shows no overfitting. |
| Does the graph still matter in v2? | Honestly, very little inside the ensemble (±0.3 points). It matters in v1, with very few or no labels, and on the unseen TEC. |

---

## References
1. Maazallahi, A., Asadpour, M., & Bazmi, P. (2025). Advancing emotion recognition in social media: A novel integration of heterogeneous neural networks with fine-tuned language models. *Information Processing & Management*, 62, 103974.
2. Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A dataset of fine-grained emotions. *ACL 2020*.
3. Mohammad, S. M. (2012). #Emotional tweets. *\*SEM 2012*.
4. Chen, S.-Y., Hsu, C.-C., Kuo, C.-C., Huang, T.-H. K., & Ku, L.-W. (2018). EmotionLines: An emotion corpus of multi-party conversations. *LREC 2018*.
5. Hartmann, J., Heitmann, M., Siebert, C., & Schamp, C. (2023). More than a feeling: Accuracy and application of sentiment analysis. *International Journal of Research in Marketing*, 40(1).
6. Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv:1907.11692.
7. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
8. Alhuzali, H., & Ananiadou, S. (2021). SpanEmo: Casting multi-label emotion classification as span-prediction. *EACL 2021*.
9. Dong, H., Bao, Z., Li, M., & Yang, Z. (2026). Emotion meets coordination: Designing multi-agent LLMs for fine-grained user sentiment detection on social media. *PLOS ONE*, 21(2), e0342053.
10. Dawid, A. P., & Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates using the EM algorithm. *Journal of the Royal Statistical Society: Series C*, 28(1).
11. Ratner, A., Bach, S. H., Ehrenberg, H., Fries, J., Wu, S., & Ré, C. (2017). Snorkel: Rapid training data creation with weak supervision. *Proceedings of the VLDB Endowment*, 11(3).
12. Uma, A. N., Fornaciari, T., Hovy, D., Paun, S., Plank, B., & Poesio, M. (2021). Learning from disagreement: A survey. *Journal of Artificial Intelligence Research*, 72.
13. Fornaciari, T., Uma, A., Paun, S., Plank, B., Hovy, D., & Poesio, M. (2021). Beyond black & white: Leveraging annotator disagreement via soft-label multi-task learning. *NAACL 2021*.
14. Plank, B. (2022). The "problem" of human label variation: On ground truth in data, modeling and evaluation. *EMNLP 2022*.
15. Xu, J., Theune, M., & Braun, D. (2024). Leveraging annotator disagreement for text classification. arXiv:2409.17577.
16. Northcutt, C., Jiang, L., & Chuang, I. (2021). Confident learning: Estimating uncertainty in dataset labels. *Journal of Artificial Intelligence Research*, 70.
17. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv:1503.02531.
18. Lee, D.-H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. *ICML 2013 Workshop on Challenges in Representation Learning*.
19. Xie, Q., Luong, M.-T., Hovy, E., & Le, Q. V. (2020). Self-training with Noisy Student improves ImageNet classification. *CVPR 2020*.
20. Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing order into texts. *EMNLP 2004*.
21. Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS 2017*.
22. Wang, X., Ji, H., Shi, C., Wang, B., Ye, Y., Cui, P., & Yu, P. S. (2019). Heterogeneous graph attention network. *WWW 2019*.
23. Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., & Welling, M. (2018). Modeling relational data with graph convolutional networks. *ESWC 2018*.
24. Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). Heterogeneous graph transformer. *WWW 2020*.
25. Lv, Q., et al. (2021). Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous graph neural networks. *KDD 2021*.
26. Yao, L., Mao, C., & Luo, Y. (2019). Graph convolutional networks for text classification. *AAAI 2019*.
27. Lin, Y., Meng, Y., Sun, X., Han, Q., Kuang, K., Li, J., & Wu, F. (2021). BertGCN: Transductive text classification by combining GCN and BERT. *Findings of ACL-IJCNLP 2021*.
28. Huang, Q., He, H., Singh, A., Lim, S.-N., & Benson, A. R. (2021). Combining label propagation and simple models out-performs graph neural networks. *ICLR 2021*.
29. Shi, Y., Huang, Z., Feng, S., Zhong, H., Wang, W., & Sun, Y. (2021). Masked label prediction: Unified message passing model for semi-supervised classification. *IJCAI 2021*.
30. Rong, Y., Huang, W., Xu, T., & Huang, J. (2020). DropEdge: Towards deep graph convolutional networks on node classification. *ICLR 2020*.
31. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*.
32. Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure. *Neural Computation*, 14(1).
33. Lipton, Z., Wang, Y.-X., & Smola, A. (2018). Detecting and correcting for label shift with black box predictors. *ICML 2018*.
34. Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S. (2021). Long-tail learning via logit adjustment. *ICLR 2021*.
35. Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2).
36. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP-IJCNLP 2019*.
37. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.

## 15. Conclusions

**Headline — Setting B (N = 500 gold labels per dataset), 5 seeds, untouched 80% test split. Total accuracy / weighted F1:**

| Dataset | Paper's best HGNN (reported) | Paper HGNN-Max (our replication, same split) | RoBERTa-large | **DA-HGNN-F (ours)** | Gain vs paper (reported) |
|---|---|---|---|---|---|
| GoEmotions | .737 / .738 | .666 / .696 | .616 / .652 | **.788 / .785** | **+5.1 / +4.7 pts** |
| Friends | .535 / .578 | .635 / .644 | .650 / .663 | **.719 / .717** | **+18.4 / +13.9 pts** |
| TEC (unseen) | .436 / .443 | .405 / .432 | .392 / .437 | **.517 / .518** | **+8.1 / +7.5 pts** |

Against the paper's HGNN (our replication) and both raw LMs, every gain is significant on every dataset: one-sample t over 5 seeds, p < 0.001; McNemar against RoBERTa-large and the paper's HGNN, p < 10⁻¹⁰ for every seed.

**What made the difference (ablation, §10; mean change in total accuracy / F1 when a component is removed)**
* Calibration of the LM teacher on the N labels: −3.6 / −2.3 points, the largest effect (fitting it per agreement group vs globally makes no measurable difference inside the fused model).
* Multi-view features (both LMs, disagreement descriptors, embeddings) vs the paper's DistilRoBERTa-only features: −1.6 / −2.3.
* The DA-HGNN student inside the fusion: −1.2 / −1.8 on average, and −3.2 / −4.8 on the unseen TEC.
* LM sentence embeddings: −1.4 / −1.2. The heterogeneous graph as a whole: −0.4 / −0.5 (TEC −1.1 / −1.4).
* Semantic attention now does real work. At the first layer, tweet nodes rely almost entirely on semantic neighbours (Friends 0.999, GoEmotions 0.84) or on phrases (TEC 0.999), unlike the paper's ≈ 0.5 / 0.5.

**What did not help (honest negatives)**
* No measurable effect: score masking, confidence weighting, per-group vs global calibration inside the fused model, and teacher smoothing on the test split.
* The tweet–tweet and tweet–phrase relations are **interchangeable**: removing either one alone changes nothing, while removing both (no graph) hurts.
* **GoEmotions:** DA-HGNN-F only ties the calibrated LM ensemble (+0.5 accuracy points, p = 0.006; F1 not significant). The LMs were trained on GoEmotions, so calibration already captures most of what can be gained.
* **TEC at N = 500:** a plain supervised MLP on the LM embeddings is better (.537 vs .517). The LMs' agreed labels are only 43% correct on TEC, so distilling from them holds the model back (ablation "− distillation": +2.4 accuracy on TEC).
* **Label-free setting:** no gain over simply averaging the two LMs, except through label-space alignment on TEC (.452 vs the paper's .436).

**Label efficiency (§11).** DA-HGNN-F is the best or tied-best method for **N ≤ 200–500 labels** on all three datasets. With 50 labels, for example, it beats a supervised model by +5.9 points on GoEmotions, +2.0 on Friends and +2.9 on TEC. With ≥ 1,000 labels, ordinary supervised learning catches up and overtakes it.

**Take-away.** The decisive idea is to treat **disagreement between the two LMs as information**: calibrate disagreed posts separately and give the model a small amount of supervision. The heterogeneous graph student adds robustness when labels are scarce and the domain is new to the LMs.

**Future work:** learn the fusion weight (or switch off distillation) per domain from the gold set; use active learning to pick which disagreed posts to label; multi-label output; conversation context for Friends/Reddit.

## 7. Replication verdict

**Reproduced**
* **Datasets (Table 1):** Friends is identical (11,731 utterances, same counts per class). TEC has 21,051 tweets vs 21,047. GoEmotions has 31,621 posts vs 32,291; the paper does not describe its subset, and this is the closest reconstruction.
* **Compliance split (Table 4):** within 2 percentage points on all three datasets.
* **Language-model baselines (Tables 5–6):** RoBERTa-large, DistilRoBERTa and DistilRoBERTa-v2 are within about 0.02 of the paper in almost every cell, on all three datasets.
* **HGNN-Max on GoEmotions:** within 0.01 of the paper on every accuracy and F1 cell (ours .713 / .536 / .665 accuracy vs the paper's .720 / .530 / .671). On TEC, HGNN-Max is close, slightly above the paper.

**Not reproduced**
* **HGNN-Min**, the paper's best GoEmotions result (0.677 accuracy on non-compliance posts). Our least RoBERTa-like configuration reaches 0.397.
* **Table 3 (attention > mean):** across 30 matched configurations, attention − mean = +0.010 F1 (p = 0.70). Averaged over configurations, both are below the language models, not above them.
* **Table 7 (significant gains):** retrained over 10 seeds, no HGNN configuration is significantly better than RoBERTa-large or DistilRoBERTa on any dataset. The only significant wins are over the Friends-specific DistilRoBERTa-v2 on TEC.
* **Friends:** the paper's HGNN models score *below* the baselines even on the compliance set (0.652 and 0.550 vs 0.703), which means they lose accuracy on posts they were trained on. Ours stay at the baseline level.

**Why — two findings**
1. **Zero-feature configurations are unstable.** The paper's second option gives non-agreed tweets zero features (§2.2.3). The model never sees a zero-feature tweet during training, so its predictions for them are essentially arbitrary. Retrained with 10 seeds, the HGNN-Max configuration (mean, hidden 32, 3 layers, zero) reaches **0.344 ± 0.228** accuracy on GoEmotions non-compliance posts, ranging from 0.048 to 0.611. The grid run that matches the paper is one draw from this range. Validation loss cannot detect this, because every validation tweet is an agreed tweet. That is why the val-selected model collapses (0.05).
2. **The non-compliance set is mostly neutral.** In the GoEmotions non-compliance set, 75.9% of the gold labels are *neutral*. A rule with no training — *keep the agreed label, answer neutral on disagreement* — scores **0.759** accuracy on non-compliance posts and **0.740** total weighted F1. That is at or above the paper's best HGNN (0.677 and 0.738). Gains on the non-compliance set should therefore be judged against this rule, not only against the language models. (On TEC, which has no neutral class, the rule scores 0 on the non-compliance set.)

**What this means for Phase 3B (improving accuracy)**
* Train the model on inputs that look like the disagreed tweets it will be tested on. For example, randomly zero or mix the features of agreed tweets during training, so the model learns to use phrase evidence.
* Use all the evidence: both models' scores (concatenation), with DistilRoBERTa-v2 as a third opinion.
* Choose models with a validation set that resembles the disagreement set, not only agreed tweets.
* Always report the "disagree → neutral" rule and RoBERTa-large as the bars to beat, with multiple seeds.

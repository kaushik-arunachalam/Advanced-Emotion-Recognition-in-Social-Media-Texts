## 7. Conclusions (Phase 3C)

**Confirmatory test result.** Untouched 80% test split, 5 seeds, the same 500 labels for every method. Total accuracy / weighted F1:

| Dataset | Paper best HGNN (reported) | v1 DA-HGNN-F (3B) | **v2 DA-HGNN-E** | v2 − v1 (paired t) | **v2 − paper** |
|---|---|---|---|---|---|
| GoEmotions | .737 / .738 | .788 / .785 | **.801 / .795** | +1.3 / +1.0 (p = .002 / .02) | **+6.4 / +5.7** |
| Friends | .535 / .578 | .719 / .717 | **.732 / .724** | +1.3 / +0.7 (p = .002 / .05) | **+19.7 / +14.6** |
| TEC (unseen) | .436 / .443 | .517 / .518 | **.552 / .546** | +3.6 / +2.9 (p = .002 / .002) | **+11.6 / +10.4** |

* **The selection held up.** The selected ensemble scored 0.694 on the held-out pool and 0.692 on the test split, so choosing on held-out data did not overfit.
* **v2 is the best few-label system on GoEmotions and Friends, and tied-best on TEC.** Paired t-tests over 5 seeds:
  * vs v1: accuracy significantly better on all three datasets (p ≤ .002); F1 significant on GoEmotions and TEC, borderline on Friends (p = .051);
  * vs the calibrated teacher: significant on every dataset and metric;
  * vs fine-tuned RoBERTa-large: significant on GoEmotions and Friends; **tie on TEC** (+0.7 points, p = 0.36);
  * vs fine-tuned DistilRoBERTa: significant except GoEmotions accuracy (p = 0.07);
  * vs the supervised MLP: accuracy significant everywhere; F1 significant on TEC only (GoEmotions/Friends p = 0.06–0.07).
* **A strong new baseline.** Simply fine-tuning RoBERTa-large with LoRA on the 500 labels already beats the paper by +3.4 / +17.8 / +11.0 accuracy points. With a few labels, *what you train on* matters more than the architecture.

**Where the extra gain comes from** (leave-one-member-out on test; change in accuracy / F1 points when the member is removed):
* fine-tuned DistilRoBERTa −0.48 / −0.35, fine-tuned RoBERTa-large −0.31 / −0.35, supervised MLP −0.32 / −0.16: **these carry the gain**;
* DA-HGNN student +0.29 / +0.02, calibrated teacher +0.20 / +0.20: **within noise inside v2**.

*Honest reading:* once two fine-tuned LMs are in the ensemble, the graph student and the calibrated teacher no longer add measurable accuracy. Their value lies in v1, in the label-free and very-few-label regimes, and in robustness on the unseen TEC (3B).

**Negative / exploratory results**
* Correct & Smooth with gold labels: −7.6 points on the held-out pool, dropped.
* Self-training (Noisy Student) looked promising (+0.3 to +1.2 points on top of the 5-way ensemble, GoEmotions and Friends, 1 seed). It is too slow and crash-prone on a 6 GB laptop GPU to evaluate properly, so it is left as future work.

**Remaining headroom.** The oracle over the 5 members reaches 0.822 accuracy on held-out data vs 0.694 for the equal-weight vote. Next steps: a learned combiner (stacking with cross-fitting), self-training on stronger hardware, and active learning of which disagreed posts to label.

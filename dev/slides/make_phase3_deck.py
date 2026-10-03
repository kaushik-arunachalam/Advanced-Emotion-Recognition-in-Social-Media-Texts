# Phase 3 deck — 6 slides (slide limit), same visual rules as Capstone_Phase2_Slides.pptx.
# Usage: python make_phase3_deck.py <output.pptx>
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deck_style import *   # primitives, palette and the Presentation object `prs`

OUT = sys.argv[1] if len(sys.argv) > 1 else 'Capstone_Phase3_Slides.pptx'
B = {'bold': True, 'color': BLACK}

# ================================================================ 1. Title
s = new_slide()
text(s, X0, 1.55, W, 0.40, ['CAPSTONE PROJECT — PHASE 3', '', 'PSID - 181'], size=14, color=BLACK, spc=200)
line(s, X0, 2.55, X0 + W, 2.55)
text(s, X0, 2.85, W, 0.90, 'Advancing Emotion Recognition in Social Media', font='Cambria', size=34, bold=True, color=BLACK)
text(s, X0, 3.65, W, 0.70, 'A Novel Integration of Heterogeneous Neural Networks with Fine-Tuned Language Models', size=16, italic=True, color=DARK)
text(s, X0, 4.45, W, 0.40, 'Problem Identification & Literature Survey  •  Datasets & Preprocessing  •  Algorithmic Design & Results',
     size=13, color=GREY)
text(s, X0, 6.35, W, 0.35, 'M A KAUSHIK', size=14, bold=True, color=BLACK)
text(s, X0, 6.70, W, 0.35, 'CH.SC.U4CSE24123', size=12, color=BLACK)
notes(s, 'Phase 3: we replicated the base paper, found why its best results are unstable, and built an improved model that beats it on all three datasets.')

# ================================================================ 2. Problem identification & literature survey
s = new_slide(); page(s)
title(s, 'Problem Identification & Literature Survey',
      'Two fine-tuned language models disagree on 27–33% of posts, and on those posts each is right only 14–46% of the time — which label do we trust?')
table(s, X0, 1.85, [3.95, 3.70, 4.28], [
    ['Area — key works', 'What it offers', 'Gap for our problem'],
    [[[('Fine-tuned emotion LMs', B)], 'Hartmann et al. (2023); Liu et al. (2019) RoBERTa'], 'Strong off-the-shelf 7-class emotion models', 'Black boxes that disagree with each other'],
    [[[('Learning from disagreement', B)], 'Uma et al. (2021); Fornaciari et al. (2021); Plank (2022)'], 'Disagreement is signal; soft-label training', 'Studied for human annotators, never LM vs LM'],
    [[[('Weak supervision & distillation', B)], 'Ratner et al. (2017); Hinton et al. (2015); Xie et al. (2020)'], 'Learning from noisy / soft labels', 'The student copies the teacher\'s bias'],
    [[[('Graphs for text', B)], 'TextRank (2004); TextGCN (2019); BertGCN (2021); HAN (2019)'], 'Relational context; LM features inside GNNs', 'Sparse phrase graphs; attention unused with one relation'],
    [[[('Benchmarking & calibration', B)], 'Lv et al. (2021); Huang et al. (2021); Guo et al. (2017)'], 'Simple, calibrated baselines rival GNNs', 'Many HGNN gains vanish under fair tuning'],
    [[[('Recent LLM work & PEFT', B)], 'Dong et al. (2026) multi-agent LLMs; Hu et al. (2022) LoRA'], 'Explicit conflict resolution; cheap fine-tuning', '70B-scale models; still need labels'],
], [0.45] + [0.62] * 6, body_size=11.5)
text(s, X0, 6.42, W, 0.45, [[('Example (TEC): ', B), ('“Another early morning tomorrow and a long drive home.”  DistilRoBERTa → neutral, RoBERTa-large → fear, '
                                                     'human → joy. Our model → joy.', {})]], size=12.5)
notes(s, '27 works surveyed in total (full table in the review record). The learning-from-disagreement literature says disagreement carries information; '
         'Lv et al. 2021 motivated our strict evaluation protocol. Disagreement: GoEmotions 27%, Friends 28%, TEC 33%.')

# ================================================================ 3. Gaps in the base paper (replication)
s = new_slide(); page(s)
title(s, 'Gaps in the Base Paper — Replication (Phase 3A)',
      ' Maazallahi, Asadpour & Bazmi (2025) — HGNN re-implemented; all 180 grid configurations trained on 3 datasets',
      sub_bold_prefix='Base paper:')
table(s, X0, 1.85, [3.45, 2.30, 2.70, 3.48], [
    ['Check', 'Paper', 'Our replication', 'Gap it reveals'],
    ['Agreement split; LM baselines', 'Tables 4–6', 'within ±0.02', 'Reproduced'],
    ['HGNN-Max, GoEmotions (agreed / disagreed / total)', '.720 / .530 / .671', '.713 / .536 / .665', 'Reproduced — but in one run only'],
    ['HGNN-Max over 10 seeds (disagreed accuracy)', 'not reported', '0.344 ± 0.228', 'G3  Train/test mismatch: zero features never seen in training'],
    ['HGNN-Min, paper\'s best (disagreed accuracy)', '.677', '.397', 'Not reproducible'],
    ['Attention better than mean aggregation', '.56 vs .50 F1', '+0.010 (p = 0.70)', 'G7  Sparse graph: one relation per tweet'],
    ['Rule "models disagree → neutral" (no training)', 'not reported', '.759 / .740', 'G1  Disagreement used only as a filter'],
    ['Tweet features; model selection', '7 DistilRoBERTa scores; best-of-grid on test', 'agreed labels only 43–72% correct', 'G2 thin features · G4 noisy labels · G5 weak protocol'],
], [0.45, 0.50, 0.50, 0.55, 0.50, 0.50, 0.50, 0.55], body_size=11.5)
text(s, X0, 6.20, W, 0.55, [[('Root cause: ', B), ('disagreed posts are a different population (76% neutral on GoEmotions; TEC has no neutral at all — G6), '
                                                  'and the paper\'s best results depend on the random seed.', {})]], size=12.5)
notes(s, 'Everything except the paper\'s headline HGNN-Min result reproduced. Each failure points to a research gap G1-G7 that the new model addresses.')

# ================================================================ 4. Datasets & preprocessing
s = new_slide(); page(s)
title(s, 'Handling Datasets and Data Preprocessing',
      'Pipeline: raw text → unescape → remove URLs / @mentions / placeholders → strip # → normalise → LM tokenizer | TextRank phrases')
table(s, X0, 1.80, [2.30, 2.05, 1.30, 1.95, 4.33], [
    ['Dataset', 'Domain', 'Posts', 'Classes', 'Role'],
    ['GoEmotions', 'Reddit comments', '31,621', '7', 'Main; inside the LMs\' training data (in-domain)'],
    ['Friends (EmotionLines)', 'TV dialogue', '11,731', '7', 'Conversational; identical to the paper'],
    ['TEC', 'Tweets', '21,051', '6 (no neutral)', 'Unseen by every model — the clean test'],
], [0.40, 0.40, 0.40, 0.40], body_size=12)
heading(s, X0, 3.65, COLW, 'Inconsistencies → Handling', size=15)
bullets(s, X0, 4.07, COLW, 2.80, [
    [('GoEmotions: ', B), ('rater-level, multi-label (211k ratings) → aggregate; keep one Ekman label (31,621 ≈ paper)', {})],
    [('Friends: ', B), ('“non-neutral” class, Windows-1252 text → removed, re-decoded (11,731 = paper)', {})],
    [('TEC: ', B), ('no neutral, yet LMs predict it 19–27% → label-space alignment', {})],
    [('All: ', B), ('neutral 56–70%, leakage risk → weighted + macro metrics; TEC as clean test', {})],
], size=12.5, after=6)
line(s, DIV, 3.65, DIV, 6.05)
heading(s, COL2, 3.65, COLW, 'Features Extracted (paper → ours)', size=15)
bullets(s, COL2, 4.07, COLW, 2.80, [
    [('Tweet: ', B), ('7 DistilRoBERTa scores → both LMs\' scores + disagreement descriptors + sentence embeddings', {})],
    [('Phrase: ', B), ('TextRank mean / std → + contextual embedding', {})],
    [('Edges: ', B), ('tweet–phrase–emotion + links to 10 semantic nearest neighbours', {})],
    [('Protocol: ', B), ('20% labelled pool / 80% untouched test; 500 gold labels; 5 seeds', {})],
], size=12.5, after=6)
text(s, X0, 6.35, W, 0.40, 'Every baseline receives the same 500 gold labels (1.6–4.3% of each dataset); gold labels never touch the agreement split.', size=12.5)
notes(s, 'Each fix is backed by evidence from the data. GoEmotions subset reconstructed to within 2% of the paper (31,621 vs 32,291); Friends identical; '
         'TEC 21,051 vs 21,047. Every baseline gets the same 500 labels (1.6-4.3% of each dataset).')

# ================================================================ 5. Algorithmic design & approach
s = new_slide(); page(s)
title(s, 'Algorithmic Design and Approach',
      'v1 DA-HGNN-F: disagreement-aware GNN + calibrated LM teacher  →  v2 DA-HGNN-E: equal-vote ensemble with fine-tuned LMs')
bw, gap = 2.62, 0.484
boxes = [('1 · Two LM annotators', 'DistilRoBERTa + RoBERTa-large: scores, embeddings, agreement split'),
         ('2 · Heterogeneous graph', 'Tweet–Phrase–Emotion (paper) + semantic Tweet–Tweet links'),
         ('3 · DA-HGNN student', 'Paper\'s Eq. 1 + semantic attention; soft distillation + 500 gold labels'),
         ('4 · Decision layer', 'Calibrated LM teacher ⊕ student; label-space alignment; v2 adds fine-tuned LMs')]
for i, (h, b_) in enumerate(boxes):
    x = X0 + i * (bw + gap)
    rect(s, x, 1.85, bw, 1.25)
    text(s, x + 0.12, 1.93, bw - 0.24, 1.10, [[(h, {'bold': True, 'color': BLACK, 'size': 13})], '', [(b_, {'size': 11.5})]], color=DARK)
    if i < 3:
        arrow(s, x + bw + 0.07, 2.34, w=gap - 0.14, h=0.28)
heading(s, X0, 3.45, COLW, 'Evidence Behind the Design', size=15)
bullets(s, X0, 3.87, COLW, 3.00, [
    [('Calibrating the LM teacher', B), (' on the few labels: −3.6 points if removed (largest effect)', {})],
    'Multi-view features −1.6; LM embeddings −1.4; graph student −1.2 (TEC −3.2)',
    'Rejected in 6 prototyping rounds: label-free student (= LM average), zero-feature option (0.34 ± 0.23), gold-label propagation (−7.6)',
    'Best when labels are scarce: at 50 labels +5.9 / +2.0 / +2.9 over a supervised model',
], size=12.5, after=6)
line(s, DIV, 3.45, DIV, 5.95)
heading(s, COL2, 3.45, COLW, 'Room for Improvement (Phase 3C)', size=15)
bullets(s, COL2, 3.87, COLW, 3.00, [
    'Candidates scored on held-out pool data only; selection rule fixed in advance; test split used once',
    [('Added equal votes: supervised MLP + LoRA-fine-tuned RoBERTa / DistilRoBERTa → ', {}), ('.694 vs .677', B), (' (v1)', {})],
    'No selection overfitting: 0.694 held-out vs 0.692 test',
    'Self-training promising (+0.3 to +1.2); oracle headroom 0.82 → learned combiner next',
], size=12.5, after=6)
text(s, X0, 6.35, W, 0.40, 'Every design choice was made on dev / held-out data — the test split was used only for the final numbers.', size=12.5)
notes(s, 'The layer is the paper\'s own Eq. 1 and Eq. 4-5; what changes is the input features, the training signal and the decision layer. '
         'Every design choice was made on dev data, never on the test split.')

# ================================================================ 6. Results & conclusion
s = new_slide(); page(s)
title(s, 'Results — Beating the Base Paper', 'Untouched 80% test split · 500 gold labels · mean of 5 seeds · total accuracy')
cd = CategoryChartData(); cd.categories = ['GoEmotions', 'Friends', 'TEC']
series = [('Paper best (reported)', (0.737, 0.535, 0.436), 'BFBFBF'), ('Paper HGNN (our replication)', (0.666, 0.635, 0.405), 'D9D9D9'),
          ('Fine-tuned RoBERTa (LoRA)', (0.771, 0.713, 0.546), '8C8C8C'), ('v1 DA-HGNN-F (ours)', (0.788, 0.719, 0.517), '404040'),
          ('v2 DA-HGNN-E (ours)', (0.801, 0.732, 0.552), '000000')]
for n_, v, _ in series: cd.add_series(n_, v)
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(X0), Inches(1.80), Inches(7.30), Inches(4.30), cd)
ch = gf.chart; chart_style(ch, size=10)
pl = ch.plots[0]; pl.gap_width = 60; pl.overlap = -5; pl.has_data_labels = True
dl = pl.data_labels; dl.number_format = '.00'; dl.number_format_is_linked = False; dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(9); dl.font.color.rgb = RGB(DARK)
for ser, (_, _, col) in zip(pl.series, series):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = RGB(col); ser.format.line.fill.background()
ch.value_axis.maximum_scale = 0.9; ch.value_axis.minimum_scale = 0.0; ch.value_axis.major_unit = 0.1
ch.value_axis.tick_labels.number_format = '0.0'; ch.value_axis.tick_labels.number_format_is_linked = False
heading(s, 8.35, 1.80, 4.28, 'Key Results', size=15)
bullets(s, 8.35, 2.22, 4.28, 3.95, [
    [('+6.4 / +19.7 / +11.6', B), (' accuracy points over the paper\'s best (p < 0.001)', {})],
    'v2 beats v1 on every dataset (p ≤ 0.002) and ties fine-tuned RoBERTa on TEC',
    [('Disagreed GoEmotions posts: ', {}), ('.765', B), (' vs .356 for RoBERTa-large', {})],
    'Limits: needs ~500 labels; the graph helps most when labels are scarce',
], size=12.5, after=8)
text(s, X0, 6.35, W, 0.45, [[('Take-away: ', B), ('treat disagreement between models as information, calibrate it with a few gold labels, '
                                               'and spend those labels where the models are weakest.', {})]], size=12.5)
notes(s, 'Compared on the same test split, v2 beats the paper\'s replicated HGNN by +13.6 / +9.7 / +14.8 points. '
         'Honest limits: label-free we only match the LM average; inside v2 the graph student contributes within +/-0.3 points.')

prs.save(OUT)
print('saved', OUT, 'slides:', len(prs.slides))

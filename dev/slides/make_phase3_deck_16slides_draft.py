# Builds Capstone_Phase3_Slides.pptx in the exact visual style of Capstone_Phase2_Slides.pptx
# (16:9, white, Cambria titles, Calibri body, grey rules, F2F2F2 table headers, thin black process boxes).
import sys
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION
from pptx.oxml.ns import qn
from lxml import etree

OUT = sys.argv[1] if len(sys.argv) > 1 else 'Capstone_Phase3_Slides.pptx'
BLACK, DARK, GREY, RULE, HDR, ARROW, WHITE = '000000', '404040', '6E6E6E', 'D9D9D9', 'F2F2F2', 'BFBFBF', 'FFFFFF'
X0, W = 0.70, 11.93
COL2 = 6.92; COLW = 5.71; DIV = 6.67

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
RGB = RGBColor.from_string
PAGE = [1]


# ---------------------------------------------------------------- primitives
def _box(slide, x, y, w, h, anchor='t'):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = s.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {'t': MSO_ANCHOR.TOP, 'm': MSO_ANCHOR.MIDDLE, 'b': MSO_ANCHOR.BOTTOM}[anchor]
    return s, tf


def _run(p, text, font='Calibri', size=14, bold=False, italic=False, color=DARK, spc=None):
    r = p.add_run(); r.text = text
    f = r.font; f.name = font; f.size = Pt(size); f.bold = bold; f.italic = italic; f.color.rgb = RGB(color)
    if spc: r._r.get_or_add_rPr().set('spc', str(spc))
    return r


def _para_fmt(p, line=None, after=None, bullet=False, align=None):
    pPr = p._p.get_or_add_pPr()
    if bullet:
        pPr.set('marL', '177800'); pPr.set('indent', '-177800')
    for tag in ('a:lnSpc', 'a:spcAft', 'a:buSzPct', 'a:buChar', 'a:buNone'):
        for e in pPr.findall(qn(tag)): pPr.remove(e)
    if line:
        e = etree.SubElement(pPr, qn('a:lnSpc')); etree.SubElement(e, qn('a:spcPct')).set('val', str(int(line * 1000)))
    if after is not None:
        e = etree.SubElement(pPr, qn('a:spcAft')); etree.SubElement(e, qn('a:spcPts')).set('val', str(int(after * 100)))
    if bullet:
        etree.SubElement(pPr, qn('a:buSzPct')).set('val', '100000'); etree.SubElement(pPr, qn('a:buChar')).set('char', '•')
    if align: p.alignment = {'l': PP_ALIGN.LEFT, 'c': PP_ALIGN.CENTER, 'r': PP_ALIGN.RIGHT}[align]


def text(slide, x, y, w, h, paras, font='Calibri', size=14, bold=False, italic=False, color=DARK, align='l', spc=None,
         anchor='t', line=None, after=None):
    """paras: str (\\n = new paragraph) or list of paragraphs; a paragraph is str or list of (text, overrides)."""
    s, tf = _box(slide, x, y, w, h, anchor)
    if isinstance(paras, str): paras = paras.split('\n')
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        runs = [(para, {})] if isinstance(para, str) else para
        for t, o in runs:
            _run(p, t, o.get('font', font), o.get('size', size), o.get('bold', bold), o.get('italic', italic), o.get('color', color), o.get('spc', spc))
        _para_fmt(p, line, after, align=align)
    return s


def bullets(slide, x, y, w, h, items, size=14, color=DARK, after=10, line=108):
    s, tf = _box(slide, x, y, w, h)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        runs = [(it, {})] if isinstance(it, str) else it
        for t, o in runs:
            _run(p, t, 'Calibri', o.get('size', size), o.get('bold', False), o.get('italic', False), o.get('color', color))
        _para_fmt(p, line, after, bullet=True)
    return s


def line(slide, x1, y1, x2, y2, color=RULE, width=1.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGB(color); c.line.width = Pt(width)
    st = c._element.find(qn('p:style'))
    if st is not None: c._element.remove(st)
    return c


def rect(slide, x, y, w, h, line_color=BLACK, fill=WHITE, width=1.0):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = RGB(fill)
    if line_color: s.line.color.rgb = RGB(line_color); s.line.width = Pt(width)
    else: s.line.fill.background()
    s.shadow.inherit = False
    s.text_frame.text = ''
    return s


def arrow(slide, x, y, w=0.66, h=0.28):
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = RGB(ARROW); s.line.fill.background(); s.shadow.inherit = False
    return s


def _cell_border(cell, color=RULE, w=9525):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ('a:lnL', 'a:lnR', 'a:lnT', 'a:lnB'):
        for e in tcPr.findall(qn(tag)): tcPr.remove(e)
    for i, tag in enumerate(('a:lnL', 'a:lnR', 'a:lnT', 'a:lnB')):
        ln = etree.Element(qn(tag), w=str(w), cap='flat', cmpd='sng', algn='ctr')
        sf = etree.SubElement(ln, qn('a:solidFill')); etree.SubElement(sf, qn('a:srgbClr')).set('val', color)
        etree.SubElement(ln, qn('a:prstDash')).set('val', 'solid')
        tcPr.insert(i, ln)


def table(slide, x, y, col_w, rows, row_h, hdr_size=12.5, body_size=11.5, bold_first_col=False, col_align=None):
    """rows[0] = header. A cell is str ('\\n' = new paragraph) or list of paragraphs of (text, overrides) runs."""
    shp = slide.shapes.add_table(len(rows), len(col_w), Inches(x), Inches(y), Inches(sum(col_w)), Inches(sum(row_h)))
    tbl = shp.table
    tblPr = tbl._tbl.tblPr
    for a in list(tblPr.attrib): del tblPr.attrib[a]
    for e in list(tblPr): tblPr.remove(e)
    for j, cw in enumerate(col_w): tbl.columns[j].width = Inches(cw)
    for i, rh in enumerate(row_h): tbl.rows[i].height = Inches(rh)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Emu(76200); c.margin_top = c.margin_bottom = Emu(50800)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid(); c.fill.fore_color.rgb = RGB(HDR if i == 0 else WHITE)
            tf = c.text_frame; tf.word_wrap = True
            paras = val.split('\n') if isinstance(val, str) else val
            for k, para in enumerate(paras):
                p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                runs = [(para, {})] if isinstance(para, str) else para
                for t, o in runs:
                    hdr = i == 0
                    _run(p, t, 'Calibri', o.get('size', hdr_size if hdr else body_size),
                         o.get('bold', hdr or (bold_first_col and j == 0)), o.get('italic', False), o.get('color', BLACK if hdr else DARK))
                _para_fmt(p, align=(col_align[j] if col_align else 'l'))
            _cell_border(c)
    return shp


def page(slide):
    PAGE[0] += 1
    text(slide, 12.13, 7.02, 0.50, 0.25, str(PAGE[0]), size=10, color=GREY, align='r')


def title(slide, t, sub=None, sub_bold_prefix=None):
    text(slide, X0, 0.55, W, 0.70, t, font='Cambria', size=30, bold=True, color=BLACK)
    if sub:
        runs = ([(sub_bold_prefix, {'bold': True, 'color': BLACK})] if sub_bold_prefix else []) + [(sub, {})]
        text(slide, X0, 1.30, W, 0.40, [runs], size=13, italic=True, color=GREY if not sub_bold_prefix else DARK)


def heading(slide, x, y, w, t, size=16):
    text(slide, x, y, w, 0.40, t, font='Cambria', size=size, bold=True, color=BLACK)


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def new_slide():
    s = prs.slides.add_slide(BLANK)
    return s


def chart_style(ch, legend=True, size=10):
    ch.font.name = 'Calibri'; ch.font.size = Pt(size); ch.font.color.rgb = RGB(DARK)
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(size); ch.legend.font.name = 'Calibri'
    va, ca = ch.value_axis, ch.category_axis
    va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = RGB('E6E6E6'); va.major_gridlines.format.line.width = Pt(0.75)
    va.format.line.fill.background(); va.tick_labels.font.size = Pt(size); va.tick_labels.font.color.rgb = RGB(GREY)
    ca.format.line.color.rgb = RGB(RULE); ca.tick_labels.font.size = Pt(size); ca.tick_labels.font.color.rgb = RGB(DARK)
    ca.has_major_gridlines = False


# ================================================================ 1. Title (identical structure to Phase 2)
s = new_slide()
text(s, X0, 1.55, W, 0.40, ['CAPSTONE PROJECT — PHASE 3', '', 'PSID - 181'], size=14, color=BLACK, spc=200)
line(s, X0, 2.55, X0 + W, 2.55)
text(s, X0, 2.85, W, 0.90, 'Advancing Emotion Recognition in Social Media', font='Cambria', size=34, bold=True, color=BLACK)
text(s, X0, 3.65, W, 0.70, 'A Novel Integration of Heterogeneous Neural Networks with Fine-Tuned Language Models', size=16, italic=True, color=DARK)
text(s, X0, 4.45, W, 0.40, 'Problem Identification & Literature Survey  •  Datasets & Preprocessing  •  Algorithmic Design & Results',
     size=13, color=GREY)
text(s, X0, 6.35, W, 0.35, 'M A KAUSHIK', size=14, bold=True, color=BLACK)
text(s, X0, 6.70, W, 0.35, 'CH.SC.U4CSE24123', size=12, color=BLACK)
notes(s, 'Phase 3: we replicated the base paper, found why its best results are unstable, and built an improved model (DA-HGNN-F, '
         'then the ensemble DA-HGNN-E) that beats the paper on all three datasets.')

# ================================================================ 2. Problem identification
s = new_slide(); page(s)
title(s, 'Problem Identification', 'Two strong fine-tuned language models often give different emotions for the same post — which label do we trust?')
heading(s, X0, 2.05, COLW, 'The Problem')
bullets(s, X0, 2.55, COLW, 3.40, [
    'Social-media posts are short, informal, sarcastic and full of implicit emotion',
    [('DistilRoBERTa and RoBERTa-large ', {}), ('disagree on 27–33% of posts', {'bold': True}), (' (label non-compliance)', {})],
    'On disagreed posts each model alone is right only 14–46% of the time',
    'Goal: label every post — including the disagreed ones — without large-scale new annotation',
])
line(s, DIV, 2.05, DIV, 5.95)
heading(s, COL2, 2.05, COLW, 'Example — a disagreed TEC tweet')
text(s, COL2, 2.55, COLW, 0.55, '“Another early morning tomorrow and a long drive home.”', size=14, italic=True, color=BLACK)
table(s, COL2, 3.20, [3.35, 2.36], [['Source', 'Emotion label'], ['DistilRoBERTa', 'neutral'], ['RoBERTa-large', 'fear'],
                                     ['Human annotation (gold)', 'joy'], [[[('Our model (DA-HGNN-F)', {'bold': True, 'color': BLACK})]], [[('joy  — correct', {'bold': True, 'color': BLACK})]]]],
      [0.42, 0.42, 0.42, 0.42, 0.42], body_size=12.5)
text(s, X0, 6.15, W, 0.60, 'Why it matters: public-mood tracking, distress detection and brand monitoring fail exactly on these ambiguous posts.', size=12.5)
notes(s, 'Disagreement rates: GoEmotions 27%, Friends 28%, TEC 33%. On disagreed posts, DistilRoBERTa is right 14-38% and RoBERTa-large 32-46% of the time.')

# ================================================================ 3. Base paper & replication
s = new_slide(); page(s)
title(s, 'Base Paper and Replication (Phase 3A)', ' Maazallahi, Asadpour & Bazmi (2025) — compliance-driven training set + Tweet–Phrase–Emotion HGNN; all 180 grid configurations re-trained',
      sub_bold_prefix='Base paper:')
table(s, X0, 1.85, [3.55, 2.55, 3.55, 2.28], [
    ['Check', 'Paper', 'Our replication', 'Verdict'],
    ['Agreement split (GoEmotions / Friends / TEC)', '74 / 70 / 65 %', '73 / 72 / 67 %', 'Reproduced'],
    ['LM baselines (3 models × 3 datasets)', 'Tables 5–6', 'within ±0.02 on almost every cell', 'Reproduced'],
    ['HGNN-Max, GoEmotions accuracy (agreed / disagreed / total)', '.720 / .530 / .671', '.713 / .536 / .665', 'Reproduced (one run)'],
    ['HGNN-Min — paper\'s best (disagreed accuracy)', '.677', '.397', 'Not reproduced'],
    ['Attention better than mean aggregation', '.56 vs .50 F1', '+0.010 (p = 0.70)', 'Not reproduced'],
    ['Stability of HGNN-Max over 10 seeds', 'not reported', '0.344 ± 0.228 (0.05 – 0.61)', 'Lucky seed'],
    ['Rule "if models disagree → neutral" (no training)', 'not reported', '.759 disagreed acc / .740 total F1', 'Matches paper\'s best'],
], [0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48], body_size=12, bold_first_col=False)
text(s, X0, 6.15, W, 0.60, 'Root cause: disagreed posts are a different population (76% neutral on GoEmotions) and the paper\'s zero-feature option '
                           'is never seen in training — so its best results depend on the random seed.', size=12.5)
notes(s, 'We reproduced every baseline and the paper\'s HGNN-Max. The paper\'s headline HGNN-Min result did not reproduce; '
         'our 10-seed study shows that configuration family swings between 0.05 and 0.61 on disagreed posts.')

# ================================================================ 4. Literature survey
s = new_slide(); page(s)
title(s, 'Literature Survey — Related Work')
lit = [['Area', 'Key works', 'What it offers', 'Gap for our problem'],
       ['Fine-tuned emotion LMs', 'Hartmann et al. (2023); Liu et al. (2019) RoBERTa; Sanh et al. (2019)', 'Strong off-the-shelf 7-class emotion models', 'Black boxes that disagree with each other'],
       ['Learning from disagreement', 'Uma et al. (2021); Fornaciari et al. (2021); Plank (2022); Xu et al. (2024)', 'Disagreement is signal; soft-label training', 'Studied for human annotators, not LM vs LM'],
       ['Weak supervision & distillation', 'Ratner et al. (2017) Snorkel; Hinton et al. (2015); Xie et al. (2020) Noisy Student', 'Learning from noisy / soft labels', 'Student can copy the teacher\'s bias'],
       ['Graphs for text', 'Mihalcea & Tarau (2004) TextRank; Yao et al. (2019) TextGCN; Lin et al. (2021) BertGCN; Wang et al. (2019) HAN',
        'Relational context; LM features inside GNNs', 'Sparse phrase graphs; attention unused with one relation'],
       ['Benchmarking & label propagation', 'Lv et al. (2021) SimpleHGN; Huang et al. (2021) Correct & Smooth', 'Simple baselines often match GNNs', 'Many HGNN gains vanish under fair tuning'],
       ['Calibration & label shift', 'Guo et al. (2017); Saerens et al. (2002); Menon et al. (2021)', 'Fix class priors with a small dev set', 'Never applied per agreement group'],
       ['Recent LLM & PEFT', 'Dong et al. (2026) multi-agent LLMs; Hu et al. (2022) LoRA', 'Explicit conflict resolution; cheap fine-tuning', '70B-scale models; needs labels']]
table(s, X0, 1.40, [2.45, 3.95, 2.83, 2.70], lit, [0.45] + [0.62] * 7, hdr_size=12.5, body_size=11)
text(s, X0, 6.62, W, 0.35, '27 works surveyed in total (full table in the review record); gaps G1–G7 on the next slide.', size=12, color=GREY, italic=True)
notes(s, 'Six families of work. The key insight from the learning-from-disagreement literature: disagreement carries information. '
         'Lv et al. 2021 motivated our strict evaluation protocol.')

# ================================================================ 5. Research gaps → our response
s = new_slide(); page(s)
title(s, 'Research Gaps → Our Response', 'Gaps found in the literature and confirmed by our replication, each mapped to a component of the improved model')
table(s, X0, 1.85, [3.35, 4.45, 4.13], [
    ['Gap', 'Evidence', 'Our component'],
    ['G1  Disagreement used only as a filter', 'How the models disagree (confidence, divergence) is discarded', 'Disagreement descriptors + per-group calibration'],
    ['G2  Thin features', '7 DistilRoBERTa scores per tweet; 23% of phrase nodes have empty features', 'Both LMs\' scores + sentence embeddings; contextual phrase embeddings'],
    ['G3  Train / test mismatch', 'Trained on agreed, tested on disagreed posts; zero-feature runs: 0.34 ± 0.23', 'Disagreement-conditioned calibration; score masking; few gold labels'],
    ['G4  Hard, noisy pseudo-labels', 'Agreed labels only 71% / 72% / 43% correct', 'Confidence-weighted soft labels + small gold set'],
    ['G5  Weak evaluation protocol', 'Model picked by similarity on test data; single runs', '20/80 split, 5 seeds, ablations, t-tests, pre-registered selection'],
    ['G6  Label-space mismatch', 'TEC has no neutral, yet LMs predict neutral 19–27% of the time', 'Label-space alignment (applied to every baseline too)'],
    ['G7  Sparse graph', 'Tweets linked only by exact phrases; attention ≈ mean', 'Semantic tweet–tweet k-NN relation'],
], [0.45] + [0.56] * 7, body_size=11.5)
notes(s, 'Every component of our model answers one gap; the ablation later shows which ones actually matter.')

# ================================================================ 6. Datasets
s = new_slide(); page(s)
title(s, 'Datasets', 'Three public corpora: in-domain for the language models, conversational, and completely unseen')
table(s, X0, 1.85, [1.75, 1.75, 1.35, 1.45, 3.25, 2.38], [
    ['Dataset', 'Domain', 'Posts', 'Classes', 'Source', 'Role'],
    ['GoEmotions', 'Reddit comments', '31,621', '7 (Ekman + neutral)', 'Google Research raw rater-level release', 'Main; in the LMs\' training data'],
    ['Friends (EmotionLines)', 'TV dialogue', '11,731', '7', 'Official EmotionX release (= paper)', 'Conversational test'],
    ['TEC', 'Tweets', '21,051', '6 (no neutral)', 'Twitter Emotion Corpus (Mohammad, 2012)', 'Unseen by every model'],
], [0.45, 0.62, 0.62, 0.62], body_size=12)
for i, (num, cap) in enumerate((('27–33%', 'of posts get conflicting labels\nfrom the two models'),
                                ('56–70%', 'neutral posts in GoEmotions and\nFriends — severe class imbalance'),
                                ('43%', 'accuracy of the agreed label on TEC —\nagreement is not correctness'))):
    x = X0 + i * 4.07
    text(s, x, 4.65, 3.80, 0.70, num, font='Cambria', size=36, bold=True, color=BLACK)
    text(s, x, 5.40, 3.80, 0.70, cap, size=13, color=GREY)
notes(s, 'GoEmotions subset reconstructed to within 2% of the paper (31,621 vs 32,291); Friends identical (11,731); TEC 21,051 vs 21,047.')

# ================================================================ 7. Inconsistencies & preprocessing
s = new_slide(); page(s)
title(s, 'Handling Data Inconsistencies', 'Every problem found in the raw data, the evidence for it, and how it was handled')
table(s, X0, 1.85, [3.00, 4.40, 4.53], [
    ['Inconsistency', 'Evidence', 'Handling'],
    ['GoEmotions is rater-level, multi-label', '211,225 ratings for 58,011 comments; 27 emotions + neutral', 'Aggregate per comment; keep posts with exactly one Ekman label (31,621)'],
    ['Friends "non-neutral" class', '2,772 utterances without a majority emotion', 'Removed → 11,731 utterances, identical to the paper'],
    ['Broken character encoding', 'Friends text like "he\\x92s lost it" (Windows-1252)', 'Re-decoded latin-1 → cp1252'],
    ['Duplicate / multi-label tweets', '42 duplicate tweet IDs in TEC, sometimes with different labels', 'Kept as the corpus defines them; documented'],
    ['Label-space mismatch', 'TEC has no neutral; LMs still predict it 19–27% of the time', 'Restrict predictions to the dataset\'s own classes'],
    ['Noise in text', 'URLs, @mentions, [NAME] placeholders, HTML entities, #hashtags', 'Removed / unescaped; # stripped but word kept'],
    ['Class imbalance & population shift', 'Neutral 56–70%; disagreed set more neutral (76% vs 68%)', 'Weighted + macro metrics; per-group calibration'],
    ['Leakage risk', 'GoEmotions in LM training; Friends in DistilRoBERTa-v2 training', 'TEC kept as the clean test; leakage reported'],
], [0.42] + [0.50] * 8, body_size=11)
text(s, X0, 6.55, W, 0.35, 'Pipeline: raw text → unescape → remove URLs / mentions / placeholders / control characters → strip # → normalise whitespace → LM tokenizer | TextRank',
     size=11.5, color=GREY, italic=True)
notes(s, 'Each fix is backed by evidence from the data; the GoEmotions reconstruction is the closest of several filters we tested against the paper\'s counts.')

# ================================================================ 8. Features & protocol
s = new_slide(); page(s)
title(s, 'Feature Extraction and Evaluation Protocol', 'Richer node features than the paper, evaluated under a protocol that never tunes on the test split')
heading(s, X0, 2.05, COLW, 'Node Features (paper → ours)')
bullets(s, X0, 2.55, COLW, 3.50, [
    [('Tweet: ', {'bold': True}), ('7 DistilRoBERTa scores → both LMs\' scores + disagreement descriptors (confidence, entropy, JS divergence) + sentence embeddings', {})],
    [('Phrase: ', {'bold': True}), ('mean / std of agreed tweets → + contextual phrase embedding (no empty phrase vectors)', {})],
    [('Emotion: ', {'bold': True}), ('one-hot vectors (unchanged)', {})],
    [('Edges: ', {'bold': True}), ('tweet–phrase–emotion (paper rule) + semantic tweet–tweet links to the 10 nearest neighbours', {})],
], size=13.5)
line(s, DIV, 2.05, DIV, 5.95)
heading(s, COL2, 2.05, COLW, 'Protocol (no test-set tuning)')
bullets(s, COL2, 2.55, COLW, 3.50, [
    'Each dataset: 20% labelled pool / 80% untouched test split',
    [('Setting A — label-free', {'bold': True}), (' (paper\'s setting); ', {}), ('Setting B — 500 gold labels', {'bold': True}), (' (1.6–4.3% of data)', {})],
    'Every baseline gets exactly the same 500 labels',
    'Design chosen on held-out pool data only; 5 seeds per result',
    'Significance: paired t-tests over seeds + McNemar tests',
], size=13.5)
text(s, X0, 6.20, W, 0.50, 'Metrics: accuracy, weighted F1 (the paper\'s metric) and macro-F1 — reported on agreed, disagreed and all posts', size=12.5)
notes(s, 'Setting B is our main setting; every comparison method also receives the 500 labels, so the comparison is fair.')

# ================================================================ 9. Algorithmic design
s = new_slide(); page(s)
title(s, 'Algorithmic Design — DA-HGNN-F', 'Disagreement-Aware Heterogeneous Graph Neural Network with teacher Fusion')
bw, gap = 2.62, 0.484
boxes = [('1 · Two LM annotators', 'DistilRoBERTa + RoBERTa-large: scores, embeddings, agreement split'),
         ('2 · Heterogeneous graph', 'Tweet–Phrase–Emotion (paper) + semantic Tweet–Tweet k-NN'),
         ('3 · DA-HGNN student', 'Eq. 1 GraphSAGE + semantic attention; soft distillation + 500 gold labels'),
         ('4 · Decision layer', 'Calibrated LM teacher ⊕ student, per agreement group; label-space alignment')]
for i, (h, b) in enumerate(boxes):
    x = X0 + i * (bw + gap)
    rect(s, x, 1.85, bw, 1.30)
    text(s, x + 0.12, 1.93, bw - 0.24, 1.15, [[(h, {'bold': True, 'color': BLACK, 'size': 13.5})], '', [(b, {'size': 11.5})]], color=DARK)
    if i < 3:
        arrow(s, x + bw + 0.07, 2.36, w=gap - 0.14, h=0.28)
heading(s, X0, 3.55, COLW, 'Why this design')
bullets(s, X0, 4.05, COLW, 2.10, [
    'Disagreed posts have their own class prior → calibrate the LMs with the few labels',
    'Embeddings + both models\' scores give the GNN real evidence',
    'A few gold labels teach what the LMs do not know',
    'Fusion pairs the in-domain teacher with the robust graph student',
], size=13.5, after=8)
line(s, DIV, 3.55, DIV, 6.10)
heading(s, COL2, 3.55, COLW, 'Alternatives Rejected (prototyping)')
bullets(s, COL2, 4.05, COLW, 2.10, [
    'Label-free student — only re-learns the LM average',
    'Logistic-regression resolver — weaker than calibration at 200 labels',
    'Zero features for disagreed posts — seed-unstable (0.34 ± 0.23)',
    'Gold-label propagation (Correct & Smooth) — −7.6 points',
], size=13.5, after=8)
text(s, X0, 6.35, W, 0.40, 'Loss = confidence-weighted KL to both LMs\' product-of-experts distribution (agreed posts) + cross-entropy on the gold labels; 30% score masking',
     size=12, color=GREY, italic=True)
notes(s, 'The layer is the paper\'s own Eq. 1 and Eq. 4-5; what changes is the input, the training signal and the decision layer.')

# ================================================================ 10. Design log
s = new_slide(); page(s)
title(s, 'Design Log — Six Prototyping Rounds', 'All decisions made on held-out dev data; the test split was never used for design')
table(s, X0, 1.85, [0.85, 3.65, 4.48, 2.95], [
    ['Round', 'Question', 'Finding', 'Decision'],
    ['1', 'Can a label-free student beat the two LMs?', 'No — every student ≈ score average (0.64); calibration lifts all to 0.80', 'Calibration is essential'],
    ['2', 'Teacher vs students vs graph smoothing', 'All calibrated variants within ±0.01; smoothing best F1', 'Keep graph smoothing'],
    ['3', 'Blend weight; smoothing strength (3 seeds)', 'Differences ≤ 0.005; α = 0.5 best on average', 'α = 0.5'],
    ['4', 'Semi-supervised DA-HGNN vs calibration vs supervised MLP', 'Graph beats no-graph by 3–4 points at 200 labels; beats the calibrated teacher on TEC by 3–7', 'Train DA-HGNN semi-supervised'],
    ['5', 'Logistic-regression resolver on graph features', 'Weak at 200 labels; competitive only ≥ 500', 'Dropped'],
    ['6', 'Fuse DA-HGNN with calibrated teacher', 'w = 0.5 best: accuracy .687 vs .677 (teacher)', 'Final DA-HGNN-F'],
], [0.45] + [0.62] * 6, body_size=11.5, col_align=['c', 'l', 'l', 'l'])
notes(s, 'This slide shows the approach was evidence-driven: each round answered one question and fixed one design choice.')

# ================================================================ 11. Results v1
s = new_slide(); page(s)
title(s, 'Results — DA-HGNN-F vs the Base Paper', 'Untouched 80% test split · mean of 5 seeds · total accuracy')
cd = CategoryChartData(); cd.categories = ['GoEmotions', 'Friends', 'TEC']
series = [('Paper best (reported)', (0.737, 0.535, 0.436), 'BFBFBF'), ('Paper HGNN (our replication)', (0.666, 0.635, 0.405), 'D9D9D9'),
          ('RoBERTa-large (raw)', (0.616, 0.650, 0.392), '8C8C8C'), ('DA-HGNN-F (ours)', (0.788, 0.719, 0.517), '000000')]
for n_, v, _ in series: cd.add_series(n_, v)
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(X0), Inches(1.85), Inches(6.60), Inches(4.45), cd)
ch = gf.chart; chart_style(ch)
pl = ch.plots[0]; pl.gap_width = 70; pl.overlap = -5; pl.has_data_labels = True
dl = pl.data_labels; dl.number_format = '0.00'; dl.number_format_is_linked = False; dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(9); dl.font.color.rgb = RGB(DARK)
for ser, (_, _, col) in zip(pl.series, series):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = RGB(col); ser.format.line.fill.background()
ch.value_axis.maximum_scale = 0.9; ch.value_axis.minimum_scale = 0.0; ch.value_axis.tick_labels.number_format = '0.0'; ch.value_axis.tick_labels.number_format_is_linked = False
heading(s, 7.70, 1.85, 4.93, 'Key Results')
bullets(s, 7.70, 2.35, 4.93, 3.90, [
    [('+5.1 / +18.4 / +8.1', {'bold': True, 'color': BLACK}), (' accuracy points over the paper\'s best (GoEmotions / Friends / TEC)', {})],
    [('Disagreed posts (GoEmotions): ', {}), ('0.755', {'bold': True, 'color': BLACK}), (' vs 0.356 (RoBERTa) and 0.538 (paper HGNN)', {})],
    'Significant vs the paper\'s HGNN and both LMs on every dataset (p < 0.001; McNemar p < 10⁻¹⁰)',
    'Biggest gain on Friends, where the paper\'s HGNN lost accuracy even on agreed posts',
], size=13.5)
notes(s, 'Compared on the same test split, our gains over the paper\'s replicated HGNN are even larger: +12.2 / +8.4 / +11.2 points.')

# ================================================================ 12. Ablation & label efficiency
s = new_slide(); page(s)
title(s, 'What Matters — Ablation and Label Efficiency', 'Each component removed in turn (3 datasets × 3 seeds); then the number of gold labels varied from 50 to 2,000')
heading(s, X0, 1.85, 6.4, 'Change in accuracy when a component is removed (points)', size=14)
cd = CategoryChartData()
abl = [('Score masking', 0.0), ('Soft targets', -0.3), ('Whole graph (→ MLP)', -0.4), ('Fusion', -0.8), ('Graph student', -1.2),
       ('LM embeddings', -1.4), ('Multi-view features', -1.6), ('Calibration (few labels)', -3.6)]
cd.categories = [a for a, _ in abl]; cd.add_series('Δ accuracy', [v for _, v in abl])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(X0), Inches(2.25), Inches(6.30), Inches(4.20), cd)
ch = gf.chart; chart_style(ch, legend=False)
pl = ch.plots[0]; pl.gap_width = 45; pl.has_data_labels = True
dl = pl.data_labels; dl.number_format = '0.0'; dl.number_format_is_linked = False; dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(10); dl.font.color.rgb = RGB(DARK)
ser = pl.series[0]; ser.format.fill.solid(); ser.format.fill.fore_color.rgb = RGB('595959'); ser.invert_if_negative = False
ch.has_title = False; ch.value_axis.minimum_scale = -4.0; ch.value_axis.maximum_scale = 0.5; ch.value_axis.major_unit = 1.0; ch.value_axis.tick_labels.number_format = '0'; ch.value_axis.tick_labels.number_format_is_linked = False
ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
line(s, 7.30, 1.85, 7.30, 6.45)
heading(s, 7.55, 1.85, 5.08, 'Total accuracy vs number of gold labels', size=14)
table(s, 7.55, 2.35, [1.20, 1.29, 1.29, 1.30], [
    ['Labels', 'GoEmotions', 'Friends', 'TEC'],
    ['50', [[('.770', {'bold': True, 'color': BLACK}), (' / .711', {})]], [[('.704', {'bold': True, 'color': BLACK}), (' / .684', {})]], [[('.475', {'bold': True, 'color': BLACK}), (' / .446', {})]]],
    ['200', [[('.779', {'bold': True, 'color': BLACK}), (' / .749', {})]], [[('.706', {'bold': True, 'color': BLACK}), (' / .690', {})]], [[('.507', {'bold': True, 'color': BLACK}), (' / .497', {})]]],
    ['500', [[('.788', {'bold': True, 'color': BLACK}), (' / .770', {})]], [[('.717', {'bold': True, 'color': BLACK}), (' / .714', {})]], '.515 / .534'],
    ['2,000', '.788 / .801', '.727 / .738', '.536 / .584'],
], [0.42] + [0.46] * 4, body_size=12, col_align=['c', 'c', 'c', 'c'])
text(s, 7.55, 4.85, 5.08, 1.50, 'DA-HGNN-F (bold) vs supervised MLP on the same labels. DA-HGNN-F is best when labels are scarce (about 50–500); '
                              'with ≥ 1,000 labels plain supervised learning overtakes it.', size=12.5)
notes(s, 'Calibration of disagreed posts is the single largest lever. Score masking had no measurable effect - an honest negative result.')

# ================================================================ 13. Room for improvement
s = new_slide(); page(s)
title(s, 'Room for Improvement (Phase 3C)', 'Candidates judged on held-out pool posts only; selection rule fixed in advance; test split used once')
table(s, X0, 1.85, [6.45, 2.05, 3.43], [
    ['Candidate (5 seeds, held-out pool)', 'Score  (acc + F1) / 2', 'Outcome'],
    ['v1 DA-HGNN-F (graph student ⊕ calibrated teacher)', '.677', 'Starting point'],
    ['+ supervised MLP as a third equal vote', '.687', 'Helps on all three datasets'],
    ['Fine-tune RoBERTa-large on the 500 labels (LoRA), alone', '.681', 'Strong new baseline'],
    [[[('5-way vote: graph student + teacher + MLP + both fine-tuned LMs', {'bold': True, 'color': BLACK})]], [[('.694', {'bold': True, 'color': BLACK})]],
     [[('Selected → DA-HGNN-E (v2)', {'bold': True, 'color': BLACK})]]],
    ['Correct & Smooth with gold labels', '.617', 'Hurts — dropped'],
    ['Self-training / Noisy Student (2 datasets, 1 seed)', '+0.3 to +1.2', 'Promising; too slow on a 6 GB GPU'],
], [0.45] + [0.52] * 6, body_size=12, col_align=['l', 'c', 'l'])
text(s, X0, 5.75, W, 0.80, 'Headroom: at least one of the five members is right on 82% of held-out posts vs 69% for the vote — a learned combiner is the next step. '
                           'Selection did not overfit: 0.694 on held-out data, 0.692 on test.', size=12.5)
notes(s, 'Pre-registered rule: highest mean of (accuracy + F1)/2 across datasets over 5 seeds, ties to fewer members.')

# ================================================================ 14. Final results
s = new_slide(); page(s)
title(s, 'Final Results — DA-HGNN-E (v2)', 'Untouched 80% test split · 500 gold labels · mean of 5 seeds')
for i, (num, ds, gain) in enumerate((('0.801', 'GoEmotions accuracy', '+6.4 points vs paper'), ('0.732', 'Friends accuracy', '+19.7 points vs paper'),
                                     ('0.552', 'TEC accuracy (unseen domain)', '+11.6 points vs paper'))):
    x = X0 + i * 4.07
    text(s, x, 1.85, 3.80, 0.75, num, font='Cambria', size=40, bold=True, color=BLACK)
    text(s, x, 2.65, 3.80, 0.35, ds, size=14, bold=True, color=DARK)
    text(s, x, 2.98, 3.80, 0.35, gain, size=13, color=GREY)
table(s, X0, 3.65, [2.10, 2.17, 2.17, 2.62, 1.75, 1.12], [
    ['Dataset (acc / F1)', 'Paper best (reported)', 'v1 DA-HGNN-F', 'Fine-tuned RoBERTa (LoRA)', 'v2 DA-HGNN-E', 'v2 − v1'],
    ['GoEmotions', '.737 / .738', '.788 / .785', '.771 / .774', [[('.801 / .795', {'bold': True, 'color': BLACK})]], '+1.3 / +1.0'],
    ['Friends', '.535 / .578', '.719 / .717', '.713 / .708', [[('.732 / .724', {'bold': True, 'color': BLACK})]], '+1.3 / +0.7'],
    ['TEC', '.436 / .443', '.517 / .518', '.546 / .541', [[('.552 / .546', {'bold': True, 'color': BLACK})]], '+3.6 / +2.9'],
], [0.45, 0.45, 0.45, 0.45], body_size=12.5, col_align=['l', 'c', 'c', 'c', 'c', 'c'])
text(s, X0, 5.75, W, 0.80, 'v2 beats v1 in accuracy on every dataset (p ≤ 0.002), beats fine-tuned RoBERTa on GoEmotions and Friends, and ties it on TEC (p = 0.36). '
                           'Inside v2 the gain comes from the fine-tuned LMs and the MLP; the graph student contributes within ±0.3 points.', size=12.5)
notes(s, 'Even plain LoRA fine-tuning of RoBERTa-large on 500 labels beats the paper - with few labels, what you train on matters more than the architecture.')

# ================================================================ 15. Conclusion
s = new_slide(); page(s)
title(s, 'Conclusion and Future Work')
heading(s, X0, 1.55, COLW, 'Conclusions')
bullets(s, X0, 2.05, COLW, 4.20, [
    'Base paper reproduced (baselines within ±0.02); its best HGNN results depend on the random seed',
    'Calibrating the LMs with a few labels is the biggest single lever (−3.6 points if removed)',
    'Our models beat the paper on all three datasets: up to +19.7 accuracy points, p < 0.001',
    'With a few labels, fine-tuning the LMs matters more than the graph architecture',
], size=14)
line(s, DIV, 1.55, DIV, 5.70)
heading(s, COL2, 1.55, COLW, 'Limitations and Future Work')
bullets(s, COL2, 2.05, COLW, 4.20, [
    'Needs ~500 gold labels; label-free we only match the LM average',
    'Inside v2 the graph student contributes within ±0.3 points — it matters with very few labels and unseen domains',
    'Self-training looked promising (+0.3 to +1.2) but needs a larger GPU',
    'Next: learned combiner (oracle 0.82), active labelling of disagreed posts, multi-label output, conversation context',
], size=14)
text(s, X0, 6.20, W, 0.50, [[('Take-away: ', {'bold': True, 'color': BLACK}), ('treat disagreement between models as information, and spend a few gold labels where the models are weakest.', {})]], size=13.5)
notes(s, 'Close on the honest message: reproduction first, then evidence-driven improvement, with negatives reported.')

# ================================================================ 16. References
s = new_slide(); page(s)
title(s, 'References')
refs_l = ['Maazallahi, Asadpour & Bazmi (2025). Advancing emotion recognition in social media … Information Processing & Management, 62, 103974.',
          'Demszky et al. (2020). GoEmotions: A dataset of fine-grained emotions. ACL.',
          'Mohammad (2012). #Emotional tweets. *SEM.',
          'Chen et al. (2018). EmotionLines: An emotion corpus of multi-party conversations. LREC.',
          'Hartmann et al. (2023). More than a feeling: Accuracy and application of sentiment analysis. IJRM, 40(1).',
          'Uma et al. (2021). Learning from disagreement: A survey. JAIR, 72.',
          'Fornaciari et al. (2021). Beyond black & white: Leveraging annotator disagreement via soft-label multi-task learning. NAACL.',
          'Hinton, Vinyals & Dean (2015). Distilling the knowledge in a neural network. arXiv:1503.02531.']
refs_r = ['Hamilton, Ying & Leskovec (2017). Inductive representation learning on large graphs (GraphSAGE). NeurIPS.',
          'Wang et al. (2019). Heterogeneous graph attention network (HAN). WWW.',
          'Lin et al. (2021). BertGCN: Transductive text classification by combining GCN and BERT. Findings of ACL.',
          'Lv et al. (2021). Are we really making much progress? Revisiting heterogeneous GNNs. KDD.',
          'Huang et al. (2021). Combining label propagation and simple models out-performs GNNs. ICLR.',
          'Guo et al. (2017). On calibration of modern neural networks. ICML.',
          'Hu et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR.',
          'Dong et al. (2026). Emotion meets coordination: Multi-agent LLMs for fine-grained sentiment detection. PLOS ONE, 21(2).']
bullets(s, X0, 1.55, COLW, 5.20, refs_l, size=11, after=6, line=100)
line(s, DIV, 1.55, DIV, 5.90)
bullets(s, COL2, 1.55, COLW, 5.20, refs_r, size=11, after=6, line=100)
text(s, X0, 6.80, W, 0.30, 'Full list of 37 references in Phase3_Review_Record.md', size=10.5, color=GREY, italic=True)

prs.save(OUT)
print('saved', OUT, 'slides:', len(prs.slides))

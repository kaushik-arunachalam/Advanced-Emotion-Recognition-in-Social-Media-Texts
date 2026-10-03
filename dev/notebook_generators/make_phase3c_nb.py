import nbformat as nbf, json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Capstone Phase 3C — Room for Improvement: DA-HGNN-E (ensemble)
**M A Kaushik · CH.SC.U4CSE24123 · PSID 181**
Prerequisites: Phase 3A (LM scores) and Phase 3B (embeddings, graphs and cached DA-HGNN / MLP runs).

### Why a Phase 3C
Phase 3B's DA-HGNN-F beat the paper on all three datasets. It was **not** uniformly the best few-label method, though: a supervised MLP won on TEC, and the calibrated teacher tied on GoEmotions. Also, on held-out data the three models disagree a lot. On 78% of posts at least one of them is right, against 68% for v1. This notebook asks how much of that headroom can be recovered.

### Protocol (no test-set tuning)
* All candidates are judged on the **held-out pool**: the gold-labelled pool posts that were *not* among the 500 training labels (GoEmotions ≈ 5.8k, Friends ≈ 1.8k, TEC ≈ 3.7k per seed). The 80% test split is not touched in Sections 2–4.
* **Selection rule, fixed before evaluation:** the candidate with the highest mean of (total accuracy + weighted F1)/2 over the three datasets, averaged over 5 seeds. Ties go to fewer members.
* The selected model is evaluated **once** on the test split (Section 5).

### New ingredient: fine-tuning the LMs on the same 500 labels
This is a baseline any reviewer will ask for, and a strong, diverse ensemble member. **RoBERTa-large** is fine-tuned with **LoRA** (r = 16 on the query/key/value projections; Hu et al., 2022) because full fine-tuning does not fit a 6 GB GPU. **DistilRoBERTa** is fully fine-tuned. Both use 80% of the 500 labels for training and 20% for early stopping (the same split as DA-HGNN), and predict only the dataset's own classes.
""")

# ---- shared pipeline from 3B (unchanged)
nb3b = json.load(open(os.path.join(PROJ, 'Capstone_Phase3B_Improved_Model.ipynb'), encoding='utf-8'))
md("## 1. Shared pipeline (identical to Phase 3B)\nData, features, graph, the DA-HGNN model, the decision layer and the evaluation helpers are reused unchanged. Cached 3B runs are loaded, not retrained.")
for c in nb3b['cells']:
    if c['cell_type'] != 'code': continue
    src = ''.join(c['source']); code(src)
    if 'def summarize(' in src: break

code(r'''
import math
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
V2_DIR = os.path.join(PROJECT_DIR, 'results', 'improved_v2'); V2_RUNS = os.path.join(V2_DIR, 'runs')
os.makedirs(V2_RUNS, exist_ok=True)

def run_cached_v2(key, fn):
    path = os.path.join(V2_RUNS, re.sub(r'[^A-Za-z0-9_.=-]+', '_', key) + '.npy')
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    out = fn(); np.save(path, out.astype(np.float16)); return out

def finetune_lm(n, lab, seed, hf, lr, lora=False, epochs=10, bs=16, patience=2, max_len=128):
    # Fine-tune an emotion LM on the gold labels; returns log-probabilities for every post (our label order).
    f, d = FEAT[n], DATA[n]
    set_seed(seed); rng = np.random.RandomState(seed)
    lab = np.array(lab, dtype=int); rng.shuffle(lab)
    n_va = max(10, len(lab) // 5); va, tr = lab[:n_va], lab[n_va:]
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForSequenceClassification.from_pretrained(hf)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    order = [next(k for k, v in id2label.items() if v == l) for l in LABELS]; to_model = np.array(order)
    if lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type='SEQ_CLS', r=16, lora_alpha=32, lora_dropout=0.1, target_modules=['query', 'key', 'value']))
    model.to(DEVICE); texts = d['clean'].tolist()
    dis = torch.zeros(N_CLS, dtype=torch.bool, device=DEVICE); dis[torch.tensor(to_model[~f['allowed']], dtype=torch.long)] = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(tr) / bs); sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler('cuda'); y_model = torch.tensor(to_model[f['gold']], device=DEVICE)
    def logits_for(idx):
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(DEVICE)
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(**enc).logits.float()
        return out.masked_fill(dis, -1e4)
    def predict(idx, bsz=128):
        model.eval(); oi = np.argsort([len(texts[i]) for i in idx]); res = np.zeros((len(idx), N_CLS), np.float32)
        with torch.no_grad():
            for s in range(0, len(idx), bsz):
                b = oi[s:s + bsz]; res[b] = torch.log_softmax(logits_for(idx[b]), -1).cpu().numpy()
        return res
    best, best_state, wait = 1e9, None, 0
    for ep in range(epochs):
        model.train(); perm = rng.permutation(tr)
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            loss = F.cross_entropy(logits_for(b), y_model[torch.tensor(b, device=DEVICE)])
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        vl = F.cross_entropy(torch.tensor(predict(va)), torch.tensor(to_model[f['gold'][va]])).item()
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if 'lora' in k or 'classifier' in k or not lora}
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state, strict=False)
    lp = predict(np.arange(len(texts))); del model; torch.cuda.empty_cache()
    return lp[:, order]
''')

md("""
## 2. Ensemble members (5 seeds; seed *s* = label draw *s*, the same 500 labels for every member)
| Member | Source | Trained on |
|---|---|---|
| `teacher` | calibrated, graph-smoothed LM average (3B) | 500 labels (calibration only) |
| `stu` | DA-HGNN semi (3B) | LM distillation + 500 labels |
| `sup` | supervised MLP on LM embeddings (3B) | 500 labels |
| `ftr` | **RoBERTa-large + LoRA, fine-tuned** (new) | 500 labels |
| `ftd` | **DistilRoBERTa, fully fine-tuned** (new) | 500 labels |

Ensembling is an **unweighted average of log-probabilities** followed by label-space alignment. No weights are fitted.
""")
code(r'''
N = CFG['n_labels']
MEM = {}
for n in CFG['datasets']:
    for s in CFG['seeds']:
        lab = draw_labels(n, N, s)
        L1 = lambda k: run_cached(f'{n}|N{N}|{k}|s{s}', lambda: train_model(n, lab, seed=s, **({'graph': False, 'kd': False} if k == 'MLP-sup' else {})))
        m = dict(teacher=teacher(n, lab), stu=L1('DAHGNN-semi'), sup=L1('MLP-sup'),
                 ftr=run_cached_v2(f'{n}|N{N}|FT-roberta-lora|s{s}', lambda: finetune_lm(n, lab, s, CFG['lms']['roberta'], 3e-4, lora=True)),
                 ftd=run_cached_v2(f'{n}|N{N}|FT-distil|s{s}', lambda: finetune_lm(n, lab, s, CFG['lms']['distil'], 3e-5)))
        MEM[(n, s)] = (lab, {k: norm_lp(v) for k, v in m.items()})
    print(n, 'members ready', time.strftime('%X'))

def ens(m, keys, n):
    return align(FEAT[n], sum(m[k] for k in keys) / len(keys))

def heldout(n, lab):
    v = np.zeros(len(FEAT[n]['gold']), bool); v[np.setdiff1d(FEAT[n]['pool'], lab)] = True; return v

def metrics_on(n, pred, mask):
    f = FEAT[n]; out = {}
    for s_, sm in (('C', mask & f['comp']), ('NC', mask & ~f['comp']), ('T', mask)):
        out[f'acc_{s_}'] = accuracy_score(f['gold'][sm], pred[sm]); out[f'f1_{s_}'] = f1_score(f['gold'][sm], pred[sm], average='weighted', zero_division=0)
    out['macroF1_T'] = f1_score(f['gold'][mask], pred[mask], average='macro', zero_division=0)
    return out
''')

md("""
## 3. Headroom on the held-out pool
If at least one member is right on a post, a perfect combiner could get it right. The gap between this **oracle** and the best real combination shows how much the members complement each other.
""")
code(r'''
hr = []
for (n, s), (lab, m) in MEM.items():
    val, gold = heldout(n, lab), FEAT[n]['gold']
    for name, keys in (('v1 (3B): stu + teacher', ['stu', 'teacher']), ('oracle over {teacher, stu, sup}', ['teacher', 'stu', 'sup']),
                       ('oracle over all 5 members', ['teacher', 'stu', 'sup', 'ftr', 'ftd'])):
        if name.startswith('oracle'):
            P = np.stack([m[k].argmax(1) for k in keys]); pred = np.where((P == gold).any(0), gold, P[0])
        else:
            pred = ens(m, keys, n).argmax(1)
        hr.append(dict(dataset=n, seed=s, variant=name, **metrics_on(n, pred, val)))
HR = pd.DataFrame(hr)
HR.groupby(['variant', 'dataset'], sort=False)['acc_T'].mean().unstack('dataset').assign(mean=lambda t: t.mean(axis=1)).round(3)
''')

md("""
## 4. Candidate improvements on the held-out pool, and selection
Every candidate is scored on the held-out pool over 5 seeds. The table also includes **Correct & Smooth** with the gold labels (Huang et al., 2021): the residual errors of the labelled posts are propagated over the semantic k-NN graph. It is reported as a negative result.
""")
code(r'''
def propagate(n, Y, alpha, iters=20):
    ei = GRAPH[n]['ei'][('tweet', 'similar', 'tweet')]
    Y0 = torch.tensor(Y, device=DEVICE, dtype=torch.float32); Z = Y0.clone()
    for _ in range(iters):
        Z = (1 - alpha) * Y0 + alpha * scatter(Z[ei[0]], ei[1], dim=0, dim_size=len(Z), reduce='mean')
    return Z.cpu().numpy()

def correct_and_smooth(n, lp, lab, a1=0.8, a2=0.8):
    f = FEAT[n]; P = np.exp(lp); Y = np.eye(N_CLS)[f['gold']]
    E = np.zeros_like(P); E[lab] = Y[lab] - P[lab]
    Eh = propagate(n, E, a1); sigma = np.abs(E[lab]).sum(1).mean()
    Eh = Eh * (sigma / (np.abs(Eh).sum(1, keepdims=True) + 1e-9))
    G = np.clip(P + Eh, 0, None); G[lab] = Y[lab]
    S = propagate(n, G, a2); S[:, ~f['allowed']] = 0
    return np.log(S / S.sum(1, keepdims=True) + 1e-9)

CANDS = {
    'v1 (3B): stu + teacher':                  ['stu', 'teacher'],
    '3-way: stu + teacher + sup':              ['stu', 'teacher', 'sup'],
    '4-way: + FT-RoBERTa':                     ['stu', 'teacher', 'sup', 'ftr'],
    '4-way: + FT-DistilRoBERTa':               ['stu', 'teacher', 'sup', 'ftd'],
    '5-way: + both FT':                        ['stu', 'teacher', 'sup', 'ftr', 'ftd'],
    'no graph: teacher + sup + FT-RoBERTa':    ['teacher', 'sup', 'ftr'],
    'stu + FT-RoBERTa':                        ['stu', 'ftr'],
    'teacher + FT-RoBERTa':                    ['teacher', 'ftr'],
    'FT-RoBERTa (LoRA) alone':                 ['ftr'],
    'FT-DistilRoBERTa alone':                  ['ftd'],
}
cand = []
for (n, s), (lab, m) in MEM.items():
    val = heldout(n, lab)
    for name, keys in CANDS.items():
        cand.append(dict(dataset=n, seed=s, variant=name, members=len(keys), **metrics_on(n, ens(m, keys, n).argmax(1), val)))
    cand.append(dict(dataset=n, seed=s, variant='C&S (gold labels) on 3-way', members=3,
                     **metrics_on(n, correct_and_smooth(n, sum(m[k] for k in ['stu', 'teacher', 'sup']) / 3, lab).argmax(1), val)))
CAND = pd.DataFrame(cand); CAND['score'] = (CAND.acc_T + CAND.f1_T) / 2
CAND.to_csv(os.path.join(V2_DIR, 'heldout_candidates.csv'), index=False)
sel_tab = CAND.groupby(['variant', 'dataset'], sort=False)[['acc_T', 'f1_T']].mean().unstack('dataset')
sel_tab[('rule', 'score')] = CAND.groupby(['variant', 'dataset'], sort=False)['score'].mean().unstack('dataset').mean(axis=1)
sel_tab[('rule', 'sd over seeds')] = CAND.groupby(['variant', 'seed'], sort=False)['score'].mean().groupby('variant').std()
sel_tab[('rule', 'members')] = CAND.groupby('variant', sort=False)['members'].first()
sel_tab = sel_tab.sort_values([('rule', 'score'), ('rule', 'members')], ascending=[False, True])
SELECTED = sel_tab.index[0]; SEL_KEYS = CANDS[SELECTED]
sel_tab.to_csv(os.path.join(V2_DIR, 'heldout_selection_table.csv'))
print('Selected by the pre-registered rule:', SELECTED, '→ members', SEL_KEYS)
sel_tab.round(4)
''')

md("""
### 4b. Exploratory: self-training (Noisy Student)
RoBERTa-large (LoRA) is fine-tuned on the 500 gold labels **plus the 5-way ensemble's confident soft labels** on unlabelled posts (top 50% by confidence, weight 0.5). Only GoEmotions and Friends at seed 0 finished on this laptop: GoEmotions took about 22 minutes, and TEC crashed after 30+ minutes because GPU memory spilled into system RAM. So it is **not** eligible for selection and is reported as evidence for future work.
""")
code(r'''
st_rows = []
for n in ('GoEmotions', 'Friends'):
    p = os.path.join(V2_RUNS, f'{n}_N500_ST-roberta-lora_ens5_s0.npy')
    if not os.path.exists(p): continue
    lab, m = MEM[(n, 0)]; val = heldout(n, lab); m = dict(m, st=norm_lp(np.load(p).astype(np.float32)))
    for name, keys in (('FT-RoBERTa (no self-training)', ['ftr']), ('self-trained RoBERTa', ['st']), ('5-way ensemble', ['stu', 'teacher', 'sup', 'ftr', 'ftd']),
                       ('5-way + self-trained RoBERTa', ['stu', 'teacher', 'sup', 'ftr', 'ftd', 'st'])):
        st_rows.append(dict(dataset=n, variant=name, **metrics_on(n, ens(m, keys, n).argmax(1), val)))
ST = pd.DataFrame(st_rows)
ST.pivot_table(index='variant', columns='dataset', values=['acc_T', 'f1_T'], sort=False).round(4) if len(ST) else 'self-training runs not found'
''')

md("""
## 5. Confirmatory evaluation on the test split (single look)
The selected ensemble, **DA-HGNN-E**, is compared on the untouched 80% test split with v1 (3B), the fine-tuned LMs, the other few-label baselines, the paper's HGNN (3A replication) and the raw LMs. All results are over 5 seeds.
""")
code(r'''
T = []
sel3a = pd.read_csv(os.path.join(REP_DIR, 'table5_6_ours.csv'))
for n in CFG['datasets']:
    f = FEAT[n]
    for nm in ('HGNN-Max', 'HGNN-Min'):
        r = sel3a[(sel3a.dataset == n) & (sel3a.model == nm)].iloc[0]
        p = os.path.join(PROC_DIR, 'grid_preds', f"{n}_{r['agg']}_{int(r.hidden)}_{int(r.layers)}_{r.nonagreed}.npy")
        if os.path.exists(p): T.append(dict(dataset=n, seed=0, model=f'Paper {nm} (3A replication)', **evaluate(n, np.load(p).astype(int))))
    T.append(dict(dataset=n, seed=0, model='RoBERTa-large (raw)', **evaluate(n, f['lb'])))
for (n, s), (lab, m) in MEM.items():
    for name, keys in (('DA-HGNN-E (proposed v2)', SEL_KEYS), ('DA-HGNN-F (v1, 3B)', ['stu', 'teacher']),
                       ('FT RoBERTa-large (LoRA)', ['ftr']), ('FT DistilRoBERTa', ['ftd']), ('Calibrated teacher', ['teacher']),
                       ('Supervised MLP', ['sup']), ('DA-HGNN semi', ['stu'])):
        T.append(dict(dataset=n, seed=s, model=name, **evaluate(n, ens(m, keys, n).argmax(1))))
TEST = pd.DataFrame(T); TEST.to_csv(os.path.join(V2_DIR, 'test_results.csv'), index=False)
order = ['DA-HGNN-E (proposed v2)', 'DA-HGNN-F (v1, 3B)', 'FT RoBERTa-large (LoRA)', 'FT DistilRoBERTa', 'Calibrated teacher', 'Supervised MLP',
         'DA-HGNN semi', 'Paper HGNN-Max (3A replication)', 'Paper HGNN-Min (3A replication)', 'RoBERTa-large (raw)']
g = TEST.groupby(['dataset', 'model'])[MAIN_COLS]
test_tab = (g.mean().round(3).astype(str) + np.where(g.std().fillna(0).values > 0, ' ± ' + g.std().fillna(0).round(3).astype(str), ''))
test_tab = test_tab.reindex([(d, mo) for d in CFG['datasets'] for mo in order])
test_tab.to_csv(os.path.join(V2_DIR, 'test_results_table.csv'))
test_tab
''')

code(r'''
from scipy.stats import binomtest
def mcnemar_p(n, pa, pb):
    f = FEAT[n]; m = f['test']; ca, cb = pa[m] == f['gold'][m], pb[m] == f['gold'][m]
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
sig = []
for n in CFG['datasets']:
    ours = TEST[(TEST.dataset == n) & (TEST.model == 'DA-HGNN-E (proposed v2)')].sort_values('seed')
    for other in ('DA-HGNN-F (v1, 3B)', 'FT RoBERTa-large (LoRA)', 'FT DistilRoBERTa', 'Calibrated teacher', 'Supervised MLP'):
        th = TEST[(TEST.dataset == n) & (TEST.model == other)].sort_values('seed')
        keys = {'DA-HGNN-F (v1, 3B)': ['stu', 'teacher'], 'FT RoBERTa-large (LoRA)': ['ftr'], 'FT DistilRoBERTa': ['ftd'],
                'Calibrated teacher': ['teacher'], 'Supervised MLP': ['sup']}[other]
        mc = max(mcnemar_p(n, ens(MEM[(n, s)][1], SEL_KEYS, n).argmax(1), ens(MEM[(n, s)][1], keys, n).argmax(1)) for s in CFG['seeds'])
        for met in ('acc_T', 'f1_T'):
            t = stats.ttest_rel(ours[met].values, th[met].values)
            sig.append(dict(dataset=n, versus=other, metric=met, ours=ours[met].mean(), theirs=th[met].mean(), diff=ours[met].mean() - th[met].mean(),
                            p_paired_t=t.pvalue, mcnemar_max_p=mc))
    for other in ('Paper HGNN-Max (3A replication)', 'RoBERTa-large (raw)'):
        th = TEST[(TEST.dataset == n) & (TEST.model == other)]
        for met in ('acc_T', 'f1_T'):
            t = stats.ttest_1samp(ours[met].values, th[met].iloc[0])
            sig.append(dict(dataset=n, versus=other, metric=met, ours=ours[met].mean(), theirs=th[met].iloc[0], diff=ours[met].mean() - th[met].iloc[0],
                            p_paired_t=t.pvalue, mcnemar_max_p=np.nan))
SIG = pd.DataFrame(sig); SIG['better & p<0.05'] = (SIG['diff'] > 0) & (SIG.p_paired_t < 0.05)
SIG.to_csv(os.path.join(V2_DIR, 'test_significance.csv'), index=False)
SIG.round(4)
''')

md("### 5b. Leave-one-member-out on the test split (what each member contributes)")
code(r'''
loo = []
for (n, s), (lab, m) in MEM.items():
    full = evaluate(n, ens(m, SEL_KEYS, n).argmax(1))
    for k in SEL_KEYS:
        r = evaluate(n, ens(m, [x for x in SEL_KEYS if x != k], n).argmax(1))
        loo.append(dict(dataset=n, seed=s, removed=k, d_acc=r['acc_T'] - full['acc_T'], d_f1=r['f1_T'] - full['f1_T']))
LOO = pd.DataFrame(loo)
loo_tab = LOO.groupby(['removed', 'dataset'])[['d_acc', 'd_f1']].mean().unstack('dataset')
loo_tab[('mean', 'd_acc')] = LOO.groupby('removed').d_acc.mean(); loo_tab[('mean', 'd_f1')] = LOO.groupby('removed').d_f1.mean()
loo_tab.to_csv(os.path.join(V2_DIR, 'leave_one_out.csv'))
(100 * loo_tab).round(2)
''')

md("## 6. Comparison with the base paper (v1 and v2)")
code(r'''
PAPER_BEST = {'GoEmotions': ('HGNN-Min', .737, .738), 'Friends': ('HGNN-Max', .535, .578), 'TEC': ('HGNN-Min', .436, .443)}
rows = []
for n in CFG['datasets']:
    rows.append(dict(dataset=n, system=f'Paper reported: {PAPER_BEST[n][0]}', acc_T=PAPER_BEST[n][1], f1_T=PAPER_BEST[n][2]))
    for mo in ('Paper HGNN-Max (3A replication)', 'RoBERTa-large (raw)', 'FT RoBERTa-large (LoRA)', 'DA-HGNN-F (v1, 3B)', 'DA-HGNN-E (proposed v2)'):
        x = TEST[(TEST.dataset == n) & (TEST.model == mo)][['acc_T', 'f1_T']].mean()
        rows.append(dict(dataset=n, system=mo, acc_T=x.acc_T, f1_T=x.f1_T))
VS2 = pd.DataFrame(rows)
VS2['gain vs paper (acc pts)'] = VS2.apply(lambda r: 100 * (r.acc_T - PAPER_BEST[r.dataset][1]), axis=1)
VS2['gain vs paper (F1 pts)'] = VS2.apply(lambda r: 100 * (r.f1_T - PAPER_BEST[r.dataset][2]), axis=1)
VS2.to_csv(os.path.join(V2_DIR, 'comparison_with_paper_v2.csv'), index=False)

fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
cols = {'Paper reported': '#8C8C8C', 'Paper HGNN-Max (3A replication)': '#BDBDBD', 'RoBERTa-large (raw)': '#DD8452',
        'FT RoBERTa-large (LoRA)': '#937860', 'DA-HGNN-F (v1, 3B)': '#55A868', 'DA-HGNN-E (proposed v2)': '#4C72B0'}
for ax, n in zip(axes, CFG['datasets']):
    sub = VS2[VS2.dataset == n].reset_index(drop=True); x = np.arange(len(sub))
    c = [cols['Paper reported'] if s.startswith('Paper reported') else cols[s] for s in sub.system]
    ax.bar(x - 0.2, sub.acc_T, 0.4, color=c); ax.bar(x + 0.2, sub.f1_T, 0.4, color=c, alpha=0.5)
    for i, (a, f1) in enumerate(zip(sub.acc_T, sub.f1_T)):
        ax.text(i - 0.2, a + 0.01, f'{a:.3f}', ha='center', fontsize=7); ax.text(i + 0.2, f1 + 0.01, f'{f1:.3f}', ha='center', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([s.replace('Paper reported: ', 'Paper: ').replace(' (3A replication)', '\n(3A repl.)').replace(' (proposed v2)', '\n(v2, ours)').replace(' (v1, 3B)', '\n(v1, ours)')
                                          for s in sub.system], rotation=50, ha='right', fontsize=8)
    ax.set_ylim(0, 1); ax.set_title(f'{n}: total accuracy (solid) / weighted F1 (light)')
plt.tight_layout(); plt.savefig(os.path.join(V2_DIR, 'fig_v2_comparison.png'), dpi=150, bbox_inches='tight'); plt.show()
VS2.round(3)
''')

concl = os.path.join(HERE, 'conclusions_3c.md')
if os.path.exists(concl):
    md(open(concl, encoding='utf-8').read())

nb = nbf.v4.new_notebook(); nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'},
                  'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'}}
nbf.write(nb, sys.argv[1]); print('written', len(cells), 'cells')

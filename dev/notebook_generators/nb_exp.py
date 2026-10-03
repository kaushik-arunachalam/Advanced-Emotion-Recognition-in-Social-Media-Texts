# Experiment cells for the Phase 3B notebook (exec'd by make_improved_nb.py; md() and code() come from there)

md("""
## 6. Baselines (no training)
Everything is evaluated on the same 80% test portions. The paper's HGNN uses the **exact predictions of our 3A replication** (the paper's HGNN-Max and HGNN-Min configurations).
""")
code(r'''
ALL = []          # every result row: dataset, setting, model, seed, metrics
def add(n, setting, model, pred, seed=0):
    ALL.append(dict(dataset=n, setting=setting, model=model, seed=seed, **evaluate(n, pred)))

sel3a = pd.read_csv(os.path.join(REP_DIR, 'table5_6_ours.csv'))
for n in CFG['datasets']:
    f = FEAT[n]
    for name, P_ in (('DistilRoBERTa', f['A']), ('RoBERTa-large', f['B']), ('DistilRoBERTa-v2 (Friends-tuned)', f['V2']), ('Score average', f['M'])):
        add(n, '0 Baselines (raw)', name, P_.argmax(1))
    add(n, '0 Baselines (raw)', 'Rule: disagree → neutral', np.where(f['comp'], f['la'], NEU))
    for nm in ('HGNN-Max', 'HGNN-Min'):
        r = sel3a[(sel3a.dataset == n) & (sel3a.model == nm)].iloc[0]
        p = os.path.join(PROC_DIR, 'grid_preds', f"{n}_{r['agg']}_{int(r.hidden)}_{int(r.layers)}_{r.nonagreed}.npy")
        if os.path.exists(p):
            add(n, '0 Baselines (raw)', f'Paper {nm} (3A replication)', np.load(p).astype(int))
pd.DataFrame(ALL).set_index(['dataset', 'model'])[MAIN_COLS].round(3)
''')

md("""
## 7. Setting A — label-free (same information as the paper)
No gold labels at all. Every method below is label-space aligned.
""")
code(r'''
for n in CFG['datasets']:
    f = FEAT[n]
    add(n, 'A label-free', 'RoBERTa-large + alignment', align(f, np.log(f['B'] + 1e-9)).argmax(1))
    add(n, 'A label-free', 'Score average + alignment', teacher(n, cal=False, sm=False).argmax(1))
    add(n, 'A label-free', 'Score average + alignment + smoothing', teacher(n, cal=False, sm=True).argmax(1))
    for s in CFG['seeds']:
        lp = run_cached(f'{n}|LF|DAHGNN|s{s}', lambda: train_model(n, (), seed=s))
        add(n, 'A label-free', 'DA-HGNN (label-free)', lp.argmax(1), s)
        add(n, 'A label-free', 'DA-HGNN (label-free) ⊕ teacher', fuse(n, lp, teacher(n, cal=False, sm=True)).argmax(1), s)
    print(n, 'done', time.strftime('%X'))
summarize(pd.DataFrame([r for r in ALL if r['setting'].startswith('A')]))
''')

md("""
## 8. Setting B — few labels (main result)
Every method gets the **same N = 500 gold labels** (one random draw per seed; seed *s* uses draw *s*). The test portion is untouched.
* **Calibrated LMs / teacher:** the gold labels fit the disagreement-conditioned calibration.
* **Supervised MLP:** a classifier on the same features trained on the N labels only.
* **MLP semi:** distillation from the LMs + N labels, no graph.
* **DA-HGNN semi:** our graph model (distillation + N labels).
* **DA-HGNN-F (proposed):** DA-HGNN semi ⊕ calibrated, graph-smoothed teacher.
""")
code(r'''
N = CFG['n_labels']; LP = {}
for n in CFG['datasets']:
    f = FEAT[n]
    for s in CFG['seeds']:
        lab = draw_labels(n, N, s)
        add(n, 'B few labels', 'RoBERTa-large + calibration', align(f, calibrate(f, np.log(f['B'] + 1e-9), lab)).argmax(1), s)
        add(n, 'B few labels', 'Score average + calibration', teacher(n, lab, sm=False).argmax(1), s)
        t_lp = teacher(n, lab, sm=True)
        add(n, 'B few labels', 'Score average + calibration + smoothing', t_lp.argmax(1), s)
        sup = run_cached(f'{n}|N{N}|MLP-sup|s{s}', lambda: train_model(n, lab, seed=s, graph=False, kd=False))
        add(n, 'B few labels', 'Supervised MLP (N labels only)', sup.argmax(1), s)
        mlp = run_cached(f'{n}|N{N}|MLP-semi|s{s}', lambda: train_model(n, lab, seed=s, graph=False))
        add(n, 'B few labels', 'MLP semi (distillation + N labels, no graph)', mlp.argmax(1), s)
        stu = run_cached(f'{n}|N{N}|DAHGNN-semi|s{s}', lambda: train_model(n, lab, seed=s))
        add(n, 'B few labels', 'DA-HGNN semi', stu.argmax(1), s)
        fin = fuse(n, stu, t_lp); LP[(n, s)] = fin
        add(n, 'B few labels', 'DA-HGNN-F (proposed)', fin.argmax(1), s)
    print(n, 'done', time.strftime('%X'))
RES = pd.DataFrame(ALL); RES.to_csv(os.path.join(RES_DIR, 'all_results.csv'), index=False)
main_tab = summarize(RES)
main_tab.to_csv(os.path.join(RES_DIR, 'main_results_table.csv'))
main_tab
''')

md("""
## 9. Comparison with the base paper
The paper reports full-dataset numbers; ours are on the untouched 80% test portion. The *Paper HGNN (3A replication)* rows evaluate the paper's models on exactly our test portion, which makes them the apples-to-apples comparison.
""")
code(r'''
PAPER_BEST = {   # paper Tables 5-6: best reported HGNN by total F1, and best LM baseline
    'GoEmotions': dict(hgnn=('HGNN-Min', .737, .738, .677, .624), base=('DistilRoBERTa-v2', .669, .686, .551, .589)),
    'Friends':    dict(hgnn=('HGNN-Max', .535, .578, .258, .303), base=('DistilRoBERTa-v2', .868, .867, .845, .843)),
    'TEC':        dict(hgnn=('HGNN-Min', .436, .443, .354, .345), base=('RoBERTa-large', .401, .430, .308, .337))}
mean_of = lambda n, setting, model: RES[(RES.dataset == n) & (RES.setting == setting) & (RES.model == model)][['acc_T', 'f1_T', 'acc_NC', 'f1_NC']].mean()
rows = []
for n in CFG['datasets']:
    pb = PAPER_BEST[n]
    rows.append(dict(dataset=n, system=f"Paper reported: {pb['hgnn'][0]}", acc_T=pb['hgnn'][1], f1_T=pb['hgnn'][2], acc_NC=pb['hgnn'][3], f1_NC=pb['hgnn'][4]))
    rows.append(dict(dataset=n, system=f"Paper reported: best LM ({pb['base'][0]})", acc_T=pb['base'][1], f1_T=pb['base'][2], acc_NC=pb['base'][3], f1_NC=pb['base'][4]))
    for setting, model, label in (('0 Baselines (raw)', 'Paper HGNN-Max (3A replication)', 'Paper HGNN-Max, our replication'),
                                  ('0 Baselines (raw)', 'Paper HGNN-Min (3A replication)', 'Paper HGNN-Min, our replication'),
                                  ('0 Baselines (raw)', 'RoBERTa-large', 'RoBERTa-large (raw)'),
                                  ('A label-free', 'DA-HGNN (label-free) ⊕ teacher', 'Ours — Setting A (label-free)'),
                                  ('B few labels', 'DA-HGNN-F (proposed)', 'Ours — Setting B (N=500)')):
        rows.append(dict(dataset=n, system=label, **mean_of(n, setting, model).to_dict()))
VS = pd.DataFrame(rows)
best_hgnn = VS[VS.system.str.startswith('Paper reported: HGNN')].set_index('dataset')
VS['Δ acc_T vs paper HGNN'] = VS.apply(lambda r: r.acc_T - best_hgnn.loc[r.dataset, 'acc_T'], axis=1)
VS['Δ f1_T vs paper HGNN'] = VS.apply(lambda r: r.f1_T - best_hgnn.loc[r.dataset, 'f1_T'], axis=1)
VS.to_csv(os.path.join(RES_DIR, 'comparison_with_paper.csv'), index=False)
VS.round(3)
''')

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
pal = {'paper': '#8C8C8C', 'repl': '#BDBDBD', 'lm': '#DD8452', 'A': '#55A868', 'B': '#4C72B0'}
for ax, n in zip(axes, CFG['datasets']):
    sub = VS[VS.dataset == n].reset_index(drop=True)
    colors = [pal['paper'] if s.startswith('Paper reported') else pal['repl'] if 'replication' in s else pal['lm'] if 'RoBERTa' in s
              else pal['A'] if 'Setting A' in s else pal['B'] for s in sub.system]
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub.acc_T, 0.4, color=colors); ax.bar(x + 0.2, sub.f1_T, 0.4, color=colors, alpha=0.5)
    for i, (a, f1) in enumerate(zip(sub.acc_T, sub.f1_T)):
        ax.text(i - 0.2, a + 0.01, f'{a:.2f}', ha='center', fontsize=7); ax.text(i + 0.2, f1 + 0.01, f'{f1:.2f}', ha='center', fontsize=7)
    names = [s.replace('Paper reported: ', 'Paper: ').replace(', our replication', '\n(3A replication)').replace('Ours — ', 'Ours: ') for s in sub.system]
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=55, ha='right', fontsize=8)
    ax.set_title(f'{n}: total accuracy (solid) / weighted F1 (light)'); ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig(os.path.join(RES_DIR, 'fig_main_comparison.png'), dpi=150, bbox_inches='tight'); plt.show()
''')

md("""
## 10. Ablation study (Setting B, N = 500, 3 seeds)
Each row removes **one** component from the proposed DA-HGNN-F. The table shows the change in total accuracy and weighted F1 per dataset and on average.
""")
code(r'''
VARIANTS = {   # name -> (student kwargs, or None if no student is used; decision-layer options)
    'DA-HGNN-F (full)':                          (dict(), dict()),
    '− fusion (DA-HGNN semi alone)':             (dict(), dict(no_fusion=True)),
    '− student (calibrated teacher alone)':      (None, dict()),
    '− graph (MLP semi ⊕ teacher)':              (dict(graph=False), dict()),
    '− tweet–tweet k-NN relation':               (dict(knn=False), dict()),
    '− tweet–phrase–emotion relations':          (dict(phrases=False), dict()),
    '− LM sentence embeddings':                  (dict(use_emb=False), dict()),
    '− distillation (gold labels only)':         (dict(kd=False), dict()),
    '− soft targets (hard agreed labels)':       (dict(soft=False), dict()),
    '− confidence weighting':                    (dict(weighted=False), dict()),
    '− score masking':                           (dict(mask_p=0.0), dict()),
    'paper tweet features (DistilRoBERTa only)': (dict(paper_feats=True), dict()),
    '− per-group calibration (one global)':      (dict(), dict(global_cal=True)),
    '− calibration':                             (dict(), dict(no_cal=True)),
    '− teacher smoothing':                       (dict(), dict(no_smooth=True)),
}
abl = []
for n in CFG['datasets']:
    f = FEAT[n]
    for s in CFG['ablation_seeds']:
        lab = draw_labels(n, N, s)
        for name, (skw, dec) in VARIANTS.items():
            stu = None
            if skw is not None:
                key = f'{n}|N{N}|DAHGNN-semi|s{s}' if not skw else f'{n}|N{N}|ABL|' + '|'.join(f'{k}={v}' for k, v in sorted(skw.items())) + f'|s{s}'
                stu = run_cached(key, lambda: train_model(n, lab, seed=s, **skw))
            lp_t = np.log(f['M'] + 1e-9)
            if not dec.get('no_cal'):
                lp_t = calibrate(f, lp_t, lab, per_group=not dec.get('global_cal'))
            lp_t = align(f, lp_t)
            if not dec.get('no_smooth'):
                lp_t = smooth(n, lp_t)
            final = lp_t if stu is None else (stu if dec.get('no_fusion') else fuse(n, stu, lp_t))
            abl.append(dict(dataset=n, seed=s, variant=name, **evaluate(n, final.argmax(1))))
    print(n, 'done', time.strftime('%X'))
# label-space alignment only matters on TEC (the only dataset whose taxonomy lacks a class)
f = FEAT['TEC']
for s in CFG['ablation_seeds']:
    lab = draw_labels('TEC', N, s); saved = f['allowed'].copy(); f['allowed'][:] = True
    stu = run_cached(f'TEC|N{N}|ABL|noalign|s{s}', lambda: train_model('TEC', lab, seed=s))
    final = fuse('TEC', stu, smooth('TEC', align(f, calibrate(f, np.log(f['M'] + 1e-9), lab))))
    f['allowed'][:] = saved
    abl.append(dict(dataset='TEC', seed=s, variant='− label-space alignment', **evaluate('TEC', final.argmax(1))))
ABL = pd.DataFrame(abl); ABL.to_csv(os.path.join(RES_DIR, 'ablation_runs.csv'), index=False)

full = ABL[ABL.variant == 'DA-HGNN-F (full)'].groupby('dataset')[['acc_T', 'f1_T']].mean()
abl_tab = ABL.groupby(['variant', 'dataset'], sort=False)[['acc_T', 'f1_T']].mean().unstack('dataset')
for m in ('acc_T', 'f1_T'):
    for n in CFG['datasets']:
        abl_tab[('Δ' + m, n)] = abl_tab[(m, n)] - full.loc[n, m]
    abl_tab[('Δ' + m, 'mean')] = abl_tab['Δ' + m][[c for c in CFG['datasets'] if c in abl_tab['Δ' + m]]].mean(axis=1)
abl_tab = abl_tab.sort_values(('Δf1_T', 'mean'), ascending=False)
abl_tab.to_csv(os.path.join(RES_DIR, 'ablation_table.csv'))
abl_tab.round(3)
''')

code(r'''
d_ = abl_tab[('Δf1_T', 'mean')].drop('DA-HGNN-F (full)').sort_values()
fig, ax = plt.subplots(figsize=(9.5, 6))
ax.barh(d_.index, d_.values, color=['#C44E52' if v < 0 else '#55A868' for v in d_.values])
ax.axvline(0, color='k', lw=0.8); ax.set_xlabel('change in total weighted F1 vs full DA-HGNN-F (mean over datasets)')
ax.set_title(f'Ablation study (Setting B, N={N}, {len(CFG["ablation_seeds"])} seeds)')
plt.tight_layout(); plt.savefig(os.path.join(RES_DIR, 'fig_ablation.png'), dpi=150); plt.show()
''')

md("""
## 11. Label efficiency: how many gold labels are needed?
For N ∈ {50 … 2000} (capped at the pool size), with 3 label draws each, we compare the calibrated teacher, a supervised MLP, DA-HGNN semi and DA-HGNN-F. The dashed line is the paper's best reported HGNN; the dotted line is raw RoBERTa-large.
""")
code(r'''
bud = []
for n in CFG['datasets']:
    f = FEAT[n]
    for b in CFG['budgets']:
        if b > len(f['pool']): continue
        for s in CFG['budget_seeds']:
            lab = draw_labels(n, b, s)
            t_lp = teacher(n, lab)
            stu = run_cached(f'{n}|N{b}|DAHGNN-semi|s{s}', lambda: train_model(n, lab, seed=s))
            sup = run_cached(f'{n}|N{b}|MLP-sup|s{s}', lambda: train_model(n, lab, seed=s, graph=False, kd=False))
            for model, lp in (('Calibrated teacher (+smoothing)', t_lp), ('Supervised MLP', sup), ('DA-HGNN semi', stu),
                              ('DA-HGNN-F (proposed)', fuse(n, stu, t_lp))):
                bud.append(dict(dataset=n, budget=b, seed=s, model=model, **evaluate(n, lp.argmax(1))))
    print(n, 'done', time.strftime('%X'))
BUD = pd.DataFrame(bud); BUD.to_csv(os.path.join(RES_DIR, 'label_budget_runs.csv'), index=False)

fig, axes = plt.subplots(2, 3, figsize=(19, 9))
for j, n in enumerate(CFG['datasets']):
    for i, m in enumerate(('acc_T', 'f1_T')):
        ax = axes[i, j]
        for model, g in BUD[BUD.dataset == n].groupby('model', sort=False):
            st = g.groupby('budget')[m].agg(['mean', 'std'])
            ax.errorbar(st.index, st['mean'], yerr=st['std'], marker='o', capsize=3, label=model)
        ax.axhline(PAPER_BEST[n]['hgnn'][1 if m == 'acc_T' else 2], ls='--', color='grey', label='paper best HGNN (reported)')
        ax.axhline(RES[(RES.dataset == n) & (RES.model == 'RoBERTa-large')][m].mean(), ls=':', color='#DD8452', label='RoBERTa-large (raw)')
        ax.set_xscale('log'); ax.set_xlabel('number of gold labels N'); ax.set_ylabel(m); ax.set_title(f'{n} — {m}')
axes[0, 0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(RES_DIR, 'fig_label_budget.png'), dpi=150); plt.show()
BUD.groupby(['dataset', 'model', 'budget'])[['acc_T', 'f1_T']].mean().unstack('budget').round(3)
''')

md("""
## 12. Statistical significance
* **Against the paper's HGNN and the raw LMs** (deterministic): one-sample t-test of the 5 seeds of DA-HGNN-F.
* **Against the other few-label methods:** paired t-test over seeds (same label draw), plus a **McNemar test** on the test posts for every seed. We report the largest p-value across seeds, the conservative choice.
""")
code(r'''
from scipy.stats import binomtest
def mcnemar_p(n, pa, pb):
    f = FEAT[n]; m = f['test']; ca, cb = pa[m] == f['gold'][m], pb[m] == f['gold'][m]
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue

prop = RES[RES.model == 'DA-HGNN-F (proposed)']
sig = []
for n in CFG['datasets']:
    pv = prop[prop.dataset == n].sort_values('seed')
    for cm_ in ('Paper HGNN-Max (3A replication)', 'Paper HGNN-Min (3A replication)', 'RoBERTa-large', 'DistilRoBERTa'):
        b = RES[(RES.dataset == n) & (RES.model == cm_)]
        if b.empty: continue
        for m in ('acc_T', 'f1_T'):
            t = stats.ttest_1samp(pv[m], b[m].iloc[0])
            sig.append(dict(dataset=n, versus=cm_, metric=m, ours=pv[m].mean(), theirs=b[m].iloc[0], diff=pv[m].mean() - b[m].iloc[0],
                            t=t.statistic, p=t.pvalue, test='one-sample t (5 seeds)'))
    for cm_ in ('Score average + calibration + smoothing', 'RoBERTa-large + calibration', 'Supervised MLP (N labels only)',
                'MLP semi (distillation + N labels, no graph)', 'DA-HGNN semi'):
        bv = RES[(RES.dataset == n) & (RES.model == cm_)].sort_values('seed')
        for m in ('acc_T', 'f1_T'):
            t = stats.ttest_rel(pv[m].values, bv[m].values)
            sig.append(dict(dataset=n, versus=cm_, metric=m, ours=pv[m].mean(), theirs=bv[m].mean(), diff=(pv[m].values - bv[m].values).mean(),
                            t=t.statistic, p=t.pvalue, test='paired t (5 seeds)'))
SIG = pd.DataFrame(sig)
SIG['ours better & p<0.05'] = (SIG['diff'] > 0) & (SIG.p < 0.05)
SIG.to_csv(os.path.join(RES_DIR, 'significance_tests.csv'), index=False)
display(SIG.round(4))

mc = []
for n in CFG['datasets']:
    f = FEAT[n]
    for cm_, fn_pred in (('RoBERTa-large (raw)', lambda s: f['lb']),
                         ('Paper HGNN-Max (3A replication)', lambda s: None),
                         ('Score average + calibration + smoothing', lambda s: teacher(n, draw_labels(n, N, s)).argmax(1)),
                         ('MLP semi (no graph)', lambda s: run_cached(f'{n}|N{N}|MLP-semi|s{s}', None).argmax(1)),
                         ('DA-HGNN semi', lambda s: run_cached(f'{n}|N{N}|DAHGNN-semi|s{s}', None).argmax(1))):
        if cm_.startswith('Paper'):
            r = sel3a[(sel3a.dataset == n) & (sel3a.model == 'HGNN-Max')].iloc[0]
            p = os.path.join(PROC_DIR, 'grid_preds', f"{n}_{r['agg']}_{int(r.hidden)}_{int(r.layers)}_{r.nonagreed}.npy")
            if not os.path.exists(p): continue
            ref = np.load(p).astype(int); fn_pred = lambda s, ref=ref: ref
        ps = [mcnemar_p(n, LP[(n, s)].argmax(1), fn_pred(s)) for s in CFG['seeds']]
        mc.append(dict(dataset=n, versus=cm_, max_p_over_seeds=max(ps), median_p=float(np.median(ps))))
MC = pd.DataFrame(mc); MC.to_csv(os.path.join(RES_DIR, 'mcnemar_tests.csv'), index=False)
MC
''')

md("## 13. Error analysis and interpretability")
code(r'''
fig, axes = plt.subplots(1, 3, figsize=(20, 5.4))
for ax, n in zip(axes, CFG['datasets']):
    f = FEAT[n]; m = f['test'] & ~f['comp']; pred = LP[(n, 0)].argmax(1)
    labs = [i for i in range(N_CLS) if f['allowed'][i]]
    cm = confusion_matrix(f['gold'][m], pred[m], labels=labs); cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=[LABELS[i] for i in labs], yticklabels=[LABELS[i] for i in labs], ax=ax, cbar=False)
    ax.set_title(f'{n}: DA-HGNN-F on disagreed test posts (row-normalised)'); ax.set_xlabel('predicted'); ax.set_ylabel('gold')
plt.tight_layout(); plt.savefig(os.path.join(RES_DIR, 'fig_confusion_disagreed.png'), dpi=150); plt.show()

pc = []
for n in CFG['datasets']:
    f = FEAT[n]; m = f['test'] & ~f['comp']
    for model, pred in (('RoBERTa-large', f['lb']), ('DA-HGNN-F', LP[(n, 0)].argmax(1))):
        _, _, f1_, sup_ = precision_recall_fscore_support(f['gold'][m], pred[m], labels=range(N_CLS), zero_division=0)
        for i in range(N_CLS):
            if sup_[i]: pc.append(dict(dataset=n, model=model, emotion=LABELS[i], f1=f1_[i], support=int(sup_[i])))
PC = pd.DataFrame(pc).pivot_table(index=['dataset', 'emotion', 'support'], columns='model', values='f1').reset_index()
PC['gain'] = PC['DA-HGNN-F'] - PC['RoBERTa-large']
PC.to_csv(os.path.join(RES_DIR, 'per_class_f1_disagreed.csv'), index=False)
print('Per-class F1 on disagreed test posts (seed 0):'); PC.round(3)
''')

code(r'''
for n in CFG['datasets']:
    f, d = FEAT[n], DATA[n]; pred = LP[(n, 0)].argmax(1)
    idx = np.where(f['test'] & ~f['comp'] & (pred == f['gold']) & (f['lb'] != f['gold']) & d['clean'].str.len().between(25, 140).values)[0]
    print(f'=== {n}: {len(idx):,} disagreed test posts where RoBERTa-large was wrong and DA-HGNN-F is right — examples ===')
    for i in np.random.RandomState(0).choice(idx, min(4, len(idx)), replace=False):
        ph = [GRAPH[n]['vocab'][j] for j in GRAPH[n]['Mx'][i].indices][:5]
        print(f'  "{d.loc[i, "clean"]}"')
        print(f"     gold={LABELS[f['gold'][i]]} | DistilRoBERTa={LABELS[f['la'][i]]} | RoBERTa={LABELS[f['lb'][i]]} | "
              f"DA-HGNN-F={LABELS[pred[i]]} | phrases: {', '.join(ph)}")
    print()
''')

code(r'''
att = []
for n in CFG['datasets']:
    _ = train_model(n, draw_labels(n, N, 0), seed=0)          # same recipe as the cached run; LAST_MODEL holds the best weights
    for i, L in enumerate(LAST_MODEL.layers):
        for t, (names, a) in L.alpha.items():
            for nm, v in zip(names, a.tolist()):
                att.append(dict(dataset=n, layer=i + 1, node=t, relation=nm, weight=v))
ATT = pd.DataFrame(att); ATT.to_csv(os.path.join(RES_DIR, 'attention_weights.csv'), index=False)
tw = ATT[ATT.node == 'tweet']
fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(data=tw, x='dataset', y='weight', hue='relation', ax=ax, errorbar=None)
ax.set_title('Semantic attention at tweet nodes (mean over layers): phrase graph vs semantic neighbours'); ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig(os.path.join(RES_DIR, 'fig_attention.png'), dpi=150); plt.show()
ATT.pivot_table(index=['dataset', 'node', 'relation'], columns='layer', values='weight').round(3)
''')

md("## 14. Headline numbers")
code(r'''
head = []
for n in CFG['datasets']:
    g = lambda model, setting=None: RES[(RES.dataset == n) & (RES.model == model) & ((RES.setting == setting) if setting else True)]
    ours = g('DA-HGNN-F (proposed)')
    for label, ref in (('paper best HGNN (reported)', dict(acc_T=PAPER_BEST[n]['hgnn'][1], f1_T=PAPER_BEST[n]['hgnn'][2])),
                       ('paper HGNN-Max (3A replication, same test split)', g('Paper HGNN-Max (3A replication)')[['acc_T', 'f1_T']].mean()),
                       ('RoBERTa-large (raw)', g('RoBERTa-large')[['acc_T', 'f1_T']].mean()),
                       ('best few-label baseline (calibrated teacher)', g('Score average + calibration + smoothing')[['acc_T', 'f1_T']].mean())):
        head.append(dict(dataset=n, versus=label, ours_acc_T=ours.acc_T.mean(), ours_f1_T=ours.f1_T.mean(),
                         ref_acc_T=ref['acc_T'], ref_f1_T=ref['f1_T'],
                         gain_acc_pts=100 * (ours.acc_T.mean() - ref['acc_T']), gain_f1_pts=100 * (ours.f1_T.mean() - ref['f1_T'])))
HEAD = pd.DataFrame(head); HEAD.to_csv(os.path.join(RES_DIR, 'headline_gains.csv'), index=False)
print('Saved:', sorted(os.listdir(RES_DIR)))
HEAD.round(3)
''')

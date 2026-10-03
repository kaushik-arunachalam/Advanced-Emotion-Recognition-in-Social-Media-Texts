import os, time
D = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(D, 'prelude3b.py'), encoding='utf-8').read())
exec(open(os.path.join(D, 'ft_lib.py'), encoding='utf-8').read())
exec(open(os.path.join(D, 'headroom.py'), encoding='utf-8').read().split('rows = []')[0].split("exec(open(os.path.join(os.path.dirname(__file__), 'prelude3b.py')")[0] + '\n' +
     'def ev_mask(n, pred, mask):\n    f = FEAT[n]; out = {}\n    for s_, sm in ((\"C\", mask & f[\"comp\"]), (\"NC\", mask & ~f[\"comp\"]), (\"T\", mask)):\n        out[f\"acc_{s_}\"] = accuracy_score(f[\"gold\"][sm], pred[sm]); out[f\"f1_{s_}\"] = f1_score(f[\"gold\"][sm], pred[sm], average=\"weighted\", zero_division=0)\n    return out\n')
rows = []
for n in CFG['datasets']:
    f = FEAT[n]; s = 0
    lab = draw_labels(n, 500, s); val = np.zeros(len(f['gold']), bool); val[np.setdiff1d(f['pool'], lab)] = True
    t0 = time.time()
    ftd = run_cached_v2(f'{n}|N500|FT-distil|s{s}', lambda: finetune_lm(n, lab, s, CFG['lms']['distil'], 3e-5)); t1 = time.time()
    ftr = run_cached_v2(f'{n}|N500|FT-roberta-lora|s{s}', lambda: finetune_lm(n, lab, s, CFG['lms']['roberta'], 3e-4, lora=True)); t2 = time.time()
    print(f'{n}: FT-distil {t1 - t0:.0f}s, FT-roberta-LoRA {t2 - t1:.0f}s', flush=True)
    t_sm = teacher(n, lab)
    L = lambda k: np.load(os.path.join(RUN_DIR, f'{n}_N500_{k}_s{s}.npy')).astype(np.float32)
    stu, sup = L('DAHGNN-semi'), L('MLP-sup')
    nl = norm_lp
    cand = {'v1 DA-HGNN-F': fuse(n, stu, t_sm), '3-way (stu,teacher,sup)': (nl(stu) + nl(t_sm) + nl(sup)) / 3,
            'FT DistilRoBERTa': ftd, 'FT RoBERTa-large (LoRA)': ftr,
            '3-way + FT-roberta (4-way)': (nl(stu) + nl(t_sm) + nl(sup) + nl(ftr)) / 4,
            '3-way + both FT (5-way)': (nl(stu) + nl(t_sm) + nl(sup) + nl(ftr) + nl(ftd)) / 5,
            'stu ⊕ FT-roberta': (nl(stu) + nl(ftr)) / 2}
    for k, lp in cand.items():
        rows.append(dict(dataset=n, variant=k, **ev_mask(n, lp.argmax(1), val)))
R = pd.DataFrame(rows)
t = R.pivot_table(index='variant', columns='dataset', values=['acc_T', 'f1_T'], sort=False)
t[('avg', 'acc_T')] = t['acc_T'].mean(axis=1); t[('avg', 'f1_T')] = t['f1_T'].mean(axis=1)
print(t.round(4).to_string())

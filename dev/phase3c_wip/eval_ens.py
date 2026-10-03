# 5-seed evaluation of ensemble candidates on the HELD-OUT POOL (never the test split).
# Pre-registered selection rule: highest mean over datasets of (acc_T + f1_T) / 2; ties -> fewer members.
import os, sys
D = os.path.dirname(os.path.abspath(__file__))
for f_ in ('prelude3b.py', 'ft_lib.py', 'eval_lib.py'):
    exec(open(os.path.join(D, f_), encoding='utf-8').read())
use_st = '--st' in sys.argv
CANDS = {
    'v1: stu + teacher':                         ['stu', 'teacher'],
    '3-way: stu + teacher + sup':                ['stu', 'teacher', 'sup'],
    '4-way: + FT-RoBERTa':                       ['stu', 'teacher', 'sup', 'ftr'],
    '4-way: + FT-DistilRoBERTa':                 ['stu', 'teacher', 'sup', 'ftd'],
    '5-way: + both FT':                          ['stu', 'teacher', 'sup', 'ftr', 'ftd'],
    'FT-RoBERTa alone':                          ['ftr'],
    'FT-DistilRoBERTa alone':                    ['ftd'],
    'teacher + FT-RoBERTa':                      ['teacher', 'ftr'],
    'stu + FT-RoBERTa':                          ['stu', 'ftr'],
    'no graph: teacher + sup + FT-RoBERTa':      ['teacher', 'sup', 'ftr'],
}
if use_st:
    CANDS.update({'ST alone': ['st'], '5-way + ST': ['stu', 'teacher', 'sup', 'ftr', 'ftd', 'st'], '4-way + ST': ['stu', 'teacher', 'sup', 'ftr', 'st']})
rows = []
for n in CFG['datasets']:
    for s in CFG['seeds']:
        lab, m = members(n, s); val = val_mask(n, lab)
        if use_st:
            p = os.path.join(V2_RUNS, f'{n}_N500_ST-roberta-lora_ens5_s{s}.npy')
            if not os.path.exists(p): continue
            m['st'] = norm_lp(np.load(p).astype(np.float32))
        for name, keys in CANDS.items():
            rows.append(dict(dataset=n, seed=s, variant=name, k=len(keys), **ev_mask(n, ens(m, keys).argmax(1), val)))
    print(n, 'done', flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(D, 'eval_ens' + ('_st' if use_st else '') + '.csv'), index=False)
R['score'] = (R.acc_T + R.f1_T) / 2
t = R.groupby(['variant', 'dataset'], sort=False)[['acc_T', 'f1_T']].mean().unstack('dataset')
sc = R.groupby(['variant', 'dataset'], sort=False)['score'].mean().unstack('dataset').mean(axis=1)
sd = R.groupby(['variant', 'seed'], sort=False)['score'].mean().groupby('variant').std()
t[('rule', 'score')] = sc; t[('rule', 'sd over seeds')] = sd; t[('rule', 'members')] = R.groupby('variant', sort=False)['k'].first()
t = t.sort_values([('rule', 'score'), ('rule', 'members')], ascending=[False, True])
print(t.round(4).to_string())
print('\nSELECTED (pre-registered rule):', t.index[0])

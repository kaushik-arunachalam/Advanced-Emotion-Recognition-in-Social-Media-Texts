import os, time
D = os.path.dirname(os.path.abspath(__file__))
for f_ in ('prelude3b.py', 'ft_lib.py', 'eval_lib.py'):
    exec(open(os.path.join(D, f_), encoding='utf-8').read())
rows = []
for s in (0, 1):
    for n in CFG['datasets']:
        lab, m = members(n, s); val = val_mask(n, lab)
        E5 = ens(m, ['stu', 'teacher', 'sup', 'ftr', 'ftd'])
        t0 = time.time()
        st = run_cached_v2(f'{n}|N500|ST-roberta-lora|ens5|s{s}', lambda: finetune_lm_st(n, lab, s, CFG['lms']['roberta'], 3e-4, E5))
        m['st'] = norm_lp(st)
        print(f'{n} s{s}: self-training {time.time() - t0:.0f}s', flush=True)
        cand = {'v1 (stu+teacher)': ens(m, ['stu', 'teacher']), '3-way': ens(m, ['stu', 'teacher', 'sup']),
                '4-way (+ftr)': ens(m, ['stu', 'teacher', 'sup', 'ftr']), '5-way (+ftr,ftd)': E5,
                'ST student alone': m['st'], '5-way + ST': ens(m, ['stu', 'teacher', 'sup', 'ftr', 'ftd', 'st']),
                '3-way + ST': ens(m, ['stu', 'teacher', 'sup', 'st']), 'stu + teacher + ST': ens(m, ['stu', 'teacher', 'st'])}
        for k, lp in cand.items():
            rows.append(dict(dataset=n, seed=s, variant=k, **ev_mask(n, lp.argmax(1), val)))
R = pd.DataFrame(rows); R.to_csv(os.path.join(D, 'probe_st.csv'), index=False)
t = R.groupby(['variant', 'dataset'], sort=False)[['acc_T', 'f1_T']].mean().unstack('dataset')
t[('avg', 'acc_T')] = t['acc_T'].mean(axis=1); t[('avg', 'f1_T')] = t['f1_T'].mean(axis=1)
print(t.round(4).to_string())

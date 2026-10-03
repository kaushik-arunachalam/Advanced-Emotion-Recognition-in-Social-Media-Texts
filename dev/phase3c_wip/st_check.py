# Self-training check on the runs that finished (GoEmotions & Friends, seed 0), held-out pool only.
rows = []
for n in ('GoEmotions', 'Friends'):
    s = 0; lab, m = members(n, s); val = val_mask(n, lab)
    m['st'] = norm_lp(np.load(os.path.join(V2_RUNS, f'{n}_N500_ST-roberta-lora_ens5_s{s}.npy')).astype(np.float32))
    for name, keys in {'5-way (teacher of ST)': ['stu', 'teacher', 'sup', 'ftr', 'ftd'], 'FT-RoBERTa (no ST)': ['ftr'],
                       'ST student alone': ['st'], '5-way + ST': ['stu', 'teacher', 'sup', 'ftr', 'ftd', 'st'],
                       '4-way (+FT-RoBERTa)': ['stu', 'teacher', 'sup', 'ftr'], '4-way with ST instead of FT': ['stu', 'teacher', 'sup', 'st']}.items():
        rows.append(dict(dataset=n, variant=name, **ev_mask(n, ens(m, keys).argmax(1), val)))
print(pd.DataFrame(rows).pivot_table(index='variant', columns='dataset', values=['acc_T', 'f1_T'], sort=False).round(4).to_string())

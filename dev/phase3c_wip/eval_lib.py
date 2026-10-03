def ev_mask(n, pred, mask):
    f = FEAT[n]; out = {}
    for s_, sm in (('C', mask & f['comp']), ('NC', mask & ~f['comp']), ('T', mask)):
        out[f'acc_{s_}'] = accuracy_score(f['gold'][sm], pred[sm]); out[f'f1_{s_}'] = f1_score(f['gold'][sm], pred[sm], average='weighted', zero_division=0)
    out['mF1'] = f1_score(f['gold'][mask], pred[mask], average='macro', zero_division=0)
    return out

def members(n, s):
    lab = draw_labels(n, 500, s)
    L1 = lambda k: np.load(os.path.join(RUN_DIR, f'{n}_N500_{k}_s{s}.npy')).astype(np.float32)
    L2 = lambda k: np.load(os.path.join(V2_RUNS, f'{n}_N500_{k}_s{s}.npy')).astype(np.float32)
    m = dict(teacher=teacher(n, lab), stu=L1('DAHGNN-semi'), sup=L1('MLP-sup'), ftr=L2('FT-roberta-lora'), ftd=L2('FT-distil'))
    return lab, {k: norm_lp(v) for k, v in m.items()}

def ens(m, keys): return sum(m[k] for k in keys) / len(keys)

def val_mask(n, lab):
    v = np.zeros(len(FEAT[n]['gold']), bool); v[np.setdiff1d(FEAT[n]['pool'], lab)] = True; return v

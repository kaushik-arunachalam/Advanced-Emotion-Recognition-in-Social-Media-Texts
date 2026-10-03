import os
exec(open(os.path.join(os.path.dirname(__file__), 'prelude3b.py'), encoding='utf-8').read())

def ev_mask(n, pred, mask):
    f = FEAT[n]; out = {}
    for s_, sm in (('C', mask & f['comp']), ('NC', mask & ~f['comp']), ('T', mask)):
        out[f'acc_{s_}'] = accuracy_score(f['gold'][sm], pred[sm])
        out[f'f1_{s_}'] = f1_score(f['gold'][sm], pred[sm], average='weighted', zero_division=0)
    out['mF1'] = f1_score(f['gold'][mask], pred[mask], average='macro', zero_division=0)
    return out

def propagate(n, Y, alpha, iters=20):
    ei = GRAPH[n]['ei'][('tweet', 'similar', 'tweet')]
    Y0 = torch.tensor(Y, device=DEVICE, dtype=torch.float32); Z = Y0.clone()
    for _ in range(iters):
        Z = (1 - alpha) * Y0 + alpha * scatter(Z[ei[0]], ei[1], dim=0, dim_size=len(Z), reduce='mean')
    return Z.cpu().numpy()

def correct_and_smooth(n, lp, lab, a1=0.8, a2=0.8, scale=1.0):
    """C&S (Huang et al. 2021) with the N gold labels: propagate residual errors, then smooth with gold clamped."""
    f = FEAT[n]; P = np.exp(lp); Y = np.eye(N_CLS)[f['gold']]
    E = np.zeros_like(P); E[lab] = Y[lab] - P[lab]
    Eh = propagate(n, E, a1)
    sigma = np.abs(E[lab]).sum(1).mean()
    Eh = Eh * (sigma / (np.abs(Eh).sum(1, keepdims=True) + 1e-9)) * scale
    G = np.clip(P + Eh, 0, None); G[lab] = Y[lab]
    S = propagate(n, G, a2); S[:, ~f['allowed']] = 0
    return np.log(S / S.sum(1, keepdims=True) + 1e-9)

rows = []
for n in CFG['datasets']:
    f = FEAT[n]
    for s in CFG['seeds']:
        lab = draw_labels(n, 500, s)
        val = np.zeros(len(f['gold']), bool); val[np.setdiff1d(f['pool'], lab)] = True
        t_sm = teacher(n, lab); t_raw = teacher(n, lab, sm=False)
        L = lambda k: np.load(os.path.join(RUN_DIR, f'{n}_N500_{k}_s{s}.npy')).astype(np.float32)
        stu, sup, mse = L('DAHGNN-semi'), L('MLP-sup'), L('MLP-semi')
        cand = {
            'teacher (cal+smooth)': t_sm, 'DA-HGNN semi': stu, 'supervised MLP': sup, 'MLP semi': mse,
            'v1 DA-HGNN-F = stu⊕teacher': fuse(n, stu, t_sm),
            'stu⊕teacher⊕sup (1/3 each)': (norm_lp(stu) + norm_lp(t_sm) + norm_lp(sup)) / 3,
            'stu⊕sup': fuse(n, stu, sup), 'teacher⊕sup': fuse(n, t_sm, sup),
            '4-way (stu,teacher,sup,MLPsemi)': (norm_lp(stu) + norm_lp(t_sm) + norm_lp(sup) + norm_lp(mse)) / 4,
        }
        cand['C&S on v1 (gold labels propagated)'] = correct_and_smooth(n, cand['v1 DA-HGNN-F = stu⊕teacher'], lab)
        cand['C&S on 3-way'] = correct_and_smooth(n, cand['stu⊕teacher⊕sup (1/3 each)'], lab)
        for k, lp in cand.items():
            rows.append(dict(dataset=n, seed=s, variant=k, **ev_mask(n, lp.argmax(1), val)))
        # oracle headroom: any of teacher / student / supervised MLP right
        preds = np.stack([t_sm.argmax(1), stu.argmax(1), sup.argmax(1)])
        orc = np.where((preds == f['gold']).any(0), f['gold'], preds[0])
        rows.append(dict(dataset=n, seed=s, variant='ORACLE (any of teacher/stu/sup right)', **ev_mask(n, orc, val)))
    print(n, 'done', flush=True)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(os.path.dirname(__file__), 'headroom.csv'), index=False)
t = R.groupby(['variant', 'dataset'], sort=False)[['acc_T', 'f1_T']].mean().unstack('dataset')
t[('avg', 'acc_T')] = t['acc_T'].mean(axis=1); t[('avg', 'f1_T')] = t['f1_T'].mean(axis=1)
print(t.round(4).to_string())
print()
print(R.groupby(['variant', 'dataset'], sort=False)[['acc_C', 'acc_NC', 'mF1']].mean().unstack('dataset').round(3).to_string())

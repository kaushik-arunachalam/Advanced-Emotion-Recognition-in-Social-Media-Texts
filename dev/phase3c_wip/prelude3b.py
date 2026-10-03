display = print

import os, sys, re, json, html, random, time, itertools, warnings, subprocess
from collections import Counter, defaultdict

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch_geometric'], check=True)
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_DIR = '/content/drive/MyDrive/Capstone Project'   # change if the folder is elsewhere in Drive
else:
    PROJECT_DIR = os.getcwd()

PROC_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
REP_DIR = os.path.join(PROJECT_DIR, 'results', 'replication')
RES_DIR = os.path.join(PROJECT_DIR, 'results', 'improved')
RUN_DIR = os.path.join(RES_DIR, 'runs')            # cached predictions of every trained model (lets re-runs resume)
for d in (RES_DIR, RUN_DIR):
    os.makedirs(d, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import scipy.sparse as sp
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging as hf_logging
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
hf_logging.set_verbosity_error()
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Project dir:', PROJECT_DIR)
print('Device     :', DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')

CFG = dict(
    lms={'distil': 'j-hartmann/emotion-english-distilroberta-base', 'roberta': 'j-hartmann/emotion-english-roberta-large'},
    labels=['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'],
    datasets=['GoEmotions', 'Friends', 'TEC'],
    pool_frac=0.2, split_seed=42,            # 20% labelled pool (dev) / 80% untouched test
    n_labels=500,                            # Setting B label budget per dataset
    top_k_phrases=4, textrank_window=3, edge_mean=0.5, edge_std=0.1,   # paper's phrase rules
    knn_k=10,                                # semantic tweet-tweet neighbours
    hidden=256, layers=2, dropout=0.3, attn_size=64,
    lr=2e-3, weight_decay=1e-4, epochs=200, patience=25,
    mask_p=0.3,                              # score-masking (disagreement simulation) rate
    fusion_w=0.5,                            # weight of DA-HGNN vs calibrated teacher in the final decision
    smooth_alpha=0.5, smooth_iters=10,
    seeds=[0, 1, 2, 3, 4], ablation_seeds=[0, 1, 2],
    budgets=[50, 100, 200, 500, 1000, 2000], budget_seeds=[0, 1, 2],
)
LABELS = CFG['labels']; L2I = {l: i for i, l in enumerate(LABELS)}; N_CLS = len(LABELS); NEU = L2I['neutral']
COLS = {m: [f'{m}_{l}' for l in LABELS] for m in ('distil', 'roberta', 'distil_v2')}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

DATA = {}
for n in CFG['datasets']:
    p = os.path.join(PROC_DIR, f'{n.lower()}_scored.csv')
    if not os.path.exists(p):
        raise FileNotFoundError(f'{p} not found — run Capstone_Phase3A_Paper_Replication.ipynb first.')
    d = pd.read_csv(p, keep_default_na=False)
    pool, _ = train_test_split(np.arange(len(d)), train_size=CFG['pool_frac'], random_state=CFG['split_seed'], stratify=d['gold'])
    d['split'] = 'test'; d.loc[pool, 'split'] = 'pool'
    DATA[n] = d
summary = pd.DataFrame({n: {'posts': len(d), 'pool (labelled candidates)': (d.split == 'pool').sum(), 'test (80%)': (d.split == 'test').sum(),
                            'classes in taxonomy': d.gold.nunique(), f'N = {CFG["n_labels"]} labels = % of data': round(100 * CFG['n_labels'] / len(d), 1)}
                        for n, d in DATA.items()}).T
summary

def embed(model_name, texts, bs=64):
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    if DEVICE == 'cuda': m = m.half()
    idx = np.argsort([len(t) for t in texts]); out = np.zeros((len(texts), m.config.hidden_size), np.float16)
    with torch.inference_mode():
        for s in range(0, len(texts), bs):
            b = idx[s:s + bs]
            enc = tok([texts[i] for i in b], padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
            h = m(**enc).last_hidden_state.float(); msk = enc['attention_mask'].unsqueeze(-1).float()
            out[b] = ((h * msk).sum(1) / msk.sum(1)).cpu().numpy().astype(np.float16)
    del m
    if DEVICE == 'cuda': torch.cuda.empty_cache()
    return out

def js_div(A, B):
    M = (A + B) / 2
    kl = lambda P, Q: (P * (np.log(P + 1e-9) - np.log(Q + 1e-9))).sum(1)
    return 0.5 * kl(A, M) + 0.5 * kl(B, M)

FEAT = {}
for n, d in DATA.items():
    E = []
    for k, hf in CFG['lms'].items():
        f = os.path.join(PROC_DIR, f'{n.lower()}_emb_{k}.npy')
        if not os.path.exists(f):
            print(f'{n}: embedding with {hf} ...'); np.save(f, embed(hf, d['clean'].tolist()))
        E.append(np.load(f).astype(np.float32))
    E = np.hstack(E); E = (E - E.mean(0)) / (E.std(0) + 1e-6)
    A = d[COLS['distil']].values.astype(np.float32); B = d[COLS['roberta']].values.astype(np.float32); M = (A + B) / 2
    desc = np.stack([A.max(1), B.max(1), -(M * np.log(M + 1e-9)).sum(1) / np.log(N_CLS), js_div(A, B)], 1).astype(np.float32)
    FEAT[n] = dict(A=A, B=B, M=M, V2=d[COLS['distil_v2']].values.astype(np.float32), desc=desc, E=E.astype(np.float32),
                   la=A.argmax(1), lb=B.argmax(1), comp=A.argmax(1) == B.argmax(1), gold=d['gold'].map(L2I).values,
                   test=(d['split'] == 'test').values, pool=np.where(d['split'] == 'pool')[0],
                   allowed=np.array([l in set(d['gold']) for l in LABELS]))
    print(f'{n:10s} embeddings {E.shape}  agreed {FEAT[n]["comp"].mean():.1%}  classes allowed: {[l for l, a in zip(LABELS, FEAT[n]["allowed"]) if a]}')

STOP = set(ENGLISH_STOP_WORDS) | {"i'm", "it's", "that's", "you're", "i've", "i'll", "he's", "she's", "they're", "we're",
                                  "there's", "what's", "let's", "i'd", "you'll", "you've", "we've", "they've", 'im', 'dont', 'thats', 'youre'}
TOKEN_RE = re.compile(r"[a-z][a-z']+")

def textrank_phrases(text, top_k=CFG['top_k_phrases'], window=CFG['textrank_window']):
    words = [w.strip("'") for w in TOKEN_RE.findall(text.lower())]
    cands = [w for w in words if w not in STOP and len(w) > 2]
    if not cands:
        return []
    g = nx.Graph(); g.add_nodes_from(cands)
    for i in range(len(cands)):
        for j in range(i + 1, min(i + window, len(cands))):
            if cands[i] != cands[j]:
                g.add_edge(cands[i], cands[j])
    rank = nx.pagerank(g) if g.number_of_edges() else {w: 1.0 for w in g}
    top = set(sorted(rank, key=rank.get, reverse=True)[:top_k])
    phrases, run = set(top), []
    for i, w in enumerate(words):
        if w in top and (not run or i == run[-1][0] + 1):
            run.append((i, w))
        else:
            if len(run) > 1: phrases.add(' '.join(x for _, x in run))
            run = [(i, w)] if w in top else []
    if len(run) > 1: phrases.add(' '.join(x for _, x in run))
    return sorted(phrases)

def knn_edges(E, k):
    X = F.normalize(torch.tensor(E, device=DEVICE), dim=1); src, dst = [], []
    for s in range(0, len(X), 4096):
        sim = X[s:s + 4096] @ X.T
        sim[torch.arange(sim.shape[0]), torch.arange(s, s + sim.shape[0])] = -1
        src.append(sim.topk(k, dim=1).indices.reshape(-1))
        dst.append(torch.arange(s, s + sim.shape[0], device=DEVICE).repeat_interleave(k))
    return torch.stack([torch.cat(src), torch.cat(dst)])            # neighbour -> tweet

GRAPH = {}
for n, d in DATA.items():
    f = FEAT[n]
    phrases = d['clean'].map(textrank_phrases)
    vocab = sorted({p for ps in phrases for p in ps}); p2i = {p: i for i, p in enumerate(vocab)}
    tp = np.array([(t, p2i[p]) for t, ps in enumerate(phrases) for p in ps]).T
    Mx = sp.csr_matrix((np.ones(tp.shape[1], np.float32), (tp[0], tp[1])), shape=(len(d), len(vocab)))
    Mc = sp.diags(f['comp'].astype(np.float32)) @ Mx
    sup = np.asarray(Mc.sum(0)).ravel(); den = np.maximum(sup, 1)[:, None]
    S = np.hstack([f['A'], f['B']])
    mean = np.asarray(Mc.T @ S) / den; std = np.sqrt(np.clip(np.asarray(Mc.T @ S ** 2) / den - mean ** 2, 0, None))
    cnt = np.maximum(np.asarray(Mx.sum(0)).ravel(), 1)[:, None]
    xp = np.hstack([mean, std, (sup > 0)[:, None], np.asarray(Mx.T @ f['E']) / cnt]).astype(np.float32)
    m7 = np.asarray(Mc.T @ f['M']) / den; s7 = np.sqrt(np.clip(np.asarray(Mc.T @ f['M'] ** 2) / den - m7 ** 2, 0, None))
    pe = (m7 > CFG['edge_mean']) & (s7 < CFG['edge_std']) & (sup[:, None] > 0)
    pe_idx = np.array(np.nonzero(pe))
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.long, device=DEVICE)
    ei = {('tweet', 'has', 'phrase'): t(tp), ('phrase', 'in', 'tweet'): t(tp[::-1]),
          ('phrase', 'signals', 'emotion'): t(pe_idx), ('emotion', 'signalled_by', 'phrase'): t(pe_idx[::-1]),
          ('tweet', 'similar', 'tweet'): knn_edges(f['E'], CFG['knn_k'])}
    GRAPH[n] = dict(phrases=phrases, vocab=vocab, Mx=Mx, pe=pe, ei=ei, xp=torch.tensor(xp, device=DEVICE), xe=torch.eye(N_CLS, device=DEVICE))
    print(f'{n:10s} tweets={len(d):6,d} phrases={len(vocab):6,d} T-P={tp.shape[1]:7,d} P-E={pe_idx.shape[1]:6,d} T-T(kNN)={ei[("tweet","similar","tweet")].shape[1]:7,d} '
          f'phrases without agreed tweet={int((sup == 0).sum()):,} (now have embedding features)')

class HeteroGNNConv(nn.Module):                       # paper Eq. (1)-(2)
    def __init__(self, in_src, in_dst, out):
        super().__init__()
        self.lin_dst, self.lin_src, self.lin_update = nn.Linear(in_dst, out), nn.Linear(in_src, out), nn.Linear(2 * out, out)
    def forward(self, x_src, x_dst, ei):
        agg = scatter(x_src[ei[0]], ei[1], dim=0, dim_size=x_dst.size(0), reduce='mean')
        return self.lin_update(torch.cat([self.lin_dst(x_dst), self.lin_src(agg)], dim=-1))

class HeteroLayer(nn.Module):                          # paper Eq. (4)-(5): semantic attention over relations
    def __init__(self, dims, out, edge_types, attn_size):
        super().__init__()
        self.edge_types = edge_types
        self.convs = nn.ModuleDict({'__'.join(et): HeteroGNNConv(dims[et[0]], dims[et[2]], out) for et in edge_types})
        self.attn = nn.ModuleDict({t: nn.Sequential(nn.Linear(out, attn_size), nn.Tanh(), nn.Linear(attn_size, 1, bias=False))
                                   for t in {et[2] for et in edge_types}})
        self.alpha = {}
    def forward(self, x, ei):
        msgs, names = defaultdict(list), defaultdict(list)
        for et in self.edge_types:
            msgs[et[2]].append(self.convs['__'.join(et)](x[et[0]], x[et[2]], ei[et])); names[et[2]].append(f'{et[0]}→{et[2]}')
        out = {}
        for t, hs in msgs.items():
            H = torch.stack(hs, 1)
            if len(hs) > 1:
                a = torch.softmax(self.attn[t](H).mean(0), 0); self.alpha[t] = (names[t], a.detach().view(-1))
                out[t] = (a.unsqueeze(0) * H).sum(1)
            else:
                out[t] = H[:, 0]
        return out

class DAHGNN(nn.Module):
    def __init__(self, d_tweet, d_phrase, edge_types, hidden=CFG['hidden'], layers=CFG['layers'], dropout=CFG['dropout']):
        super().__init__()
        self.proj = nn.ModuleDict({'tweet': nn.Linear(d_tweet, hidden), 'phrase': nn.Linear(d_phrase, hidden), 'emotion': nn.Linear(N_CLS, hidden)})
        self.layers = nn.ModuleList([HeteroLayer({k: hidden for k in self.proj}, hidden, edge_types, CFG['attn_size']) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.ModuleDict({k: nn.LayerNorm(hidden) for k in self.proj}) for _ in range(layers)])
        self.skip, self.cls, self.dropout = nn.Linear(d_tweet, hidden), nn.Linear(2 * hidden, N_CLS), dropout
    def forward(self, x_tweet, g):
        x = {'tweet': self.proj['tweet'](x_tweet), 'phrase': self.proj['phrase'](g['xp']), 'emotion': self.proj['emotion'](g['xe'])}
        x = {k: F.dropout(F.relu(v), self.dropout, self.training) for k, v in x.items()}
        for layer, norm in zip(self.layers, self.norms):
            h = layer(x, g['ei'])
            x = {k: F.dropout(F.relu(norm[k](h.get(k, torch.zeros_like(v)) + v)), self.dropout, self.training) for k, v in x.items()}
        return self.cls(torch.cat([x['tweet'], F.relu(self.skip(x_tweet))], -1))

class MLP(nn.Module):                                  # same features, no graph (ablation / baseline)
    def __init__(self, d_in, hidden=CFG['hidden'], dropout=CFG['dropout']):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, N_CLS))
    def forward(self, x, g=None): return self.net(x)


def tweet_input(f, score_mask=None, use_scores=True, use_emb=True, paper_feats=False):
    if paper_feats:                                    # the paper's tweet features: DistilRoBERTa scores only
        return torch.tensor(f['A'], device=DEVICE)
    parts = []
    if use_scores:
        S = np.hstack([f['A'], f['B'], f['desc']])
        m = np.zeros((len(S), 1), np.float32) if score_mask is None else score_mask[:, None].astype(np.float32)
        parts += [S * (1 - m), m]
    if use_emb:
        parts.append(f['E'])
    return torch.tensor(np.hstack(parts), dtype=torch.float32, device=DEVICE)

def run_cached(key, fn):
    path = os.path.join(RUN_DIR, re.sub(r'[^A-Za-z0-9_.=-]+', '_', key) + '.npy')
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    out = fn(); np.save(path, out.astype(np.float16)); return out

def train_model(n, lab=(), seed=0, graph=True, kd=True, soft=True, weighted=True, mask_p=CFG['mask_p'],
                use_emb=True, use_scores=True, knn=True, phrases=True, paper_feats=False):
    # returns log-probabilities for every tweet; lab = indices of gold-labelled posts (empty -> label-free)
    f, G = FEAT[n], GRAPH[n]
    set_seed(seed); rng = np.random.RandomState(seed)
    ei = {k: v for k, v in G['ei'].items() if (knn or k[1] != 'similar') and (phrases or k[1] == 'similar')}
    g = dict(G, ei=ei)
    q = f['A'] * f['B']; q[:, ~f['allowed']] = 0; q = q / q.sum(1, keepdims=True)   # product of experts over the taxonomy
    if not soft: q = np.eye(N_CLS, dtype=np.float32)[q.argmax(1)]
    w = np.sqrt(f['A'].max(1) * f['B'].max(1)) if weighted else np.ones(len(q), np.float32)
    q, w = torch.tensor(q, device=DEVICE), torch.tensor(w, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(f['gold'], device=DEVICE); disallowed = torch.tensor(~f['allowed'], device=DEVICE)
    kd_idx = np.where(f['comp'])[0]
    lab = np.array(lab, dtype=int); rng.shuffle(lab)
    if len(lab):
        n_va = max(10, len(lab) // 5); va_l, tr_l = lab[:n_va], lab[n_va:]
        kd_tr = kd_idx
    else:
        kd_tr, kd_va = train_test_split(kd_idx, test_size=0.1, random_state=seed, stratify=f['la'][kd_idx])
    x0 = tweet_input(f, use_scores=use_scores, use_emb=use_emb, paper_feats=paper_feats)
    model = (DAHGNN(x0.shape[1], g['xp'].shape[1], list(ei.keys())) if graph else MLP(x0.shape[1])).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    T = lambda a: torch.tensor(a, device=DEVICE)
    best, wait, keep = 1e9, 0, None
    for ep in range(CFG['epochs']):
        model.train(); opt.zero_grad()
        if mask_p > 0 and use_scores and not paper_feats:
            sm = np.zeros(len(q), bool); sm[kd_idx] = rng.rand(len(kd_idx)) < mask_p
            x = tweet_input(f, sm, use_scores, use_emb)
        else:
            x = x0
        out = model(x, g).masked_fill(disallowed, -1e4)
        loss = 0
        if kd:
            kt = T(kd_tr); loss = loss + (-(q[kt] * F.log_softmax(out[kt], -1)).sum(1) * w[kt]).sum() / w[kt].sum()
        if len(lab):
            loss = loss + F.cross_entropy(out[T(tr_l)], y[T(tr_l)])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            oe = model(x0, g).masked_fill(disallowed, -1e4)
            if len(lab):
                vl = F.cross_entropy(oe[T(va_l)], y[T(va_l)]).item()
            else:
                kv = T(kd_va); vl = ((-(q[kv] * F.log_softmax(oe[kv], -1)).sum(1) * w[kv]).sum() / w[kv].sum()).item()
        if vl < best - 1e-4:
            best, wait, keep = vl, 0, F.log_softmax(oe, -1).cpu().numpy()
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= CFG['patience']: break
    global LAST_MODEL                                  # kept for inspecting attention weights
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): model(x0, g)
    LAST_MODEL = model
    return keep

def norm_lp(lp): return lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)

def align(f, lp):
    z = lp.copy(); z[:, ~f['allowed']] = -1e4; return norm_lp(z)

def fit_vs(logits, y, reg=1e-3):
    z, yy = torch.tensor(logits, dtype=torch.float32), torch.tensor(y)
    logT, b = torch.zeros(1, requires_grad=True), torch.zeros(z.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([logT, b], lr=0.1, max_iter=300)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(z / logT.exp() + b, yy) + reg * (b ** 2).sum(); loss.backward(); return loss
    opt.step(closure)
    return float(logT.exp()), b.detach().numpy()

def calibrate(f, lp, lab, per_group=True):
    out = lp.copy(); fm = np.zeros(len(lp), bool); fm[lab] = True
    groups = (('agreed', f['comp']), ('disagreed', ~f['comp'])) if per_group else (('all', np.ones(len(lp), bool)),)
    for _, g in groups:
        if (fm & g).sum() < 2: continue
        Tm, b = fit_vs(lp[fm & g], f['gold'][fm & g]); out[g] = lp[g] / Tm + b
    return norm_lp(out)

def smooth(n, lp, alpha=CFG['smooth_alpha'], iters=CFG['smooth_iters']):
    ei = GRAPH[n]['ei'][('tweet', 'similar', 'tweet')]
    Y0 = torch.tensor(np.exp(lp), device=DEVICE); Y = Y0.clone()
    for _ in range(iters):
        Y = (1 - alpha) * Y0 + alpha * scatter(Y[ei[0]], ei[1], dim=0, dim_size=len(Y), reduce='mean')
    return np.log(Y.cpu().numpy() + 1e-9)

def teacher(n, lab=None, cal=True, sm=True):
    f = FEAT[n]; lp = np.log(f['M'] + 1e-9)
    if cal and lab is not None and len(lab): lp = calibrate(f, lp, lab)
    lp = align(f, lp)
    return smooth(n, lp) if sm else lp

def fuse(n, s_lp, t_lp, w=CFG['fusion_w']):
    return w * norm_lp(s_lp) + (1 - w) * norm_lp(t_lp)

def draw_labels(n, size, seed):
    pool = FEAT[n]['pool']
    return np.random.RandomState(1000 + seed).choice(pool, min(size, len(pool)), replace=False)

def evaluate(n, pred):
    f = FEAT[n]; m = f['test']; out = {}
    for s, sm in (('C', m & f['comp']), ('NC', m & ~f['comp']), ('T', m)):
        out[f'acc_{s}'] = accuracy_score(f['gold'][sm], pred[sm])
        out[f'f1_{s}'] = f1_score(f['gold'][sm], pred[sm], average='weighted', zero_division=0)
    out['macroF1_T'] = f1_score(f['gold'][m], pred[m], average='macro', zero_division=0)
    return out

MAIN_COLS = ['acc_C', 'acc_NC', 'acc_T', 'f1_C', 'f1_NC', 'f1_T', 'macroF1_T']
def summarize(df, by=('dataset', 'setting', 'model'), cols=MAIN_COLS):
    g = df.groupby(list(by), sort=False)[cols]
    m, s = g.mean(), g.std().fillna(0)
    return m.round(3).astype(str) + np.where(s.values > 0, ' ± ' + s.round(3).astype(str), '')
# Fine-tuning the emotion LMs on the N gold labels (full fine-tuning for DistilRoBERTa, LoRA for RoBERTa-large).
import math, copy
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

V2_DIR = os.path.join(PROJECT_DIR, 'results', 'improved_v2'); V2_RUNS = os.path.join(V2_DIR, 'runs')
os.makedirs(V2_RUNS, exist_ok=True)

def run_cached_v2(key, fn):
    path = os.path.join(V2_RUNS, re.sub(r'[^A-Za-z0-9_.=-]+', '_', key) + '.npy')
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    out = fn(); np.save(path, out.astype(np.float16)); return out

def finetune_lm(n, lab, seed, hf, lr, lora=False, epochs=10, bs=16, patience=2, max_len=128):
    f, d = FEAT[n], DATA[n]
    set_seed(seed); rng = np.random.RandomState(seed)
    lab = np.array(lab, dtype=int); rng.shuffle(lab)
    n_va = max(10, len(lab) // 5); va, tr = lab[:n_va], lab[n_va:]          # same split rule as train_model
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForSequenceClassification.from_pretrained(hf)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    order = [next(k for k, v in id2label.items() if v == l) for l in LABELS]   # model column of our label i
    to_model = np.array(order)
    if lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type='SEQ_CLS', r=16, lora_alpha=32, lora_dropout=0.1,
                                                 target_modules=['query', 'key', 'value']))
    model.to(DEVICE)
    texts = d['clean'].tolist()
    dis = torch.zeros(N_CLS, dtype=torch.bool, device=DEVICE); dis[torch.tensor(to_model[~f['allowed']], dtype=torch.long)] = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(tr) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler('cuda')
    y_model = torch.tensor(to_model[f['gold']], device=DEVICE)

    def logits_for(idx, train=False):
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(DEVICE)
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(**enc).logits.float()
        return out.masked_fill(dis, -1e4)

    def predict(idx, bsz=128):
        model.eval(); outs = []
        order_idx = np.argsort([len(texts[i]) for i in idx]); res = np.zeros((len(idx), N_CLS), np.float32)
        with torch.no_grad():
            for s in range(0, len(idx), bsz):
                b = order_idx[s:s + bsz]
                res[b] = torch.log_softmax(logits_for(idx[b]), -1).cpu().numpy()
        return res

    best, best_state, wait = 1e9, None, 0
    for ep in range(epochs):
        model.train(); perm = rng.permutation(tr)
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            loss = F.cross_entropy(logits_for(b, True), y_model[torch.tensor(b, device=DEVICE)])
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        lp_va = predict(va)
        vl = F.cross_entropy(torch.tensor(lp_va), torch.tensor(to_model[f['gold'][va]])).item()
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if 'lora' in k or 'classifier' in k or not lora}
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state, strict=False)
    lp_model = predict(np.arange(len(texts)))
    del model; torch.cuda.empty_cache()
    return lp_model[:, order]                                                   # back to our label order


def finetune_lm_st(n, lab, seed, hf, lr, ens_lp, conf_q=0.5, lora=True, epochs=3, bs=32, patience=1, max_len=128, pseudo_w=0.5):
    """Self-training (Noisy Student): gold labels (CE) + the ensemble's confident soft labels on unlabelled posts (KL)."""
    f, d = FEAT[n], DATA[n]
    set_seed(seed); rng = np.random.RandomState(seed)
    lab = np.array(lab, dtype=int); rng.shuffle(lab)
    n_va = max(10, len(lab) // 5); va, tr = lab[:n_va], lab[n_va:]
    P = np.exp(ens_lp); P[:, ~f['allowed']] = 0; P = P / P.sum(1, keepdims=True)
    unl = np.setdiff1d(np.arange(len(P)), lab)
    conf = P[unl].max(1); keep = unl[conf >= np.quantile(conf, conf_q)]                # most confident (1 - conf_q) share
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForSequenceClassification.from_pretrained(hf)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    order = [next(k for k, v in id2label.items() if v == l) for l in LABELS]; to_model = np.array(order)
    if lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type='SEQ_CLS', r=16, lora_alpha=32, lora_dropout=0.1, target_modules=['query', 'key', 'value']))
    model.to(DEVICE); texts = d['clean'].tolist()
    dis = torch.zeros(N_CLS, dtype=torch.bool, device=DEVICE); dis[torch.tensor(to_model[~f['allowed']], dtype=torch.long)] = True
    # targets in MODEL column order
    Q = np.zeros((len(P), N_CLS), np.float32); Q[:, to_model] = P
    G = np.zeros((len(P), N_CLS), np.float32); G[np.arange(len(P)), to_model[f['gold']]] = 1
    reps = max(1, len(keep) // (4 * len(tr)))                                           # oversample gold so it is ~20% of each epoch
    items = np.concatenate([np.repeat(tr, reps), keep]); is_gold = np.concatenate([np.ones(len(tr) * reps, bool), np.zeros(len(keep), bool)])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(items) / bs); sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler('cuda')
    Qt, Gt = torch.tensor(Q, device=DEVICE), torch.tensor(G, device=DEVICE)
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
        model.train(); perm = rng.permutation(len(items))
        for s in range(0, len(perm), bs):
            j = perm[s:s + bs]; b = items[j]; gmask = torch.tensor(is_gold[j], device=DEVICE)
            bt = torch.tensor(b, device=DEVICE)
            tgt = torch.where(gmask[:, None], Gt[bt], Qt[bt]); w = torch.where(gmask, 1.0, pseudo_w)
            loss = ((-(tgt * torch.log_softmax(logits_for(b), -1)).sum(1)) * w).sum() / w.sum()
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        lp_va = predict(va); vl = F.cross_entropy(torch.tensor(lp_va), torch.tensor(to_model[f['gold'][va]])).item()
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if 'lora' in k or 'classifier' in k or not lora}
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state, strict=False)
    lp_model = predict(np.arange(len(texts))); del model; torch.cuda.empty_cache()
    return lp_model[:, order]

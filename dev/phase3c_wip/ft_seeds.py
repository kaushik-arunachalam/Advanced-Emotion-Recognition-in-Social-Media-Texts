import os, time
D = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(D, 'prelude3b.py'), encoding='utf-8').read())
exec(open(os.path.join(D, 'ft_lib.py'), encoding='utf-8').read())
for s in CFG['seeds']:
    for n in CFG['datasets']:
        lab = draw_labels(n, 500, s)
        run_cached_v2(f'{n}|N500|FT-distil|s{s}', lambda: finetune_lm(n, lab, s, CFG['lms']['distil'], 3e-5))
        run_cached_v2(f'{n}|N500|FT-roberta-lora|s{s}', lambda: finetune_lm(n, lab, s, CFG['lms']['roberta'], 3e-4, lora=True))
    print('seed', s, 'done', time.strftime('%X'), flush=True)

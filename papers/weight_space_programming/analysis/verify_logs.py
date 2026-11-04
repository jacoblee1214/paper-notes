"""공개 로그(lm_understand-logs, CC BY 4.0)로 블로그 수치를 다시 계산한다. CPU만, numpy/scipy만 쓴다.

  python3 analysis/verify_logs.py [logs_official 경로]   → analysis/verify_logs_results.json
"""
import glob, gzip, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'logs_official')
J = lambda p: json.load(open(os.path.join(L, p)))
rng = np.random.default_rng(0)
out = {}


def mlp_params(*dims, hidden_layers=None):
    return sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


# ---------------------------------------------------------------- C1 신경망 크기
out['C1_param_counts'] = {
    'exp1 500-32-32-5': mlp_params(500, 32, 32, 5),
    'exp2 160-64-64-144': mlp_params(160, 64, 64, 144),
    'exp4 679-128x6-16': mlp_params(679, 128, 128, 128, 128, 128, 128, 16),
    'exp5 600-32-32-6': mlp_params(600, 32, 32, 6),
    'exp6 243-512x6-82': mlp_params(243, 512, 512, 512, 512, 512, 512, 82),
}

# ---------------------------------------------------------------- C2 실험 1·2 정답 수
s1, s2 = J('exp1_which_game/eval/scores.json'), J('exp2_diffusion/eval/scores.json')
rows = lambda s: s if isinstance(s, list) else list(s.values())
out['C2_exp1_exp2'] = {
    'exp1_correct': sum(bool(r.get('correct')) for r in rows(s1)), 'exp1_n': len(rows(s1)),
    'exp2_correct': sum(bool(r.get('correct')) for r in rows(s2)), 'exp2_n': len(rows(s2)),
}

# ---------------------------------------------------------------- C3 실험 3: 복사 헤드 기준과 확률의 크기
heads = J('exp3_language/eval/heads.json')
cut = {}
for lang, hs in heads.items():
    items = hs.items() if isinstance(hs, dict) else [(h.get('head'), h) for h in hs]
    ind = {k: v['induct'] for k, v in items}
    cut[lang] = {'>0.2': sorted(k for k, v in ind.items() if v > 0.2), '>0.3': sorted(k for k, v in ind.items() if v > 0.3),
                 'max_other': round(max(v for v in ind.values() if v <= 0.2), 4)}
probs = {'python_r0': (0.56, 0.13, 0.31), 'python_r1': (0.52, 0.08, 0.40), 'romaji_r0': (0.14, 0.55, 0.31),
         'romaji_r1': (0.11, 0.62, 0.27), 'english_r0': (0.30, 0.12, 0.58), 'english_r1': (0.35, 0.17, 0.48)}
true_idx = {'python': 0, 'romaji': 1, 'english': 2}
p_true = [p[true_idx[k.split('_')[0]]] for k, p in probs.items()]
out['C3_exp3'] = {'copy_heads': cut, 'mean_p_true': round(float(np.mean(p_true)), 3),
                  'logloss': round(float(-np.mean(np.log(p_true))), 3), 'logloss_uniform': round(math.log(3), 3),
                  'note': '보기 3개가 주어진 문제. 균등 추측의 log loss는 ln 3'}

# ---------------------------------------------------------------- C4 실험 4: 바뀐 가중치의 개수와 크기
b = np.load(os.path.join(L, 'exp4_fix_car/networks/ppo_final_seed0.npz'))
e = np.load(os.path.join(L, 'exp4_fix_car/networks/edited_actor.npz'))
pre = 'params/GeneralActorCriticRNN_0/'
mp = {'embed.W': 'params/Dense_0/kernel', 'embed.b': 'params/Dense_0/bias', 'actor_head.W': pre + 'Dense_0/kernel',
      'actor_head.b': pre + 'Dense_0/bias'}
for i in range(5):
    mp[f'actor{i}.W'] = pre + f'DenseAndActivation_{i}/Dense_0/kernel'
    mp[f'actor{i}.b'] = pre + f'DenseAndActivation_{i}/Dense_0/bias'
rep = J('exp4_fix_car/eval/oneshot_result.json')
tot_b = tot_d = 0.0
per = {}
for k, bk in mp.items():
    B, E = b[bk].astype(np.float64), e[k].astype(np.float64)
    D = E - B
    tot_b += (B ** 2).sum(); tot_d += (D ** 2).sum()
    per[k] = {'max_abs_base': round(float(np.abs(B).max()), 3), 'max_abs_edited': round(float(np.abs(E).max()), 3)}
out['C4_exp4'] = {'reported_changed': rep['changes']['changed'], 'entries': rep['changes']['entries'],
                  'frac_changed': round(rep['changes']['frac_changed'], 4),
                  'reported_rel_change': round(rep['changes']['rel_change'], 2),
                  'recomputed_rel_change_fro': round(math.sqrt(tot_d / tot_b), 2),
                  'per_tensor_rel_change': {k: round(v['rel_change'], 2) for k, v in rep['changes']['per_tensor'].items()},
                  'max_abs': per,
                  'test': rep.get('test', {}).get('success') if isinstance(rep.get('test'), dict) else rep.get('test')}

# ---------------------------------------------------------------- C5 실험 5: MinAtar 점수, 에피소드 길이 상한, 씨앗 편차
mn = J('exp5_write_from_scratch/eval/minatar_eval.json')
c5 = {}
for g, d in mn.items():
    ppo100 = [v['mean'] for k, v in d.items() if k.startswith('ppo100')]
    ppo500 = [v['mean'] for k, v in d.items() if k.startswith('ppo500')]
    nz = None
    f = os.path.join(L, f'exp5_write_from_scratch/networks/{g}/hand.npz')
    if os.path.exists(f):
        z = np.load(f); nz = int(sum((z[k] != 0).sum() for k in z.files))
    c5[g] = {'hand_mean': d['hand']['mean'], 'hand_se': round(d['hand']['se'], 2), 'hand_mean_len': d['hand']['mean_length'],
             'hand_len_vs_cap': round(d['hand']['mean_length'] / 10000, 3),
             'ppo100_best': max(ppo100), 'ppo100_median': float(np.median(ppo100)), 'ppo100_n': len(ppo100),
             'ppo500_best': max(ppo500), 'ppo500_median': float(np.median(ppo500)), 'ppo500_n': len(ppo500),
             'ppo500_best_mean_len': max((v for k, v in d.items() if k.startswith('ppo500')), key=lambda v: v['mean'])['mean_length'],
             'hand_nonzero_weights': nz}
out['C5_exp5'] = c5

# ---------------------------------------------------------------- C6 실험 6: 바둑 승률과 신뢰구간
c6 = {}
for bud in (0, 10, 25, 50):
    d = J(f'exp6_go_from_scratch/eval/budget_{bud}/eval_submission.json')
    c6[f'budget_{bud}'] = {k: {'score': round(d[k]['score'], 4), 'games': d[k]['games'], 'wilson95': wilson(d[k]['wins'], d[k]['games'])}
                           for k in ('ppo_s0', 'ppo_s2') if k in d}
    c6[f'budget_{bud}']['other_opponents'] = {k: round(v['score'], 4) for k, v in d.items() if isinstance(v, dict) and 'score' in v and k not in ('ppo_s0', 'ppo_s2')}
h2h = J('exp6_go_from_scratch/eval/ppo_same_size/ppo_big_eval/h2h_lr_scan.json')
c6['h2h_same_size'] = {k: (round(v['score'], 4) if isinstance(v, dict) and 'score' in v else v) for k, v in h2h.items()}
out['C6_exp6'] = c6

# ---------------------------------------------------------------- C7 실험 7: 우연 확률
out['C7_exp7'] = {'weights_only': '4번 모두 판단 실패(블로그)', 'attention_tool_correct': '3/3',
                  'chance_3_of_3_binary': 1 / 8, 'note': '쌍 3개, A/B 동전 던지기. 우연히 다 맞힐 확률 12.5%'}


# ---------------------------------------------------------------- C8 실험 8: 문제별 짝 비교(부트스트랩)
def per_problem(run, bench, root='exp8_math/eval'):
    d = {}
    for f in glob.glob(os.path.join(L, root, run, 'parts', f'{bench}_s*', f'{bench}_per_problem.json')):
        for r in json.load(open(f)):
            d[r['id']] = np.mean(r['correct'])
    return d


def paired(base, ed, n_boot=10000):
    ids = sorted(set(base) & set(ed))
    x = np.array([ed[i] - base[i] for i in ids])
    bs = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return {'n': len(ids), 'base': round(float(np.mean([base[i] for i in ids])) * 100, 2),
            'edited': round(float(np.mean([ed[i] for i in ids])) * 100, 2),
            'diff_pp': round(float(x.mean()) * 100, 2), 'ci95_pp': [round(float(np.percentile(bs, 2.5)) * 100, 2),
                                                                  round(float(np.percentile(bs, 97.5)) * 100, 2)],
            'improved_problems': int((x > 0).sum()), 'worse_problems': int((x < 0).sum())}


c8 = {}
for bench in ('math500', 'gsm8k', 'ifeval', 'amc23', 'minerva', 'olympiad', 'mmlupro1k', 'gpqa'):
    base = per_problem('unedited', bench)
    if not base:
        continue
    c8[bench] = {run: paired(base, per_problem(run, bench)) for run in ('p10c', 'p25c', 'p50c') if per_problem(run, bench)}
merged = {run: J(f'exp8_math/eval/{run}/merged.json') for run in ('unedited', 'p10c', 'p25c', 'p50c')}
c8['reported_merged'] = {run: {k: round(m[k]['acc'] * 100, 1) for k in ('math500', 'gsm8k', 'ifeval')} | {'math500_cut_share': round(m['math500']['cut_share'] * 100, 1)}
                         for run, m in merged.items()}
man = J('exp8_math/runs/p50c/edit/manifest.json')
cfg = J('exp8_math/base_model/Qwen3.5-0.8B-text/config.json')
c8['p50c_edit'] = {'tensors': [t['name'] for t in man['tensors']], 'changed_entries_bf16': sum(t['changed_entries'] for t in man['tensors']),
                   'touched': sum(t['changed_columns'] * t['shape'][0] for t in man['tensors']),
                   'hidden_size': cfg.get('hidden_size') or cfg.get('text_config', {}).get('hidden_size'),
                   'frac_of_0.8B': round(sum(t['changed_columns'] * t['shape'][0] for t in man['tensors']) / 0.8e9 * 100, 4),
                   'relative_change': [round(t['relative_change'], 3) for t in man['tensors']]}
gb = per_problem('unedited', 'math500', 'exp8_math/gemma_followup/eval')
if gb:
    # Gemma 결과는 lenient 채점이 본 수치라 per_problem의 correct(엄격)와 다를 수 있다. 짝 비교는 참고용
    c8['gemma_strict_paired'] = {r: paired(gb, per_problem(r, 'math500', 'exp8_math/gemma_followup/eval'))
                                 for r in ('g05b', 'g10b', 'g25b', 'g50b') if per_problem(r, 'math500', 'exp8_math/gemma_followup/eval')}
c8['run_selection_qwen'] = {
    'note': 'RESULTS_excerpt.md 표의 Qwen 편집 실행(값이 있는 것). 블로그는 v3의 10·25·50 패스 3개만 보고',
    'runs': {'v2_0pass': 'no edit', 'v3_5pass': 'no edit submitted', 'v3_10pass': '12.3', 'v3_25pass': '14.1', 'v3_50pass': '48.1',
             'v4_10pass': 'no edit submitted', 'v4_25pass': '14.1', 'v5_*': '미완(…)'}}
out['C8_exp8'] = c8

json.dump(out, open(os.path.join(HERE, 'verify_logs_results.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1)[:12000])

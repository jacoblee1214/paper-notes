"""실험 3 블록 가중치만으로 K-composition 점수를 계산해 복사(induction) 헤드를 찾을 수 있는지 본다.

입력: data/exp3_language/networks/<언어>/blocks.npz (4층, 헤드 6개, d=384, 행렬은 (출력, 입력) 순서)
      data/exp3_language/eval/heads.json (실행으로 측정한 정답: induct > 0.2 인 헤드)
방법: Elhage et al. (2021)의 K-composition 점수를 잔차 공간에서 계산한다.
      점수(h1 -> h2) = ||W_OV^{h1} W_K^{h2}||_F / (||W_OV^{h1}||_F ||W_K^{h2}||_F)
      LayerNorm의 gamma는 W_K 앞에 곱하고, 평균 제거(centering)는 투영 행렬로 반영한다.
      입력층(임베딩)과 출력층이 없으므로 forward pass는 하지 않으며, 가중치만 읽는다.
      무작위 행렬의 기준선과 비교하고, 헤드별 최대 점수로 정답 복사 헤드를 맞히는지 AUC·상위 5 정밀도로 본다.
실행: .venv/bin/python -I analysis/extra_kcomp.py
→ analysis/extra_kcomp_results.json, report/figures/extra_kcomp_heads.png
"""
import os
import sys
import json
import time
import hashlib
import platform

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
DATA = os.path.join(ROOT, 'data', 'exp3_language')
FIG = os.path.join(ROOT, 'report', 'figures')
os.makedirs(FIG, exist_ok=True)

LANGS = ['python', 'romaji', 'english']
NET = {'python': 'python_L4_H6_d384_s0', 'romaji': 'romaji_L4_H6_d384_s0', 'english': 'english_L4_H6_d384_s0'}
LAYERS, HEADS, D = 4, 6, 384
DH = D // HEADS
RNG = np.random.default_rng(0)

FONT = next((f for f in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                          '/System/Library/Fonts/AppleSDGothicNeo.ttc'] if os.path.exists(f)), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams['axes.unicode_minus'] = False


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_blocks(lang):
    path = os.path.join(DATA, 'networks', lang, 'blocks.npz')
    z = np.load(path, allow_pickle=False)  # 데이터 파일만 읽는다 (pickle 금지)
    blocks = []
    for l in range(LAYERS):
        blocks.append({k: z[f'{k}.{l}'].astype(np.float64) for k in ['ln1_w', 'W_q', 'W_k', 'W_v', 'W_o']})
    return blocks, {'path': os.path.relpath(path, ROOT), 'bytes': os.path.getsize(path), 'sha256': sha256(path)}


def head_mats(blk, h):
    """행 벡터 규약: x @ M. 헤드 h의 K 입력 행렬과 OV 행렬을 돌려준다."""
    s = slice(h * DH, (h + 1) * DH)
    k = blk['W_k'][s, :].T                     # (384, 64)
    v = blk['W_v'][s, :].T                     # (384, 64)
    o = blk['W_o'][:, s].T                     # (64, 384)
    return k, v @ o                            # OV: (384, 384)


P = np.eye(D) - np.ones((D, D)) / D            # LayerNorm 평균 제거


def kcomp(ov1, k2, g2):
    k_eff = P @ (g2[:, None] * k2)
    return np.linalg.norm(ov1 @ k_eff) / (np.linalg.norm(ov1) * np.linalg.norm(k_eff))


def score_tensor(blocks):
    """S[l2, h2, l1, h1] = K-comp 점수. l1 < l2 인 쌍만 값이 있다."""
    S = np.full((LAYERS, HEADS, LAYERS, HEADS), np.nan)
    for l2 in range(LAYERS):
        g2 = blocks[l2]['ln1_w']
        for h2 in range(HEADS):
            k2, _ = head_mats(blocks[l2], h2)
            for l1 in range(l2):
                for h1 in range(HEADS):
                    _, ov1 = head_mats(blocks[l1], h1)
                    S[l2, h2, l1, h1] = kcomp(ov1, k2, g2)
    return S


def random_baseline(n=200):
    vals = []
    for _ in range(n):
        ov = RNG.standard_normal((D, DH)) @ RNG.standard_normal((DH, D))
        k = RNG.standard_normal((D, DH))
        vals.append(kcomp(ov, k, np.ones(D)))
    v = np.array(vals)
    return {'mean': float(v.mean()), 'std': float(v.std()), 'p95': float(np.quantile(v, 0.95)), 'n': n}


def auc(scores, labels):
    """Mann-Whitney U로 AUC. 동점은 평균 순위."""
    scores, labels = np.asarray(scores), np.asarray(labels, bool)
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores))
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    npos, nneg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    t_start = time.time()
    heads_json = json.load(open(os.path.join(DATA, 'eval', 'heads.json')))
    results = {'method': 'Elhage et al. (2021) K-composition, residual-space, LN gain folded and centering projected; weights only, no forward pass',
               'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'platform': platform.platform()},
               'threads': int(os.environ['OMP_NUM_THREADS']), 'seed': 0, 'nets': {}}
    truth, rows, scores_all, labels_all = {}, [], [], []
    for lang in LANGS:
        blocks, meta = load_blocks(lang)
        S = score_tensor(blocks)
        # 정답: 실행으로 측정한 induct 값 (eval/heads.json)
        rec = heads_json[NET[lang]]
        truth[lang] = {f'L{l}H{h}': rec[f'L{l}H{h}']['induct'] for l in range(LAYERS) for h in range(HEADS)}
        best = {}
        for l2 in range(LAYERS):
            for h2 in range(HEADS):
                name = f'L{l2}H{h2}'
                if l2 == 0:
                    best[name] = {'max': None, 'partner': None}   # 앞선 층이 없어 점수를 정의할 수 없다
                    continue
                flat = S[l2, h2, :l2, :].reshape(-1)
                j = int(np.nanargmax(flat))
                l1, h1 = divmod(j, HEADS)
                best[name] = {'max': float(flat[j]), 'partner': f'L{l1}H{h1}'}
                is_copy = truth[lang][name] > 0.2
                rows.append((lang, name, float(flat[j]), is_copy))
                scores_all.append(float(flat[j]))
                labels_all.append(is_copy)
        results['nets'][lang] = {'data': meta, 'arch': NET[lang],
                                 'copy_truth_induct_gt_0.2': [k for k, v in truth[lang].items() if v > 0.2],
                                 'best_kcomp_per_head': best,
                                 'max_kcomp_overall': float(np.nanmax(S))}
    # 평가: 층 1–3 헤드 54개(층 0은 앞선 층이 없어 제외). 정답 복사 헤드 5개
    sc = np.array(scores_all); lb = np.array(labels_all)
    order = np.argsort(-sc)
    top5 = [rows[i] for i in order[:5]]
    ev = {'n_eligible_heads': int(len(sc)), 'n_copy_truth': int(lb.sum()),
          'auc': auc(sc, lb), 'precision_at_5': float(lb[order[:5]].mean()), 'recall_at_5': float(lb[order[:5]].sum() / lb.sum()),
          'top5': [{'net': r[0], 'head': r[1], 'kcomp': r[2], 'copy_truth': bool(r[3])} for r in top5]}
    results['evaluation'] = ev
    # 무작위 기준선: 같은 형태의 가우시안 행렬로 계산한 점수 분포
    results['random_baseline'] = random_baseline()
    results['wall_time_s'] = round(time.time() - T0, 2)
    out_json = os.path.join(HERE, 'extra_kcomp_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 그림: 네트워크별 헤드 최대 K-comp 막대, 정답 복사 헤드는 빨간색
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    base = results['random_baseline']['mean']
    for ax, lang in zip(axes, LANGS):
        names, vals, cols = [], [], []
        for l2 in range(LAYERS):
            for h2 in range(HEADS):
                name = f'L{l2}H{h2}'
                names.append(name)
                b = results['nets'][lang]['best_kcomp_per_head'][name]['max']
                vals.append(0 if b is None else b)
                cols.append('#c0392b' if truth[lang][name] > 0.2 else '#7f8c8d')
        ax.bar(range(len(vals)), vals, color=cols)
        ax.axhline(base, ls='--', color='k', lw=0.8)
        ax.set_xticks(range(0, 24, 3)); ax.set_xticklabels(names[::3], rotation=60, fontsize=7)
        ax.set_title(f'{lang} 네트워크', fontsize=10)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel('앞선 층 헤드와의 최대 K-comp')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'extra_kcomp_heads.png'), dpi=150)
    print(json.dumps({'auc': ev['auc'], 'p5': ev['precision_at_5'], 'random_mean': base, 'top5': ev['top5']}, ensure_ascii=False, indent=1))
    print('wall time', round(time.time() - t_start, 2), 's')


if __name__ == '__main__':
    main()

"""궤적만 보고 성공 여부를 맞히는 분류기로, 궤적 안에 결과 정보가 얼마나 들어 있는지 잰다.

데이터: 공개 LeRobot 형식 로봇 롤아웃 4세트(ankile/square-01a-baseline-uniform-r{1..4}-policy-rollouts, Apache-2.0,
        로봇 패널다, 시뮬레이션 nut-assembly "square" 과제, 세트당 100 에피소드, 프레임 단위 success 플래그).
        data/ankile_square_r{1..4}/data/chunk-000/file-000.parquet (저장소에 넣지 않음).
특징: 관측 상태 9 + 행동 7 + 환경 상태 14 = 30 신호의 에피소드 요약(평균, 표준편차, 최소, 최대, 첫 프레임, 마지막 프레임).
      success, reward, done, is_valid 열은 특징에 쓰지 않는다. 에피소드 길이는 따로 본다(성공 에피소드는 성공 시점에 끝남).
분류기: L2 정규화 로지스틱 회귀(numpy Newton), 표준화는 학습 폴드 기준, 5겹 층화 교차검증을 seed 0–4로 반복.
대조군: 첫 프레임만 쓴 특징(초기 조건, 기대 AUC ~0.5), 레이블 섞기(기대 ~0.5), 길이 한 개.
실행: .venv/bin/python -I analysis/extra_outcome_leak.py
→ analysis/extra_outcome_leak_results.json, report/figures/extra_outcome_leak.png
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
import pyarrow
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
DATA = os.path.join(ROOT, 'data')
FIG = os.path.join(ROOT, 'report', 'figures')
os.makedirs(FIG, exist_ok=True)
SETS = ['r1', 'r2', 'r3', 'r4']
SIGNALS = ['observation.state', 'action', 'observation.environment_state']
LAMBDA = 1.0
N_FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]

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


def column(t, name):
    vals = t[name].to_pylist()
    return np.array(vals, dtype=float).reshape(len(vals), -1)


def load_set(path):
    """에피소드 목록: (성공 여부, 유효 프레임 수, 신호 행렬 T×30)."""
    t = pq.read_table(path)
    ep = column(t, 'episode_index')[:, 0].astype(int)
    succ = column(t, 'success')[:, 0]
    valid = column(t, 'is_valid')[:, 0] > 0
    sig = np.hstack([column(t, c) for c in SIGNALS])
    out = []
    for e in np.unique(ep):
        m = (ep == e) & valid
        s = sig[m]
        y = int(succ[ep == e].max() > 0)
        out.append((y, int(m.sum()), s))
    return out, {'rows': int(t.num_rows), 'episodes': int(len(np.unique(ep))), 'successes': int(sum(o[0] for o in out))}


def summarise(s):
    return np.concatenate([s.mean(0), s.std(0), s.min(0), s.max(0), s[0], s[-1]])


EARLY = [50, 100]   # 모든 에피소드가 최소 119 유효 프레임을 가지므로(성공 최소) 종료 전 구간만 쓴다


def build_features(eps):
    X_traj, X_first, X_last, length, y = [], [], [], [], []
    X_early = {k: [] for k in EARLY}
    for lab, n, s in eps:
        X_traj.append(summarise(s))
        for k in EARLY:
            X_early[k].append(summarise(s[:k]))
        X_first.append(s[0])
        X_last.append(s[-1])
        length.append(n)
        y.append(lab)
    out = {'traj': np.array(X_traj), 'first': np.array(X_first), 'last': np.array(X_last),
           'length': np.array(length, float)[:, None], 'y': np.array(y)}
    for k in EARLY:
        out[f'early{k}'] = np.array(X_early[k])
    return out


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_lr(X, y, lam=LAMBDA, iters=60):
    """L2 로지스틱 회귀. 절편은 벌점 없음. 뉴턴 방법."""
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    R = np.eye(d + 1) * lam
    R[0, 0] = 0.0
    W = np.zeros(d + 1)
    for _ in range(iters):
        p = sigmoid(Xb @ W)
        g = Xb.T @ (p - y) + R @ W
        H = Xb.T @ (Xb * (p * (1 - p))[:, None]) + R + 1e-8 * np.eye(d + 1)
        step = np.linalg.solve(H, g)
        W -= step
        if np.linalg.norm(step) < 1e-8:
            break
    return W


def predict_lr(W, X):
    return W[0] + X @ W[1:]


def auc(scores, labels):
    """Mann-Whitney U 기반 AUC (동점은 평균 순위)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    order = np.argsort(scores, kind='mergesort')
    ss = scores[order]
    ranks = np.empty(len(ss))
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    npos, nneg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def stratified_folds(y, k, rng):
    fold = np.empty(len(y), int)
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def cv_auc(X, y, seed, shuffle_labels=False):
    rng = np.random.default_rng(seed)
    yy = y.copy()
    if shuffle_labels:
        rng.shuffle(yy)
    fold = stratified_folds(yy, N_FOLDS, rng)
    oof = np.zeros(len(yy))
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        W = fit_lr((X[tr] - mu) / sd, yy[tr].astype(float))
        oof[te] = predict_lr(W, (X[te] - mu) / sd)
    return auc(oof, yy)


def repeated(X, y, shuffle_labels=False):
    vals = [cv_auc(X, y, s, shuffle_labels) for s in SEEDS]
    return {'mean': round(float(np.mean(vals)), 4), 'sd': round(float(np.std(vals, ddof=1)), 4), 'per_seed': [round(v, 4) for v in vals]}


def main():
    t_start = time.time()
    files, eps_all, meta = [], [], {}
    for r in SETS:
        p = os.path.join(DATA, f'ankile_square_{r}', 'data', 'chunk-000', 'file-000.parquet')
        eps, m = load_set(p)
        eps_all.append((r, eps))
        files.append({'set': r, 'path': os.path.relpath(p, ROOT), 'bytes': os.path.getsize(p), 'sha256': sha256(p), **m})
    # 세트별로 따로 교차검증하지 않고, 네 세트를 합쳐 에피소드 단위로 본다(세트 번호는 특징에 넣지 않음)
    eps = [e for _, es in eps_all for e in es]
    F = build_features(eps)
    y = F['y']
    traj_no_len = F['traj']
    traj_with_len = np.hstack([F['traj'], F['length']])
    results = {
        'data': {'source': 'https://huggingface.co/datasets/ankile/square-01a-baseline-uniform-r{1,2,3,4}-policy-rollouts (resolve/main)',
                 'license': 'apache-2.0', 'files': files,
                 'episodes_total': int(len(y)), 'successes_total': int(y.sum()),
                 'success_rate': round(float(y.mean()), 4)},
        'settings': {'lambda_l2': LAMBDA, 'folds': N_FOLDS, 'seeds': SEEDS, 'features_traj': int(traj_no_len.shape[1]),
                     'signals': SIGNALS, 'excluded_columns': ['success', 'reward', 'done', 'is_valid']},
        'length_by_outcome': {
            'success_median': float(np.median(F['length'][y == 1])), 'failure_median': float(np.median(F['length'][y == 0])),
            'success_min': float(F['length'][y == 1].min()), 'success_max': float(F['length'][y == 1].max()),
            'failure_min': float(F['length'][y == 0].min()), 'failure_max': float(F['length'][y == 0].max()),
        },
        'auc_length_only': round(auc(-F['length'][:, 0], y.astype(bool)), 4),
        'cv_auc': {
            'trajectory_summary_no_length': repeated(traj_no_len, y),
            'trajectory_summary_with_length': repeated(traj_with_len, y),
            'length_only_logreg': repeated(F['length'], y),
            'first_frame_only_control': repeated(F['first'], y),
            'last_frame_only': repeated(F['last'], y),
            'early50_summary_no_termination': repeated(F['early50'], y),
            'early100_summary_no_termination': repeated(F['early100'], y),
            'shuffled_labels_control': repeated(traj_no_len, y, shuffle_labels=True),
        },
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'pyarrow': pyarrow.__version__,
                     'platform': platform.platform()},
        'threads': 4,
    }
    results['wall_time_s'] = round(time.time() - t_start, 2)
    with open(os.path.join(HERE, 'extra_outcome_leak_results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    rng = np.random.default_rng(0)
    for k, (lab, name) in enumerate([(0, '실패'), (1, '성공')]):
        v = F['length'][y == lab, 0]
        ax.scatter(np.full(len(v), k) + rng.uniform(-0.12, 0.12, len(v)), v, s=14, alpha=0.7,
                   color='#c0392b' if lab == 0 else '#2e86c1', edgecolor='none')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['실패', '성공'])
    ax.set_ylabel('유효 프레임 수 (에피소드 길이)')
    ax.set_title('길이만으로도 결과가 갈린다 (4세트 합쳐 400 에피소드)', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'extra_outcome_leak.png'), dpi=150)
    print(json.dumps({k: results[k] for k in ['data', 'length_by_outcome', 'auc_length_only', 'cv_auc']},
                     ensure_ascii=False, indent=1, default=float))
    print('wall time', round(time.time() - t_start, 2), 's')


if __name__ == '__main__':
    main()

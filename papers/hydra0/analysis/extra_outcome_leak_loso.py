"""궤적 분류기의 결과 누설을 정책 세트 간 일반화로 본다 (leave-one-set-out).

extra_outcome_leak.py는 네 세트를 합쳐 5겹 교차검증을 했다. 그러면 같은 정책의 롤아웃이 학습과 평가에 함께 들어간다.
이 스크립트는 세트 하나를 빼고 나머지 세트로 학습해, 다른 정책의 궤적에 같은 규칙이 통하는지 본다.
데이터와 특징 정의는 extra_outcome_leak.py를 그대로 가져와 쓴다(data/ankile_square_r{1..4}/, 저장소에 넣지 않음).

주의: 세트 r3, r4는 실패 에피소드가 1개뿐이라 그 세트를 평가 세트로 쓰면 AUC가 매우 불안정하다.
      그래서 평가 지표는 세트별 값과 함께 평가 세트의 실패 수를 같이 보고한다.

실행: python3 -I analysis/extra_outcome_leak_loso.py  → analysis/extra_outcome_leak_loso_results.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python3 -I는 스크립트 폴더를 경로에 넣지 않으므로 직접 넣는다
from extra_outcome_leak import (DATA, LAMBDA, SETS, auc, build_features,  # noqa: E402
                                fit_lr, load_set, predict_lr, sha256)

RESULTS = os.path.join(HERE, 'extra_outcome_leak_loso_results.json')
FEATURES = ['traj_no_len', 'early100_no_termination', 'early50_no_termination', 'last_frame_only', 'length_only']


def feature_matrix(F, name):
    if name == 'traj_no_len':
        return F['traj']
    if name == 'early100_no_termination':
        return F['early100']
    if name == 'early50_no_termination':
        return F['early50']
    if name == 'last_frame_only':
        return F['last']
    if name == 'length_only':
        return F['length']
    raise KeyError(name)


def fit_eval(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    W = fit_lr((Xtr - mu) / sd, ytr.astype(float), lam=LAMBDA)
    s = predict_lr(W, (Xte - mu) / sd)
    return auc(s, yte.astype(bool))


def main():
    t0 = time.time()
    per_set = {}
    eps_by_set = {}
    files = []
    for r in SETS:
        p = os.path.join(DATA, f'ankile_square_{r}', 'data', 'chunk-000', 'file-000.parquet')
        eps, m = load_set(p)
        eps_by_set[r] = build_features(eps)
        files.append({'set': r, 'bytes': os.path.getsize(p), 'sha256': sha256(p), **m})

    out = {}
    for feat in FEATURES:
        rows = {}
        for test in SETS:
            tr = [s for s in SETS if s != test]
            Xtr = np.vstack([feature_matrix(eps_by_set[s], feat) for s in tr])
            ytr = np.concatenate([eps_by_set[s]['y'] for s in tr])
            Xte = feature_matrix(eps_by_set[test], feat)
            yte = eps_by_set[test]['y']
            rows[test] = {
                'auc': round(float(fit_eval(Xtr, ytr, Xte, yte)), 4),
                'test_episodes': int(len(yte)),
                'test_failures': int((yte == 0).sum()),
                'train_success_rate': round(float(ytr.mean()), 3),
            }
        aucs = np.array([rows[s]['auc'] for s in SETS])
        big = [rows[s]['auc'] for s in ('r1', 'r2')]
        out[feat] = {'per_heldout_set': rows, 'mean_all4': round(float(aucs.mean()), 4),
                     'mean_r1_r2_only': round(float(np.mean(big)), 4)}
        per_set[feat] = rows

    # 대조: 첫 프레임만(초기 조건), 같은 LOSO 틀에서 0.5 근처여야 한다
    ctrl = {}
    for test in SETS:
        tr = [s for s in SETS if s != test]
        Xtr = np.vstack([eps_by_set[s]['first'] for s in tr])
        ytr = np.concatenate([eps_by_set[s]['y'] for s in tr])
        ctrl[test] = round(float(fit_eval(Xtr, ytr, eps_by_set[test]['first'], eps_by_set[test]['y'])), 4)
    out['first_frame_only_control'] = {'per_heldout_set_auc': ctrl,
                                       'mean_all4': round(float(np.mean(list(ctrl.values()))), 4)}

    res = {
        'data': {'source': 'ankile/square-01a-baseline-uniform-r{1,2,3,4}-policy-rollouts (resolve/main), Apache-2.0',
                 'files': files},
        'settings': {'lambda_l2': LAMBDA, 'split': 'leave-one-set-out (4 sets)',
                     'features_from': 'analysis/extra_outcome_leak.py (동일 정의)',
                     'failure_count_note': 'r3, r4는 실패 1개씩이라 평가 AUC가 불안정함'},
        'results': out,
        'wall_time_s': round(time.time() - t0, 2),
    }
    with open(RESULTS, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    for feat in FEATURES:
        print(feat, out[feat]['mean_all4'], out[feat]['mean_r1_r2_only'],
              {s: out[feat]['per_heldout_set'][s]['auc'] for s in SETS}, flush=True)
    print('control first-frame', out['first_frame_only_control']['mean_all4'], flush=True)


if __name__ == '__main__':
    main()

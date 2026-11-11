"""공개 갤러리 데이터의 물체 추적 높이로 두 성공 규칙을 비교한다.

  (1) 높이 최댓값 규칙: 파지 시점 이후 추적 높이의 최댓값이 파지 시점보다 50 mm 이상이면 성공
  (2) 유지 규칙: 파지 시점 이후 높이가 50 mm 이상인 구간이 3 s 이상 이어지면 성공 (5 cm·3 s)

입력: data/<hand>/<object>/<timestamp>/object_tracking/gotrack_output/world_pose_records.json (추적 높이, 프레임별)
      data/<hand>/<object>/<timestamp>/executed_grasp/meta.json (실행 상태 시각: grasp, lift)
      공개 데이터셋 willi19/autodex-gallery에서 받은 40개 시도(무작위 추출, seed 0). 데이터는 저장소에 넣지 않는다.
좌표 부호: 기록된 z를 그대로 쓰는 규칙("as_recorded")과 부호를 뒤집은 규칙("flipped")을 함께 계산한다.
  기록 z 그대로는 50 mm 상승이 한 번도 나오지 않고, 부호를 뒤집으면 상승 시작이 lift 상태 시각 뒤 0.1–1 s에 맞는다.
  그래서 뒤집은 쪽을 주 해석으로 쓰되, 확정하지 않는다(정답 라벨 없음).
실행: .venv/bin/python -I analysis/extra_gallery_rules.py
→ analysis/extra_gallery_rules_results.json, report/figures/extra_gallery_rules.png
"""
import os
import sys
import json
import glob
import time
import math
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
DATA = os.path.join(ROOT, 'data')
FIG = os.path.join(ROOT, 'report', 'figures')
os.makedirs(FIG, exist_ok=True)
THR_MM = 50.0           # 5 cm = 50 mm
HOLD_S = 3.0
HOLD_SENS = [1.0, 2.0, 3.0, 4.0]
HF_REPO = 'willi19/autodex-gallery'
HF_LAST_MODIFIED = '2026-09-30T16:55:20.000Z'
SIGNS = {'as_recorded': 1.0, 'flipped': -1.0}

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


def runs_above(t, h, thr):
    """h >= thr가 이어지는 구간 (시작 시각, 끝 시각, 기록 끝까지 이어졌는지)."""
    out, start = [], None
    for i in range(len(h)):
        if h[i] >= thr:
            if start is None:
                start = i
        elif start is not None:
            out.append((t[start], t[i - 1], False))
            start = None
    if start is not None:
        out.append((t[start], t[-1], True))
    return out


def hold_status(t, h, thr, hold_s):
    """'yes' (hold_s 이상 유지), 'undetermined' (기록이 끝나 판정 불가), 'no'."""
    undet = False
    for t0, t1, open_end in runs_above(t, h, thr):
        if t1 - t0 >= hold_s:
            return 'yes'
        if open_end:
            undet = True
    return 'undetermined' if undet else 'no'


def load_trial(trial_dir):
    rec = json.load(open(os.path.join(trial_dir, 'object_tracking', 'gotrack_output', 'world_pose_records.json')))
    meta = json.load(open(os.path.join(trial_dir, 'executed_grasp', 'meta.json')))
    frames = [r for r in rec if r.get('status') == 'ok' and r.get('translation_world_mm') is not None]
    t = np.array([r['time_sec'] for r in frames], float)
    z = np.array([r['translation_world_mm'][2] for r in frames], float)
    st = {s['state']: s['t_rel_s'] for s in meta.get('execution_states', [])}
    return t, z, st


def analyse_trial(t, z, st, sign):
    if 'grasp' not in st:
        return None
    tg = st['grasp']
    pre = (t >= tg - 0.5) & (t <= tg)
    if pre.sum() < 3:
        return None
    z_ref = float(np.median(z[pre]))
    post = t >= tg
    tp, hp = t[post], sign * (z[post] - z_ref)
    onset = next((float(tp[i]) for i in range(len(hp)) if hp[i] >= THR_MM), None)
    return {
        'record_end_s': float(t[-1]), 'grasp_t_s': tg, 'lift_t_s': st.get('lift'),
        'zmax_mm': float(hp.max()), 'zmax_rule': bool(hp.max() >= THR_MM),
        'hold_3s': hold_status(tp, hp, THR_MM, HOLD_S),
        'hold_by_duration': {str(s): hold_status(tp, hp, THR_MM, s) for s in HOLD_SENS},
        'onset_s': onset,
        'longest_run_s': max([t1 - t0 for t0, t1, _ in runs_above(tp, hp, THR_MM)] or [0.0]),
    }


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def summarise(rows):
    det = [r for r in rows if r['hold_3s'] != 'undetermined']
    agree = sum(1 for r in det if r['zmax_rule'] == (r['hold_3s'] == 'yes'))
    cnt = {'zmax_yes_hold_yes': 0, 'zmax_no_hold_no': 0, 'zmax_yes_hold_no': 0, 'zmax_no_hold_yes': 0}
    for r in det:
        cnt[f"zmax_{'yes' if r['zmax_rule'] else 'no'}_hold_{'yes' if r['hold_3s'] == 'yes' else 'no'}"] += 1
    sens = {}
    for s in HOLD_SENS:
        key = str(s)
        dd = [r for r in rows if r['hold_by_duration'][key] != 'undetermined']
        a = sum(1 for r in dd if r['zmax_rule'] == (r['hold_by_duration'][key] == 'yes'))
        sens[key] = {'n_determined': len(dd), 'agree': a, 'agree_rate': round(a / len(dd), 4) if dd else None}
    return {
        'n_analysed': len(rows), 'n_zmax_yes': sum(1 for r in rows if r['zmax_rule']),
        'n_hold_yes': sum(1 for r in rows if r['hold_3s'] == 'yes'),
        'n_hold_undetermined': sum(1 for r in rows if r['hold_3s'] == 'undetermined'),
        'n_determined': len(det),
        'agreement': {'agree': agree, 'rate': round(agree / len(det), 4) if det else None,
                      'wilson95': wilson(agree, len(det))},
        'contingency_determined': cnt,
        'zmax_yes_but_hold_no_trials': [r['trial'] for r in det if r['zmax_rule'] and r['hold_3s'] == 'no'],
        'sensitivity_hold_duration': sens,
    }


def main():
    t_start = time.time()
    paths = sorted(glob.glob(os.path.join(DATA, '*', '*', '*', 'object_tracking', 'gotrack_output', 'world_pose_records.json')))
    trials, files, skipped = [], [], []
    for p in paths:
        d = os.path.dirname(os.path.dirname(os.path.dirname(p)))
        trials.append(os.path.relpath(d, DATA))
        files.append(os.path.relpath(p, ROOT))
        t, z, st = load_trial(d)
        trials[-1] = (trials[-1], t, z, st)
    rows = {k: [] for k in SIGNS}
    onsets = {k: [] for k in SIGNS}
    per_trial = {k: [] for k in SIGNS}
    for name, t, z, st in trials:
        for k, s in SIGNS.items():
            r = analyse_trial(t, z, st, s)
            if r is None:
                if k == 'as_recorded':
                    skipped.append(name)
                continue
            r['trial'] = name
            rows[k].append(r)
            if r['onset_s'] is not None and r['lift_t_s'] is not None:
                onsets[k].append(r['onset_s'] - r['lift_t_s'])
    alignment = {}
    for k in SIGNS:
        a = np.array(onsets[k]) if onsets[k] else np.array([])
        alignment[k] = {'n_with_onset': int(len(a)), 'median_onset_minus_lift_s': float(np.median(a)) if len(a) else None,
                        'min_s': float(a.min()) if len(a) else None, 'max_s': float(a.max()) if len(a) else None}
    res = {
        'data': {'source': f'https://huggingface.co/datasets/{HF_REPO} (resolve/main)', 'repo_last_modified': HF_LAST_MODIFIED,
                 'license': '데이터셋 카드에 license 필드 없음(미표기)',
                 'selection': 'world_pose_records 639개 중 seed 0로 40개 무작위 추출',
                 'n_files': len(files), 'bytes_records': int(sum(os.path.getsize(os.path.join(ROOT, f)) for f in files)),
                 'sha256_first_file': {files[0]: sha256(os.path.join(ROOT, files[0]))}},
        'threshold_mm': THR_MM, 'hold_s': HOLD_S,
        'skipped_no_grasp_state': skipped,
        'conventions': {k: summarise(rows[k]) for k in SIGNS},
        'sign_check_onset_minus_lift': alignment,
        'record_end_minus_lift_s': {'median': float(np.median([r['record_end_s'] - r['lift_t_s'] for r in rows['flipped'] if r['lift_t_s'] is not None])),
                                    'min': float(min(r['record_end_s'] - r['lift_t_s'] for r in rows['flipped'] if r['lift_t_s'] is not None)),
                                    'max': float(max(r['record_end_s'] - r['lift_t_s'] for r in rows['flipped'] if r['lift_t_s'] is not None))},
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
                     'platform': platform.platform()},
        'threads': 4, 'seed': 0,
    }
    res['wall_time_s'] = round(time.time() - t_start, 2)
    with open(os.path.join(HERE, 'extra_gallery_rules_results.json'), 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, k in zip(axes, SIGNS):
        rs = rows[k]
        x = np.array([r['zmax_mm'] for r in rs])
        y = np.array([r['longest_run_s'] for r in rs])
        col = []
        for r in rs:
            if r['hold_3s'] == 'undetermined':
                col.append('#7f8c8d')
            elif r['hold_3s'] == 'yes':
                col.append('#2e86c1')
            else:
                col.append('#c0392b' if r['zmax_rule'] else '#bdc3c7')
        ax.scatter(x, y, c=col, s=28, edgecolor='k', linewidth=0.3)
        ax.axvline(THR_MM, ls='--', color='k', lw=0.8)
        ax.axhline(HOLD_S, ls='--', color='k', lw=0.8)
        ax.set_xlabel('파지 이후 높이 최댓값 (mm)')
        ax.set_title('기록 z 그대로' if k == 'as_recorded' else '부호 뒤집음(추정)', fontsize=10)
    axes[0].set_ylabel('50 mm 이상 연속 구간 길이 (s)')
    fig.text(0.5, -0.02, '파랑: 두 규칙 모두 성공, 빨강: 최댓값만 성공, 회색: 기록 끝으로 판정 불가, 연회색: 둘 다 실패',
             ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'extra_gallery_rules.png'), dpi=150, bbox_inches='tight')
    print(json.dumps({'conventions': {k: {kk: res['conventions'][k][kk] for kk in
                                          ['n_analysed', 'n_zmax_yes', 'n_hold_yes', 'n_hold_undetermined', 'n_determined',
                                           'agreement', 'contingency_determined']} for k in SIGNS},
                      'alignment': alignment, 'record_end_minus_lift_s': res['record_end_minus_lift_s']},
                     ensure_ascii=False, indent=1, default=float))
    print('wall time', round(time.time() - t_start, 2), 's')


if __name__ == '__main__':
    main()

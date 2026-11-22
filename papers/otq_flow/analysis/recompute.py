"""논문 수치 재계산: Table 1 평균, Table 2 비율·Wilson·Fisher, Fig. 5 평균·검정, Fig. 2/3 곡선 주장 검증.
실행: python3 recompute.py   (numpy, scipy 필요; extract_curves.py가 만든 curves_extracted.json을 읽는다)
결과: recompute_results.json"""
import json, math, itertools
import numpy as np
from scipy import stats

out = {}

def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]

# ---------- V1: Table 1 평균 ----------
envs = ['antmaze-large', 'antmaze-giant', 'humanoidmaze-medium', 'humanoidmaze-large', 'cube-double', 'cube-triple', 'scene', 'puzzle-3x3']
T1 = {
 'FAWAC': [0.17, 0, 0.24, 0, 0.02, 0, 0.38, 0.03],
 'DSRL': [0.61, 0.03, 0.53, 0.03, 0.74, 0.01, 0.99, 0.87],
 'QSM': [0.9, 0.24, 0.82, 0.06, 0.33, 0.06, 0.78, 0.57],
 'CGQL': [0.76, 0, 0.6, 0.05, 0.38, 0.08, 0.38, 0.48],
 'FEdit': [0.58, 0.02, 0.22, 0.03, 0.4, 0.02, 0.62, 0.99],
 'QAM': [0.81, 0.18, 0.67, 0.11, 0.64, 0.03, 0.97, 1.0],
 'IFQL': [0.36, 0.01, 0.86, 0.24, 0.11, 0, 0.84, 1.0],
 'FQL': [0.76, 0, 0.68, 0.09, 0.46, 0.03, 0.78, 0.7],
 'QAM-F': [0.83, 0.12, 0.65, 0.12, 0.65, 0.03, 0.95, 0.99],
 'OTQL': [0.88, 0.19, 0.82, 0.19, 0.68, 0.04, 0.99, 0.99],
}
T1_reported_avg = {'FAWAC': 0.11, 'DSRL': 0.48, 'QSM': 0.47, 'CGQL': 0.34, 'FEdit': 0.36, 'QAM': 0.55, 'IFQL': 0.43, 'FQL': 0.44, 'QAM-F': 0.54, 'OTQL': 0.59}
v1 = {}
for k, v in T1.items():
    m = float(np.mean(v))
    v1[k] = {'recomputed': round(m, 4), 'reported': T1_reported_avg[k], 'diff': round(m - T1_reported_avg[k], 4)}
out['V1_table1_avg'] = v1
# 환경별 최고 방법
best = {}
for i, e in enumerate(envs):
    mx = max(T1[k][i] for k in T1)
    best[e] = {'max': mx, 'winners': [k for k in T1 if abs(T1[k][i] - mx) < 1e-9]}
out['V1b_best_per_env'] = best
out['V1c_otql_wins'] = sum('OTQL' in b['winners'] for b in best.values())
# 8개 환경 중 OTQL 단독/공동 1위 수, 대 QAM-F, FQL 차
out['V1d_margin_vs_fast'] = {'QAM-F': round(np.mean(T1['OTQL']) - np.mean(T1['QAM-F']), 4), 'FQL': round(np.mean(T1['OTQL']) - np.mean(T1['FQL']), 4)}
# OTQL이 QAM-F를 이긴 환경 수(동률 제외), 부호 검정(환경 8개, 시드 정보 없음)
d = np.array(T1['OTQL']) - np.array(T1['QAM-F'])
w = int((d > 0).sum()); l = int((d < 0).sum()); t = int((d == 0).sum())
out['V1e_sign_vs_QAMF'] = {'wins': w, 'losses': l, 'ties': t, 'binom_p_two_sided': round(stats.binomtest(w, w + l, 0.5).pvalue, 4) if w + l else None}
# 환경 수준 bootstrap CI of mean diff (환경 8개 재표집; 시드 불확실성은 반영 못함)
rng = np.random.default_rng(0)
bs = [d[rng.integers(0, 8, 8)].mean() for _ in range(20000)]
out['V1f_bootstrap_envs_diff_vs_QAMF'] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

# ---------- V2: Table 2 ----------
t2 = {'tape in penholder': (14, 24, 30), 'move knight': (9, 22, 30)}
v2 = {}
for k, (a, b, n) in t2.items():
    p = stats.fisher_exact([[b, n - b], [a, n - a]])
    v2[k] = {'sft': f'{a}/{n}={a/n:.3f}', 'otql': f'{b}/{n}={b/n:.3f}', 'wilson_sft': wilson(a, n), 'wilson_otql': wilson(b, n),
             'fisher_p_two_sided': round(p[1], 4), 'diff_pp': round((b - a) / n * 100, 2)}
A = sum(v[0] for v in t2.values()); B = sum(v[1] for v in t2.values()); N = sum(v[2] for v in t2.values())
p = stats.fisher_exact([[B, N - B], [A, N - A]])
v2['pooled'] = {'sft': f'{A}/{N}={A/N:.4f}', 'otql': f'{B}/{N}={B/N:.4f}', 'wilson_sft': wilson(A, N), 'wilson_otql': wilson(B, N),
                'fisher_p_two_sided': round(p[1], 4), 'diff_pp': round((B - A) / N * 100, 2),
                'text_claims': {'abstract': '38% -> 76%', 'sec4.3': 'increase by an average of 37%'}}
# 쌍 연결 불가(같은 초기 상태 불명)이므로 독립 두 표본 가정. 반올림 규칙 점검
v2['rounding'] = {'sft_pooled_pct': round(A / N * 100, 2), 'otql_pooled_pct': round(B / N * 100, 2), 'paper_floor_or_round': 'SFT 38.33->38 (둘 다 동일), OTQL 76.67->76 (반올림이면 77, 버림이면 76)'}
out['V2_table2'] = v2

# ---------- V3: Fig. 5 ----------
c = json.load(open('curves_extracted.json'))
f5 = c['fig5']
fin = {k: v['otql'][-1] for k, v in f5.items()}
base = {k: v['base'] for k, v in f5.items()}
v3 = {'final_points': fin, 'base': base,
      'mean_final': round(float(np.mean(list(fin.values()))), 4), 'mean_base': round(float(np.mean(list(base.values()))), 4),
      'paper_claim': '36% -> 86%'}
# 마지막 3점 평균, 전 구간 평균(첫 점 제외)
v3['mean_last3_each_task'] = {k: round(float(np.mean(v['otql'][-3:])), 3) for k, v in f5.items()}
v3['mean_last3_all'] = round(float(np.mean([x for v in f5.values() for x in v['otql'][-3:]])), 4)
v3['mean_all_points_excl_first'] = round(float(np.mean([x for v in f5.values() for x in v['otql'][1:]])), 4)
v3['mean_first_point'] = round(float(np.mean([v['otql'][0] for v in f5.values()])), 4)
# 정수 가능성: 점마다 10 에피소드 가정 -> 값은 0.1 배수
v3['all_values_multiples_of_0.1'] = all(abs(x * 10 - round(x * 10)) < 1e-6 for v in f5.values() for x in v['otql'] + [v['base']])
# 가정: 기준 정책도 10 에피소드 평가, 최종 점도 10 에피소드
kf = int(round(sum(fin.values()) * 10)); nf = 30
kb = int(round(sum(base.values()) * 10)); nb = 30
pp = stats.fisher_exact([[kf, nf - kf], [kb, nb - kb]])
v3['pooled_assumed_n10_each'] = {'otql_final': f'{kf}/{nf}', 'base': f'{kb}/{nb}', 'wilson_otql': wilson(kf, nf), 'wilson_base': wilson(kb, nb), 'fisher_p_two_sided': round(pp[1], 5)}
per = {}
for k in f5:
    a = int(round(fin[k] * 10)); b = int(round(base[k] * 10))
    per[k] = {'otql_final': f'{a}/10', 'base': f'{b}/10', 'fisher_p': round(stats.fisher_exact([[a, 10 - a], [b, 10 - b]])[1], 4), 'wilson_otql': wilson(a, 10), 'wilson_base': wilson(b, 10)}
v3['per_task_assumed_n10'] = per
# 같은 점이 훈련 데이터이기도 한 점: 마지막 점은 40 에피소드 시점 체크포인트의 롤아웃
v3['paper_rounding'] = {'otql_86.67': 'paper 86 (버림)', 'base_36.67': 'paper 36 (버림)'}
out['V3_fig5'] = v3

# ---------- V4: Fig. 3 주장 ----------
f3 = c['fig3']
v4 = {}
for ev in ['OpenMicrowave-v0', 'ArrangeBoxes-v0']:
    d = f3[ev]
    v4[ev] = {m: {'max': round(max(x['values']), 3), 'final': x['values'][-1], 'mean': round(float(np.mean(x['values'])), 3)} for m, x in d.items()}
mx_diff = [v4[e]['OTQL']['max'] - v4[e]['FQL']['max'] for e in v4]
fin_diff = [v4[e]['OTQL']['final'] - v4[e]['FQL']['final'] for e in v4]
mean_diff = [v4[e]['OTQL']['mean'] - v4[e]['FQL']['mean'] for e in v4]
v4['OTQL_minus_FQL'] = {'max_each': [round(x, 3) for x in mx_diff], 'max_avg_pp': round(float(np.mean(mx_diff)) * 100, 1),
                        'final_each': [round(x, 3) for x in fin_diff], 'final_avg_pp': round(float(np.mean(fin_diff)) * 100, 1),
                        'curve_mean_each': [round(x, 3) for x in mean_diff], 'curve_mean_avg_pp': round(float(np.mean(mean_diff)) * 100, 1),
                        'relative_max_each': [round(v4[e]['OTQL']['max'] / v4[e]['FQL']['max'] - 1, 3) for e in v4],
                        'paper_claim': '평균 15% 높음'}
v4['OTQL_vs_QC_max'] = {e: [v4[e]['OTQL']['max'], v4[e]['QC']['max']] for e in ['OpenMicrowave-v0', 'ArrangeBoxes-v0']}
v4['n_eval_points'] = len(f3['OpenMicrowave-v0']['OTQL']['values'])
out['V4_fig3'] = v4

# ---------- V5: Fig. 2 주장 ----------
f2 = c['fig2']
v5 = {}
for tk, d in f2.items():
    v5[tk] = {m: {'max_offline_phase(first10pts)': round(max(x['values'][:10]), 3), 'max_all': round(max(x['values']), 3), 'final': x['values'][-1],
                  'max_online_phase': round(max(x['values'][10:]), 3), 'n_points': x['n_points']} for m, x in d.items()}
out['V5_fig2'] = v5

# ---------- V6: 예산/세부 숫자 ----------
out['V6_budget'] = {
 'abstract': '50-60 episodes', 'sec4.3': '10 demos + 10 initial rollouts + 10/iter until buffer = 50 episodes',
 'D.3_single_task': '10 expert demos + 50 total rollouts (60 episodes)',
 'D.3_vla': '10 demos + 30 rollouts (40 episodes)',
 'train_steps_single_task_each_iter': 20000, 'iterations_if_50_rollouts': 'initial 10 + 4x10 -> 5 fits, or +5x10 -> 6 fits (Fig 5 has 6 points)',
 'fig5_points': 6}
# NFE 감소율: 10 -> 3 (70%), 10 -> 2 (80%)
out['V7_nfe'] = {'10_to_3': 1 - 3 / 10, '10_to_2': 1 - 2 / 10, 'paper_claim_vla': '70%', 'ok': abs((1 - 3 / 10) - 0.7) < 1e-9}
# Wilson sanity 예: 24/30
out['V8_wilson_check'] = {'24/30': wilson(24, 30), 'scipy_binomtest_ci': [round(x, 4) for x in stats.binomtest(24, 30).proportion_ci(method='wilson')]}

json.dump(out, open('recompute_results.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))

"""LoopedWAM (arXiv:2611.25595) 보고 수치 재계산. 표준 라이브러리만 사용, CPU 수 초 내 실행.

입력: 논문 본문·부록 표에서 옮겨 적은 수치(출처 표 번호를 주석으로 표시).
출력: recompute_results.json (검증 V1–Vn 항목별 재계산 값과 판정).
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = {}


def wilson(p, n, z=1.96):
    """Wilson 95% 구간 (p: 비율 0-1, n: 시행 수)."""
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def fisher_two_sided(a, n1, b, n2):
    """2x2 Fisher 정확검정 양측 p-value. a successes of n1, b successes of n2."""
    N = n1 + n2
    K = a + b

    def pmf(x):
        return math.comb(K, x) * math.comb(N - K, n1 - x) / math.comb(N, n1)

    p_obs = pmf(a)
    lo, hi = max(0, K - n2), min(K, n1)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-9)))


def two_prop_z(p1, n1, p2, n2):
    """비짝지음 두 비율 차이 z (독립 가정, 참고용)."""
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (p1 - p2) / se, se


def normal_p(z):
    return math.erfc(abs(z) / math.sqrt(2))


# ------------------------------------------------------------------ V1: LIBERO 표 1 평균 재계산
# 표 1 (suite별: Spatial, Object, Goal, Long, 보고 평균)
LIBERO = {
    'LoopedWAM': ([96.9, 99.5, 97.9, 90.1], 96.1),
    'LoopedWAM-E (Ka=4,N=2)': ([97.2, 99.3, 97.5, 88.5], 95.6),
    'LoopedWAM-E (Ka=2,N=2)': ([97.8, 99.1, 97.8, 89.2], 96.0),
    'Fast-WAM* (re-eval)': ([97.5, 99.3, 96.3, 93.5], 96.6),
    'Faster-WAM* (re-eval)': ([99.0, 99.6, 98.2, 98.1], 98.7),
    'Light-WAM* (re-eval)': ([96.7, 98.9, 96.6, 90.9], 95.8),
}
v1 = {}
for name, (suites, reported) in LIBERO.items():
    mean = sum(suites) / 4
    v1[name] = dict(recomputed=round(mean, 3), reported=reported, match=abs(round(mean, 1) - reported) < 0.051)  # 96.65 -> 96.6 은 반올림(짝수 쪽) 차이로 허용
RESULTS['V1_libero_table1_avg'] = v1

# ------------------------------------------------------------------ V2: LIBERO 신뢰구간과 차이 검정
# 한 모델당 40 task x 50 초기상태 x 3 seed = 6000 에피소드. suite당 1500, 평균은 suite 동일 가중.
# 평균의 표준오차 = sqrt(sum p_s(1-p_s)/1500) / 4 (suite 간 독립 가정, 근사)
def libero_avg_se(suites):
    ps = [s / 100 for s in suites]
    return math.sqrt(sum(p * (1 - p) / 1500 for p in ps)) / 4 * 100  # 퍼센트 포인트


v2 = {}
for name, (suites, reported) in LIBERO.items():
    se = libero_avg_se(suites)
    v2[name] = dict(avg=reported, half_width_95=round(1.96 * se, 2))
base = LIBERO['LoopedWAM'][1]
diffs = {}
for other in ('Fast-WAM* (re-eval)', 'Faster-WAM* (re-eval)', 'Light-WAM* (re-eval)'):
    d = base - LIBERO[other][1]
    se = math.sqrt(libero_avg_se(LIBERO['LoopedWAM'][0]) ** 2 + libero_avg_se(LIBERO[other][0]) ** 2)
    z = d / se
    diffs[other] = dict(diff_pts=round(d, 2), se_pts=round(se, 3), z=round(z, 2),
                        p_two_sided_unpaired=round(normal_p(z), 4))
v2['differences_vs_LoopedWAM'] = diffs
RESULTS['V2_libero_uncertainty'] = v2

# ------------------------------------------------------------------ V3: 표 5 (9개 설정) 평균·이득·지연 회귀
# (Ka, N, calls, latency_ms, Spatial, Object, Goal, Long, Avg, Gain)
T5 = [
    (1, 2, 24, 24.7, 96.87, 99.07, 96.60, 88.20, 95.18, 0.85),
    (2, 2, 36, 27.6, 97.80, 99.13, 97.80, 89.20, 95.98, 1.45),
    (4, 2, 60, 33.2, 97.20, 99.27, 97.53, 88.53, 95.63, 0.67),
    (1, 5, 60, 33.3, 97.40, 98.80, 95.80, 88.93, 95.23, 0.70),
    (2, 5, 90, 39.1, 97.67, 98.93, 96.87, 88.73, 95.55, 0.70),
    (1, 10, 120, 47.9, 97.27, 99.27, 96.07, 88.73, 95.33, 1.65),
    (4, 5, 150, 54.5, 97.67, 99.33, 97.07, 90.33, 96.10, 0.65),
    (2, 10, 180, 59.9, 97.53, 99.00, 96.60, 89.53, 95.67, 1.83),
    (4, 10, 300, 86.2, 97.60, 98.87, 97.73, 90.67, 96.22, 1.35),
]
# 표 B1 (짧은 스케줄 학생, 같은 설정): (Ka, N, lat, avg)
B1 = {(1, 2): 94.33, (2, 2): 94.53, (4, 2): 94.97, (1, 5): 94.53, (2, 5): 94.85,
      (1, 10): 93.68, (4, 5): 95.45, (2, 10): 93.83, (4, 10): 94.87}
v3 = {}
avgs_ok = []
for ka, n, calls, lat, s, o, g, l, avg, gain in T5:
    re_avg = (s + o + g + l) / 4
    avgs_ok.append(abs(round(re_avg, 2) - avg) <= 0.011)
    gain_re = avg - B1[(ka, n)]
    v3[f'Ka{ka}_N{n}'] = dict(avg_recomputed=round(re_avg, 3), avg_reported=avg,
                              gain_recomputed=round(gain_re, 3), gain_reported=gain)
gains = [t[9] for t in T5]
v3['summary'] = dict(all_avgs_match=all(avgs_ok), mean_gain_reported=round(sum(gains) / 9, 3),
                     gain_min=min(gains), gain_max=max(gains),
                     mean_of_nine_avgs=round(sum(t[8] for t in T5) / 9, 3))
# 지연 선형 회귀 (지연 ~ 호출 수 c = N(6+6Ka)) -- 본문 Fig.3c "19.8 + 0.223 c ms, R2 = 0.999"
xs = [t[2] for t in T5]
ys = [t[3] for t in T5]
mx, my = sum(xs) / 9, sum(ys) / 9
slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
icpt = my - slope * mx
ss_res = sum((y - (icpt + slope * x)) ** 2 for x, y in zip(xs, ys))
ss_tot = sum((y - my) ** 2 for y in ys)
v3['latency_fit'] = dict(intercept_ms=round(icpt, 2), slope_ms_per_call=round(slope, 4),
                         r2=round(1 - ss_res / ss_tot, 4))
# 한 샘플링 스텝 추가 비용 (본문: 2.90, 4.04, 6.63 ms at Ka=1,2,4)
v3['per_step_cost_ms'] = dict(
    Ka1=round((T5[5][3] - T5[0][3]) / 8, 2), Ka2=round((T5[7][3] - T5[1][3]) / 8, 2),
    Ka4=round((T5[8][3] - T5[2][3]) / 8, 2))
RESULTS['V3_table5_elastic'] = v3

# ------------------------------------------------------------------ V4: 표 4 속도·메모리 비율
T4 = {'Fast-WAM* 10 steps': (107.5, 15.25), 'Faster-WAM* 10': (72.6, 12.76),
      'Light-WAM*': (21.8, 4.38), 'Dense-30 10': (90.5, 3.79), 'Dense-12 10': (40.8, 2.09),
      'LoopedWAM 4/10': (91.8, 2.05), 'LoopedWAM 4/2': (35.0, 2.09),
      'LoopedWAM-E (4,2)': (33.2, 2.78), 'LoopedWAM-E (2,2)': (27.6, 2.68),
      'LoopedWAM-E (1,2)': (24.7, 2.58), 'Fast-WAM* 2 steps': (37.1, 15.32)}
base_lat, base_mem = T4['Fast-WAM* 10 steps']
v4 = {}
for k, (lat, mem) in T4.items():
    v4[k] = dict(speedup=round(base_lat / lat, 2), mem_ratio_vs_fastwam=round(base_mem / mem, 2))
v4['claim_7.4x_memory'] = round(15.25 / 2.05, 2)
v4['claim_1.8x_memory_vs_dense30'] = round(3.79 / 2.05, 2)
v4['claim_1.2x_full_depth_vs_fastwam'] = round(107.5 / 91.8, 2)
v4['claim_1.3x_at_matched_2steps_loopedwam_E_vs_fastwam_2steps'] = round(37.1 / 27.6, 2)
v4['claim_5.7x_less_memory_at_2steps'] = round(15.32 / 2.68, 2)
v4['saving_ms_2steps_vs_10steps_at_Ka4'] = round(86.2 - 33.2, 1)
v4['saving_ms_Ka4_to_Ka2_at_N2'] = round(33.2 - 27.6, 1)
v4['param_ratio_dense30_over_loopedwam'] = round(1.42 / 0.5845, 2)
RESULTS['V4_table4_efficiency'] = v4

# ------------------------------------------------------------------ V5: 파라미터 합산 검산
RESULTS['V5_params'] = dict(video_M=479.2, action_M=105.3, sum_M=round(479.2 + 105.3, 1),
                            reported_total_M=584.5, match=abs(479.2 + 105.3 - 584.5) < 0.05)

# ------------------------------------------------------------------ V6: RoboTwin 2.0 표 E2 (50 task)
E2 = [  # (task, clean, rand, avg)
    ('adjust_bottle', 100, 99, 99.5), ('beat_block_hammer', 100, 96, 98.0), ('blocks_ranking_rgb', 99, 99, 99.0),
    ('blocks_ranking_size', 98, 96, 97.0), ('click_alarmclock', 99, 100, 99.5), ('click_bell', 100, 100, 100.0),
    ('dump_bin_bigbin', 100, 98, 99.0), ('grab_roller', 100, 100, 100.0), ('handover_block', 94, 95, 94.5),
    ('handover_mic', 100, 100, 100.0), ('hanging_mug', 57, 45, 51.0), ('lift_pot', 100, 100, 100.0),
    ('move_can_pot', 93, 99, 96.0), ('move_pillbottle_pad', 100, 100, 100.0),
    ('move_playingcard_away', 100, 100, 100.0), ('move_stapler_pad', 93, 85, 89.0), ('open_laptop', 92, 93, 92.5),
    ('open_microwave', 36, 50, 43.0), ('pick_diverse_bottles', 86, 85, 85.5), ('pick_dual_bottles', 100, 96, 98.0),
    ('place_a2b_left', 97, 94, 95.5), ('place_a2b_right', 96, 92, 94.0), ('place_bread_basket', 93, 94, 93.5),
    ('place_bread_skillet', 89, 87, 88.0), ('place_burger_fries', 97, 98, 97.5),
    ('place_can_basket', 61, 57, 59.0), ('place_cans_plasticbox', 99, 96, 97.5),
    ('place_container_plate', 96, 97, 96.5), ('place_dual_shoes', 91, 89, 90.0), ('place_empty_cup', 100, 100, 100.0),
    ('place_fan', 96, 93, 94.5), ('place_mouse_pad', 96, 95, 95.5), ('place_object_basket', 86, 82, 84.0),
    ('place_object_scale', 98, 97, 97.5), ('place_object_stand', 92, 97, 94.5), ('place_phone_stand', 99, 99, 99.0),
    ('place_shoe', 96, 98, 97.0), ('press_stapler', 97, 98, 97.5), ('put_bottles_dustbin', 90, 89, 89.5),
    ('put_object_cabinet', 91, 87, 89.0), ('rotate_qrcode', 82, 80, 81.0), ('scan_object', 85, 88, 86.5),
    ('shake_bottle', 100, 100, 100.0), ('shake_bottle_horizontally', 100, 100, 100.0),
    ('stack_blocks_three', 97, 96, 96.5), ('stack_blocks_two', 100, 100, 100.0),
    ('stack_bowls_three', 79, 83, 81.0), ('stack_bowls_two', 95, 96, 95.5), ('stamp_seal', 98, 96, 97.0),
    ('turn_switch', 60, 70, 65.0),
]
n_tasks = len(E2)
clean_mean = sum(t[1] for t in E2) / n_tasks
rand_mean = sum(t[2] for t in E2) / n_tasks
avg_mean = sum(t[3] for t in E2) / n_tasks
rows_consistent = all(abs((t[1] + t[2]) / 2 - t[3]) < 1e-9 for t in E2)
# 시도 수 기준(100 x 50 task) 가중 평균: 같은 값
pooled_clean = sum(t[1] for t in E2) / (100 * n_tasks) * 100
pooled_rand = sum(t[2] for t in E2) / (100 * n_tasks) * 100
n_eps = 100 * n_tasks * 2
ci_lo, ci_hi = wilson((pooled_clean + pooled_rand) / 200, n_eps)
RESULTS['V6_robotwin_table_E2'] = dict(
    n_tasks=n_tasks, clean_mean=round(clean_mean, 2), rand_mean=round(rand_mean, 2), avg_mean=round(avg_mean, 2),
    reported=dict(clean=92.1, rand=91.7, avg=91.9), per_task_avg_consistent=rows_consistent,
    episodes_total=n_eps, wilson95_avg_pct=[round(100 * ci_lo, 2), round(100 * ci_hi, 2)],
)
# 차이 검정: LoopedWAM 91.9 vs Fast-WAM 91.8 (1만 에피소드, 비짝지음 근사)
z, se = two_prop_z(0.919, n_eps, 0.918, n_eps)
RESULTS['V6_robotwin_table_E2']['vs_fastwam_diff'] = dict(diff=0.1, z=round(z, 2), p=round(normal_p(z), 3))

# ------------------------------------------------------------------ V7: RoboCasa-GR1 표 E3 (24 task)
E3 = {
    'PnP+close': [(58.0, 62.0, 58.0), (68.0, 76.0, 70.0), (44.0, 34.0, 48.0), (48.0, 50.0, 46.0),
                  (44.0, 42.0, 36.0), (54.0, 54.0, 48.0)],
    'Cutting board': [(66.0, 58.0, 52.0), (38.0, 58.0, 44.0), (68.0, 66.0, 80.0), (56.0, 62.0, 54.0),
                      (48.0, 62.0, 36.0)],
    'Placemat': [(40.0, 48.0, 56.0), (58.0, 68.0, 52.0), (64.0, 42.0, 50.0), (24.0, 26.0, 30.0)],
    'Tray': [(54.0, 40.0, 56.0), (54.0, 62.0, 58.0), (56.0, 54.0, 56.0), (52.0, 44.0, 50.0), (28.0, 30.0, 32.0)],
    'Plate': [(58.0, 66.0, 60.0), (44.0, 42.0, 44.0), (48.0, 52.0, 42.0), (80.0, 60.0, 80.0)],
}
allt = [x for g in E3.values() for x in g]
v7 = {}
for idx, name in enumerate(('LoopedWAM', 'LoopedWAM-E (4,2)', 'LoopedWAM-E (2,2)')):
    vals = [x[idx] for x in allt]
    v7[name] = dict(task_mean=round(sum(vals) / len(vals), 2), n_tasks=len(vals))
v7['reported_avg'] = dict(LoopedWAM=52.2, E42=52.4, E22=51.6)
# 24 rollouts x 50 = 1200 episodes per model; Wilson 95% half-width at p=0.522
lo, hi = wilson(0.522, 1200)
v7['wilson95_LoopedWAM_pct'] = [round(100 * lo, 2), round(100 * hi, 2)]
v7['half_width_pts'] = round(100 * (hi - lo) / 2, 2)
# 50-rollout 기준 비교 (StarVLA 47.8, DiT4DiT 50.8는 같은 50 rollout 프로토콜)
for name, other in (('StarVLA-GR00T', 47.8), ('DiT4DiT', 50.8)):
    z, se = two_prop_z(0.522, 1200, other / 100, 1200)
    v7[f'vs_{name}'] = dict(diff=round(52.2 - other, 1), z=round(z, 2), p_two_sided=round(normal_p(z), 3))
RESULTS['V7_robocasa'] = v7

# ------------------------------------------------------------------ V8: 표 E1 그룹 가중 평균
grp_tasks = dict(pnp=6, cut=5, placemat=4, tray=5, plate=4)
e1_lw = dict(pnp=52.7, cut=55.2, placemat=46.5, tray=48.8, plate=57.5)
w = sum(e1_lw[k] * grp_tasks[k] for k in grp_tasks) / sum(grp_tasks.values())
RESULTS['V8_robocasa_group_weighting'] = dict(loopedwam_task_weighted=round(w, 3), reported=52.2)

# ------------------------------------------------------------------ V9: 표 D1 (RoboTwin-10) 열 평균
D1 = {  # per task: Dense12, 1/4, 2/2, 4/1, 4/2, 4/4, Dense30
    'adjust_bottle': [99.5, 99.5, 99.5, 99.0, 100.0, 99.5, 99.5],
    'grab_roller': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.5],
    'stack_blocks_two': [98.0, 97.5, 98.0, 99.0, 97.0, 97.5, 100.0],
    'press_stapler': [90.5, 88.5, 91.0, 84.5, 85.0, 92.5, 91.5],
    'stamp_seal': [87.0, 83.0, 90.0, 89.0, 91.5, 89.5, 93.5],
    'scan_object': [79.5, 83.0, 87.0, 83.0, 85.5, 84.5, 91.0],
    'move_stapler_pad': [61.0, 63.5, 63.5, 75.5, 69.5, 72.5, 80.0],
    'turn_switch': [59.5, 61.5, 62.0, 64.0, 67.0, 74.5, 66.0],
    'hanging_mug': [44.5, 45.5, 40.0, 45.0, 51.0, 58.5, 59.5],
    'open_microwave': [53.0, 73.5, 58.0, 70.5, 77.5, 45.0, 55.5],
}
cfg_names = ['Dense-12', '1/4', '2/2', '4/1', '4/2', '4/4', 'Dense-30']
reported_row = [77.25, 79.55, 78.90, 80.95, 82.40, 81.40, 83.60]
d1 = {}
for j, name in enumerate(cfg_names):
    col = [v[j] for v in D1.values()]
    d1[name] = dict(recomputed=round(sum(col) / 10, 2), reported=reported_row[j])
d1_ok = all(abs(d1[n]['recomputed'] - d1[n]['reported']) < 0.006 for n in cfg_names)
RESULTS['V9_robotwin10_table_D1'] = dict(columns=d1, all_match=d1_ok)
# 4 contact-rich tasks average (본문: Dense-12 54.50, 2/2 55.88, 1/4 61.00, Dense-30 65.25 등)
# 본문 값은 4개 접촉 과제(위 표에서 해당 과제 식별 불가) -> 재계산 생략, 전체 평균만 확인

# 본문: 4/1, 4/2, 4/4 세 모델의 RT-10 평균 차이 (1/4 대비 1.4 등) 재계산
RESULTS['V9_robotwin10_table_D1']['diff_4_1_vs_1_4'] = round(80.95 - 79.55, 2)

# ------------------------------------------------------------------ V10: Ablation 차이 (표 7) 통계적 크기
# LIBERO 짧은 스케줄, 모델당 6000 에피소드, 비짝지음 근사
def diff_test(a, b, n=6000):
    z, se = two_prop_z(a / 100, n, b / 100, n)
    return dict(diff=round(a - b, 2), se_pts=round(100 * se, 3), z=round(z, 2),
                p_two_sided=round(normal_p(z), 5))


RESULTS['V10_ablation_tests_LIBERO_unpaired'] = {
    'video_loops_4/1_vs_action_loops_1/4 (본문 2.3)': diff_test(94.0, 91.7),
    'late_alignment_KVmix_vs_deepest (본문 +0.45)': diff_test(94.45, 94.00),
    'random_core_vs_default (본문 -12.7)': diff_test(82.2, 94.9),
    'elastic_specialist_12_vs_elastic_student (본문 -0.6, CI -1.3..+0.1)': diff_test(93.77, 94.33, n=6000),
    'distill_teacher_traj_vs_elastic (본문 -0.21, 표 8)': diff_test(95.44, 95.65),
}
RESULTS['V10_note'] = ('표 8의 차이(-0.15~-0.33)는 같은 에피소드 짝지음 검정이라 보고되지 않은 짝지음 SE가 필요함. '
                       '비짝지음 근사로는 판정 불가. 표 5 이득의 유의성(9개 중 7개)도 에피소드 단위 데이터가 없어 재현 불가.')

# ------------------------------------------------------------------ V11: 실로봇 (표 9, 80 trial)
robot = {
    'Fast-WAM*': (57, 80), 'LoopedWAM': (57, 80), 'LoopedWAM-E (4,2)': (55, 80), 'LoopedWAM-E (2,2)': (54, 80)}
per_task = {
    'Fast-WAM*': [14, 18, 14, 11], 'LoopedWAM': [14, 18, 14, 11],
    'LoopedWAM-E (4,2)': [14, 17, 14, 10], 'LoopedWAM-E (2,2)': [13, 17, 14, 10]}
v11 = {}
for k, (s, n) in robot.items():
    lo, hi = wilson(s / n, n)
    v11[k] = dict(successes=s, trials=n, pct=round(100 * s / n, 2), wilson95=[round(100 * lo, 2), round(100 * hi, 2)],
                  per_task_sum_ok=sum(per_task[k]) == s)
v11['fisher_p_LoopedWAM_vs_FastWAM'] = round(fisher_two_sided(57, 80, 57, 80), 4)
v11['fisher_p_LoopedWAM_vs_E42'] = round(fisher_two_sided(57, 80, 55, 80), 4)
v11['fisher_p_LoopedWAM_vs_E22'] = round(fisher_two_sided(57, 80, 54, 80), 4)
v11['failures_LoopedWAM'] = dict(total=80 - 57, contact=10 + 4 + 5, non_contact=2 + 2)
v11['memory_ratio_fast_over_loopedwam'] = round(12.7 / 3.7, 2)
v11['latency_loopedwam_minus_fastwam_ms'] = 160 - 150
RESULTS['V11_real_robot'] = v11

# ------------------------------------------------------------------ 저장
out = HERE / 'recompute_results.json'
out.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2))
print(f'wrote {out}')
print(json.dumps({k: RESULTS[k] for k in ('V1_libero_table1_avg', 'V5_params')}, ensure_ascii=False, indent=1)[:1500])

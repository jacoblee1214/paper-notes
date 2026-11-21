"""공개 데이터 파일의 구조를 정적으로 읽어 논문 §3.2, §4.3의 주장을 확인한다 (표준 라이브러리만 사용).

입력: 원 저장소의 state_action_data_0.csv (재배포 금지 파일이므로 이 폴더에 복사하지 않는다).
      기본 경로: ../repo/upstream_rwm_lite/assets/data/state_action_data_0.csv
      다른 위치는 인자로 준다: python3 analysis/data_checks.py <csv 경로>
열 배치 (upstream anymal_d_flat.py 와 anymal_d_flat_cfg.py 기준): 상태 45 = 기저 속도 3+3, 중력 3,
관절 각도 9:21, 관절 속도 21:33, 관절 토크 33:45; 명령 12 = 45:57; 접촉 8 = 57:65; 종료 1 = 65.
출력: analysis/data_checks_results.json
"""
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT = ROOT / "repo" / "upstream_rwm_lite" / "assets" / "data" / "state_action_data_0.csv"
path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
if not path.exists():
    raise SystemExit(f"not found: {path} (upstream 저장소를 먼저 얕게 복제할 것, README 참조)")

rows = [[float(v) for v in r] for r in csv.reader(path.open(encoding="utf-8")) if r]
n, ncol = len(rows), len(rows[0])
res = {"rows": n, "cols": ncol, "expected_cols": 45 + 12 + 8 + 1}

reset = [i for i, r in enumerate(rows) if all(r[k] == 0 for k in range(21, 33)) and all(r[k] == 0 for k in range(45, 57))]
res["reset_like_rows_0based"] = reset
res["episode_len_from_resets"] = sorted({b - a for a, b in zip(reset, reset[1:])})
res["termination_col_nonzero"] = sum(1 for r in rows if r[ncol - 1] != 0)
res["contact_col_nonzero_rows"] = [sum(1 for r in rows if r[k] != 0) for k in range(57, 65)]
res["gravity_norm_min_max"] = [round(min(math.sqrt(r[6] ** 2 + r[7] ** 2 + r[8] ** 2) for r in rows), 6),
                               round(max(math.sqrt(r[6] ** 2 + r[7] ** 2 + r[8] ** 2) for r in rows), 6)]

# 명령 선행 시차: 명령 a[t]이 관절 각도 q[t+k]와 가장 잘 맞는 k (12개 관절 평균 상관)
def corr(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")

lags = {}
for k in range(0, 9):
    cs = []
    for j in range(12):
        cmd = [rows[t][45 + j] for t in range(0, n - k)]
        pos = [rows[t + k][9 + j] for t in range(0, n - k)]
        # 정지 구간(반복 리셋 행)은 제외
        keep = [i for i in range(len(cmd)) if (i % 1000) > 1 and (i % 1000) < 998]
        cs.append(corr([cmd[i] for i in keep], [pos[i] for i in keep]))
    lags[k] = round(sum(cs) / len(cs), 4)
res["mean_corr_cmd_t_vs_angle_t_plus_k"] = lags
res["best_lag_steps"] = max(lags, key=lambda k: lags[k])

# 창 수 (코드의 valid_indices 규칙 재현: 어떤 종료 표시도 없으므로 전 창이 유효)
H, F = 32, 8
res["windows_total"] = n - H - F + 1
# 리셋 행이 창 안쪽(첫 행을 제외한 위치)에 들어가는 창 수 = 에피소드 경계를 가로지르는 창
W = H + F
crossing = 0
rs = set(reset)
for i in range(n - W + 1):
    if any((i + p) in rs for p in range(1, W)):
        crossing += 1
res["windows_crossing_an_episode_boundary"] = crossing

out = HERE / "data_checks_results.json"
out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))

"""ART-Glove 논문 수치 재계산 (numpy 불필요, 표준 라이브러리만 사용, 1스레드, 수 초).

확인하는 것
- 2048 taxel 배분 산술(§5.1, 부록 B, Fig. 15)과 7개 FPCB
- 22 관절 = 엔코더 PCB 22개 배치 (부록 C)
- Table 2 ROM 폭, Fig. 9 오차 지표의 비율(RMSE/MAE, MaxAE/RMSE)
- 7.4:1 전달비와 AS5600 분해능(외부 데이터시트 값, 논문 미기재) 비교
- 판독 시간 예산(120 Hz, 행 수 가정)
- 저장소 BOM의 참고 가격 합계
실행: python3 analysis/table_check.py  -> analysis/table_check_results.json
"""
import json
import math
import os
import re
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

ROOT = Path(__file__).resolve().parents[1]
RES = {}

# 1) 2048 taxel 배분 (부록 B)
finger_region = 32 * 32                      # 4 손가락 32x32
thumb_meta_palm = 2 * (17 * 16)              # 엄지 metacarpal + 손바닥 각 17x16
thumb_distal_prox = 15 * 32                  # 나머지 15행 (열을 두 면이 나눔)
total = finger_region + thumb_meta_palm + thumb_distal_prox
RES["taxel"] = {"fingers": finger_region, "thumb_meta_palm": thumb_meta_palm,
                "thumb_distal_prox": thumb_distal_prox, "total": total,
                "paper_total": 64 * 32, "match": total == 64 * 32}
# Fig. 15 라벨: 손가락 열 9+9+9+5 = 32, 행 16+6+10 = 32 -> 손가락 1024
fig15_fingers = (3 * (16 + 6 + 10) * 9) + (16 + 6 + 10) * 5
fig15_thumb = 20 * 15 + 12 * 15
RES["taxel"]["fig15_fingers"] = fig15_fingers
RES["taxel"]["fig15_thumb"] = fig15_thumb
RES["taxel"]["fig15_thumb_note"] = "Fig.15 라벨은 20x15, 12x15 (32행), 본문은 15행 영역이라 함 -> 면적 480은 같고 모양 표기가 다름"

# 2) 22 관절 = 엔코더 PCB 배치 (부록 C) and DoF count (§4.2.2)
pcb = {"finger_DIP": 4, "finger_PIP": 4, "type3_arc_slot": 5, "finger_MCP_AA": 4, "others": 5}
RES["encoder_pcb"] = {"layouts": pcb, "sum": sum(pcb.values()),
                      "dof_fingers": 4 * 4, "dof_thumb": 6, "dof_total": 4 * 4 + 6}
RES["encoder_pcb"]["match_22"] = sum(pcb.values()) == 22 == 4 * 4 + 6

# 3) Table 2 ROM 폭 (deg)
rom = {"DIP": (0, 70), "PIP": (0, 75), "MCP-FE": (0, 88), "MCP-AA": (-90, 90),
       "thumb IP": (-65, 70), "thumb MCP-FE": (-30, 90), "thumb MCP-AA": (-30, 30),
       "thumb MCP-AR": (0, 30), "thumb CMC-FE": (-30, 90), "thumb CMC-AR": (0, 25)}
RES["rom_width_deg"] = {k: hi - lo for k, (lo, hi) in rom.items()}

# 4) Fig. 9 관절 정확도 (본문 수치를 그대로 옮김)
acc = {"Type I shaft-sleeve": {"MAE": 0.535, "RMSE": 0.708, "MaxAE": 2.383, "range_deg": 90},
       "Type II shaft-bearing": {"MAE": 0.387, "RMSE": 0.461, "MaxAE": 1.262, "range_deg": 90},
       "Type III arc-slot": {"MAE": 0.377, "RMSE": 0.462, "MaxAE": 1.368, "range_deg": 75}}
for k, v in acc.items():
    v["RMSE_over_MAE"] = round(v["RMSE"] / v["MAE"], 3)
    v["MaxAE_over_RMSE"] = round(v["MaxAE"] / v["RMSE"], 3)
    v["MAE_pct_of_range"] = round(100 * v["MAE"] / v["range_deg"], 2)
gauss_ratio = math.sqrt(math.pi / 2)  # 0 평균 정규분포에서 E|e| 대비 RMS
RES["joint_accuracy"] = {"per_mechanism": acc, "gaussian_RMS_over_MAE": round(gauss_ratio, 3),
                         "note": "RMSE/MAE가 1.253보다 크면 꼬리가 정규분포보다 두껍거나 편향이 있음. 원자료(샘플)가 없어 유의성 검정은 불가"}
RES["joint_accuracy"]["MAE_diff_I_vs_II"] = round(acc["Type I shaft-sleeve"]["MAE"] - acc["Type II shaft-bearing"]["MAE"], 3)
RES["joint_accuracy"]["MAE_diff_II_vs_III"] = round(acc["Type II shaft-bearing"]["MAE"] - acc["Type III arc-slot"]["MAE"], 3)

# 5) 전달비와 분해능: 엔코더 분해능 = 360 / 2^bits. AS5600은 12-bit (외부 데이터시트, 논문 미기재)
bits = 12
step_enc = 360 / 2 ** bits
ratio = 7.4
step_joint_III = step_enc / ratio
RES["resolution"] = {"AS5600_bits_assumed": bits, "encoder_step_deg": round(step_enc, 4),
                     "type3_transmission": ratio, "type3_joint_equiv_step_deg": round(step_joint_III, 4),
                     "type2_step_deg": round(step_enc, 4),
                     "note": "분해능 바닥은 MAE(0.38~0.54도)보다 1자릿수 작음 -> 측정 오차는 분해능이 아니라 기계 조립, 기어 유격, 기준 서보가 지배할 가능성이 큼(추정)"}

# 6) 판독 시간 예산: 2048 taxel, 120 Hz, 12-bit 가정
taxels, hz = 2048, 120
samples_per_s = taxels * hz
bits_per_s = samples_per_s * 12
RES["readout_budget"] = {"samples_per_s": samples_per_s, "Mbit_per_s_12bit": round(bits_per_s / 1e6, 3),
                         "frame_period_ms": round(1000 / hz, 3)}
for rows in (32, 64):
    per_row_us = 1e6 / hz / rows
    RES["readout_budget"][f"per_row_us_if_{rows}_row_scans"] = round(per_row_us, 1)
RES["readout_budget"]["note"] = "행 수와 주사 방식은 논문에 없음. 행 단위 주사 가정으로 행당 허용 시간만 계산"

# 7) 저장소 BOM 참고 가격 합계 (PCB는 '추후 공개'라 제외)
bom = (ROOT / "repo" / "hardware" / "human_centric" / "doc" / "bom.md")
price_total, priced_rows, unpriced = 0.0, 0, []
printer = 0.0
if bom.exists():
    for line in bom.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("| Item") or set(line) <= set("|- "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        name, qty, price = cells[0], cells[2], cells[3]
        if name.startswith("Readout PCB") or "PCB" in name or "FPCB" in name:
            unpriced.append(name)
            continue
        m = re.match(r"\$([\d,]+\.\d\d)", price)
        if "released" in name.lower() or "released" in qty.lower() or "released" in price.lower() or "PCB" in name or "FPCB" in name:
            unpriced.append(name)
            continue
        try:
            q = int(qty)
        except ValueError:
            continue
        if m:
            p = float(m.group(1).replace(",", "")) * q
            if "printer" in name:
                printer = p
            price_total += p
            priced_rows += 1
        else:
            unpriced.append(name)
RES["bom"] = {"priced_rows": priced_rows, "total_usd_all_priced": round(price_total, 2),
              "printer_usd": round(printer, 2),
              "total_usd_excl_printer": round(price_total - printer, 2),
              "unpriced_or_pending": unpriced,
              "note": "'or' 두 가격 중 첫 번째 값 사용. 세금, 배송, 3D 프린터 외주 비용 제외"}

out = ROOT / "analysis" / "table_check_results.json"
out.write_text(json.dumps(RES, indent=2, ensure_ascii=False))
print(json.dumps(RES, indent=2, ensure_ascii=False)[:4000])

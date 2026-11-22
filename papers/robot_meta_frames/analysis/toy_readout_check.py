"""토이 V1: MetaWAM 식 (2)–(3)의 읽기·쓰기 왕복과 슬롯 용량 검사 (numpy, 수 초).

목적: 논문의 "분석적 역변환으로 신호를 되찾는다 (decoder-free)"와 "채널 대역 할당"이 식 수준에서 맞는지 확인한다.
이 토이는 VDM(Cosmos-Predict2)을 쓰지 않는다. 디노이저 오차는 가우시안 잡음으로 흉내 낸다.

입력: 없음 (합성 신호). 출력: analysis/toy_readout_results.json, 표준출력.
실행: python3 analysis/toy_readout_check.py

구성 (코드 cosmos_policy/models/policy_text2world_model.py의 슬롯 배치를 따른다)
  - 잠재 채널 C = 16, 공간 격자 H = W = 28 (224 px 입력, 8배 압축 가정. 논문 본문에는 격자 크기가 없다)
  - ch0 과제 1, ch1–3 현재 고유감각 (9 원소), ch4–11 행동 청크 (8채널 × W 슬롯), ch12–14 미래 고유감각 (9 원소), ch15 가치 (1 계수)
  - 모든 계수는 DCT 행 k_H = 0 에만 쓴다 (논문 §4.2 "kH = 0 row")
"""
import json
from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn

OUT = Path(__file__).resolve().parent / "toy_readout_results.json"
C, H, W = 16, 28, 28
rng = np.random.default_rng(0)
RES = {"C": C, "H": H, "W": W, "assumption": "224 px 입력, 공간 압축 8배 → H = W = 28 (논문 본문에 없음)"}


def pack(task, cur, act, fut, val):
    """행 0 계수 배열 (C, W)를 만든다. 나머지 행은 0."""
    coeff = np.zeros((C, W))
    coeff[0, 0] = task
    coeff[1:4, :3] = cur.reshape(3, 3)               # 9 원소를 3채널 × 처음 3 계수에 (나머지 0)
    coeff[4:12, :] = act.reshape(8, W)
    coeff[12:15, :3] = fut.reshape(3, 3)
    coeff[15, 0] = val
    return coeff


def unpack(coeff):
    task = coeff[0, 0]
    cur = coeff[1:4, :3].reshape(-1)[:9]
    act = coeff[4:12, :].reshape(-1)
    fut = coeff[12:15, :3].reshape(-1)[:9]
    val = coeff[15, 0]
    return task, cur, act, fut, val


def to_spatial(coeff_row0):
    full = np.zeros((C, H, W))
    full[:, 0, :] = coeff_row0
    return idctn(full, axes=(1, 2), norm="ortho")


def to_coeff_row0(z):
    return dctn(z, axes=(1, 2), norm="ortho")[:, 0, :]


# 1) 왕복: 잡음 없는 경우 정확히 복원되는가
task = 1.0
cur = rng.normal(size=9)
act = rng.normal(size=32 * 7)               # RoboCasa: 청크 32 × 관절 7
fut = rng.normal(size=9)
val = rng.normal()
coeff = pack(task, cur, act, fut, val)
z = to_spatial(coeff)
back = to_coeff_row0(z)
t2, c2, a2, f2, v2 = unpack(back)
err = max(abs(t2 - task), np.max(np.abs(c2 - cur)), np.max(np.abs(a2 - act)),
          np.max(np.abs(f2 - fut)), abs(v2 - val))
RES["roundtrip_max_abs_err"] = float(err)

# 2) 행 0 전용 패킹이면 공간 잠재는 H 방향으로 상수인가 (1차원 패턴인가)
RES["spatial_constant_along_H_max_dev"] = float(np.max(np.abs(z - z[:, :1, :])))

# 3) 용량: RoboCasa 청크 32 × 7 = 224 = 8 × W (행동 대역이 꽉 찬다). LIBERO/실로봇 청크 16은 절반.
RES["action_capacity_slots"] = 8 * W
RES["action_used_robocasa"] = 32 * 7
RES["action_used_libero_or_real"] = 16 * 7
RES["future_proprio_capacity_slots"] = 3 * W
RES["current_proprio_capacity_slots"] = 3 * W

# 4) 잡음 주입: 디노이저 오차 e (iid N(0, s^2))가 있을 때 읽기 오차
s = 0.1
trials = 200
errs = []
ident_err = []
for _ in range(trials):
    e = rng.normal(scale=s, size=(C, H, W))
    read = to_coeff_row0(z + e)               # 행 0 계수
    d = read - coeff                          # 행 0 기준 차이
    errs.append(d.ravel())
    # 식 확인: 행 0 계수 = sqrt(H) * DCT_W(H 평균)
    lhs = read
    rhs = np.sqrt(H) * dctn((z + e).mean(axis=1), axes=(1,), norm="ortho")
    ident_err.append(float(np.max(np.abs(lhs - rhs))))
errs = np.concatenate(errs)
RES["noise_sigma"] = s
RES["readout_noise_std_empirical"] = float(np.std(errs))
RES["readout_noise_std_expected_equal_sigma"] = s
RES["identity_readout_eq_sqrtH_DCT_of_Hmean_max_err"] = float(max(ident_err))

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
if __name__ == "__main__":
    for k, v in RES.items():
        print(k, ":", v)

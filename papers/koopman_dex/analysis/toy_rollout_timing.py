"""토이 T1: K-UBM 한 스텝 롤아웃(z <- K z)의 CPU 시간 측정과 닫힌 형태 재계산.
논문 표 1(0.42 ms)과 그림 5(0.017 ms)의 값이 이 연산 크기에서 어느 범위인지 본다.
실행: python3 analysis/toy_rollout_timing.py  (단일 프로세스, 수 초)"""
import json, time
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
rng = np.random.default_rng(0)
out = {}
for name, dz in (("dexart_dz412", 412), ("adroit_goal_dz542", 542), ("real_dz407", 407)):
    K = rng.standard_normal((dz, dz)).astype(np.float32) / np.sqrt(dz)
    z = rng.standard_normal(dz).astype(np.float32)
    lin = torch.nn.Linear(dz, dz, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(K))
    zt = torch.from_numpy(z)
    n = 20000
    # (a) numpy matvec
    t = time.perf_counter()
    x = z
    for _ in range(n):
        x = K @ x
        x /= max(1.0, float(np.abs(x).max()))
    a = (time.perf_counter() - t) / n * 1e3
    # (b) torch Linear + numpy 변환 (공식 추론 스크립트의 스텝과 같은 호출 구성)
    t = time.perf_counter()
    x = zt
    with torch.no_grad():
        for _ in range(n):
            x = lin(x)
            _ = x.cpu().detach().numpy()
            x = x / torch.clamp(x.abs().max(), min=1.0)
    b = (time.perf_counter() - t) / n * 1e3
    out[name] = {"dz": dz, "numpy_matvec_ms": round(a, 4), "torch_linear_plus_numpy_ms": round(b, 4),
                 "flops_per_step": 2 * dz * dz}
out["paper_reported_ms"] = {"table1_sim_bucket": 0.42, "fig5_real": 0.017}
out["note"] = "측정은 이 기기의 한 스레드 CPU. 값은 기기와 호출 방식에 따라 달라지므로 크기 비교용이다."
Path(__file__).with_name("toy_rollout_timing_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))

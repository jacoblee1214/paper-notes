"""NEXT 추론 시간 점검: 논문 A.1의 '단일 스레드 1.76 ms (i9-10920X, 568 Hz)' 주장을 이 맥의 CPU에서 재본다.
모델 구조는 repo의 config/train.yaml(LSTM 2층, hidden 128, 입력 18 = 관절 6 x [q, qdot, dq], H=50, 머리 MLP 256, 출력 6)과 같다.
가중치는 무작위 초기화 (시간 측정만 목적). torch가 필요하다. 실행 약 30초.
출력: analysis/cpu_timing_results.json
"""
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

OUT = Path(__file__).with_name("cpu_timing_results.json")


class Next(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(18, 128, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 6))

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.head(y[:, -1])


def bench(threads, n=2000, warm=200):
    torch.set_num_threads(threads)
    m = Next().eval()
    x = torch.randn(1, 50, 18)
    ts = []
    with torch.no_grad():
        for _ in range(warm):
            m(x)
        for _ in range(n):
            t0 = time.perf_counter()
            m(x)
            ts.append((time.perf_counter() - t0) * 1e3)
    return {"threads": threads, "median_ms": round(statistics.median(ts), 3),
            "p95_ms": round(sorted(ts)[int(0.95 * n)], 3), "hz_from_median": round(1000 / statistics.median(ts), 1)}


N_THREADS = torch.get_num_threads()
n_params = sum(p.numel() for p in Next().parameters())
res = {"machine": platform.machine(), "torch": torch.__version__,
       "n_params": n_params, "single_thread": bench(1), "multi_thread": bench(N_THREADS),
       "paper_claim": {"single_thread_ms": 1.76, "hz": 568, "cpu": "Intel i9-10920X (A.1)"}}
OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps(res, indent=1, ensure_ascii=False))

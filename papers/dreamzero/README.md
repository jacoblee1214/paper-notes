# DreamZero (arXiv:2602.15922, CoRL 2026 구두) 분석 노트

*World Action Models are Zero-shot Policies* (DreamZero, NVIDIA)를 논문 표 재계산, 신뢰구간과 검정, 공식 코드의 정적 대조, 소규모 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다. 논문 PDF에 저작권 표기(© 2026 NVIDIA)가 있어 CC BY 4.0과 충돌한다고 보았습니다.

## 구현 가능성 (요약)

- **14B 공식 추론(7 Hz)과 학습: 이 맥에서 불가능합니다.** 가중치 BF16이 24 GiB 메모리보다 크고, 청크당 추론이 수십 초에서 수 분으로 추정됩니다.
- **축소 재구현(2–4 M 파라미터, MPS 또는 CPU): 조건부로 가능합니다.** 설계 요소 검증에 적합합니다.
- **논문 150 ms는 저정밀(FP8 또는 FP4) 경로를 가정해야 산술이 맞습니다.** 공개 README의 기본 경로는 GB200 0.6 s입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, 코드·가중치 표 |
| `analysis/03_analysis.md` | 표 재계산, 신뢰구간과 검정, 정합성 점검, 지연·메모리 산술, 토이 결과 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 차이, 주장 판정, 구현 가능성 |
| `analysis/table_check.py` | 표 재계산과 검정 (scipy, 약 1초) |
| `analysis/latency_arithmetic.py` | FLOPs, 메모리, 필요 처리량 산술 (표준 라이브러리만) |
| `analysis/toy_dit_cache.py` | DiT 캐시 단계 수와 끝점 오차 토이 (numpy, 약 1초) |
| `analysis/toy_decoupled_noise.py` | 영상과 행동 잡음 스케줄 토이 (numpy, 단일 스레드 약 4분) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/latency_arithmetic.py
python3 analysis/toy_dit_cache.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 analysis/toy_decoupled_noise.py
python3 analysis/plot_figures.py
```

공식 코드는 실행하지 않았습니다. 정적 읽기만 했습니다. 공식 저장소는 `git clone --depth 1 https://github.com/dreamzero0/dreamzero.git`로 받을 수 있습니다(분석 기준 HEAD `ab790c1`).

## 출처와 라이선스

- 논문은 arXiv:2602.15922 (CC BY 4.0, arXiv abs 페이지 확인, 2026-11-15)입니다. 원문 PDF의 저작권 표기 때문에 논문 그림은 싣지 않았습니다.
- 공식 코드는 Apache-2.0입니다. 체크포인트 카드에 따르면 DreamZero-DROID 가중치는 cc-by-nc-4.0이고 DreamZero-AgiBot 가중치는 apache-2.0입니다. 가중치는 받지 않았습니다.
- 표 수치는 인용과 재계산을 위해 옮겼습니다. 그림 수치는 막대 라벨을 읽은 값입니다.
- 그래프와 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다. 토이의 파라미터는 설계 가정이며 논문 값에 맞추지 않았습니다.

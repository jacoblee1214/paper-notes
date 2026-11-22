# Robot Meta-Frames (arXiv:2611.14365, CoRL 2026 포스터) 분석 노트

*From Video Frames to Robot Meta-Frames: A Unified Latent Interface for World Action Models* (MetaWAM, arXiv v1, 2026-11-10)을 표 재계산, 통계 검정, 식 수준 토이, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2611.14365에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **학습과 시뮬 평가 전체는 이 맥에서 불가입니다.** GPU 4장과 CUDA 커널 전제입니다.
- **공식 체크포인트 추론은 조건부입니다.** CUDA 전용 의존성의 MPS 포팅이 필요합니다.
- **재계산과 식 수준 토이는 완료했습니다.** `analysis/recompute.py`, `analysis/toy_readout_check.py` (각 수 초, numpy).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 점검, 토이, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/toy_readout_check.py` | 토이: 읽기·쓰기 왕복과 슬롯 용량 (numpy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_readout_check.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다. 원문 PDF의 저작권 표시는 찾지 못했습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 기준선 대조에는 arXiv:2603.22078 표 4를 썼습니다.
- 공식 코드는 `github.com/showlab/MetaWAM` (Apache-2.0)에 있습니다. 이 저장소에는 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

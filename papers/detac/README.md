# DeTac: Dual-branch Decoupling of Texture and Force Signals in Vision-Based Tactile Sensors (arXiv:2610.16463, CoRL 2026) 분석 노트

*DeTac* (arXiv v1, 2026-10-12)을 표 재계산, 주장 검증, 구조 속도 측정으로 분석한 자료입니다. 공식 코드와 프로젝트 페이지를 찾지 못했습니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 쌍 데이터는 센서 제작과 마커 제작 전후 수집이 필요하고, 힘·표면·직물 실험은 로봇과 힘센서가 필요합니다.
- **구조 자체는 돌릴 수 있는 규모입니다.** 가정 구조 23.77 M 파라미터, 300 epoch 학습이 MPS 128×128에서 약 5.6 h, 256×256에서 약 20.5 h(추정, 해상도는 가정)입니다.
- **합성 데이터 축소판은 조건부로 가능합니다**(노력 1–2주 추정).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 방법, 데이터, 실험 수치, 주장과 증거 사이의 점검 |
| `analysis/02_prior_work.md` | 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산 V1–V13, 표 A1 검정, 맥 속도 측정 |
| `analysis/04_claim_verification.md` | 주장 C1–C12, 식 읽기, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 1초 미만) |
| `analysis/toy_unet_bench.py` | 부록 A.1 구조의 파라미터 수와 한 스텝 시간 (torch, 랜덤 가중치, 약 1분) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
pip install torch                       # 속도 측정에만 필요
python3 analysis/toy_unet_bench.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-23). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다. 속도 측정은 랜덤 가중치 구조이며 학습을 하지 않았습니다.

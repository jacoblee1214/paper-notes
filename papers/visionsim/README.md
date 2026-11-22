# VisionSIM (arXiv:2611.08368, CoRL 2026 스포트라이트) 분석 노트

*VisionSIM: Beyond Idealized Sensors — A Multi-Modal Simulator for Robot Perception* (arXiv v1, 2026-11-05)을 수치 재계산, 작은 시뮬레이션 두 개, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.08368에서 받을 수 있습니다). 논문 그림은 한 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 전체 재현은 어렵습니다.** 이유는 GPU가 아니라 코드(능동 SPAD dToF 모델이 공개 main에 없음), 데이터(100장면 자산과 체크섬 미공개), 하드웨어(ADS6311 등 실기)입니다.
- **공개된 수동 SPAD, 이벤트, RGB, IMU, iToF, 정적 열 모델은 CPU로 돌 수 있어 보입니다.** 설치와 코드 실행이 필요하고 이 분석에서는 하지 않았습니다.
- **핵심 물리의 작은 시뮬레이션은 가능하고 완료했습니다.** `analysis/toy_dtof_pileup.py`와 `analysis/toy_itof_failures.py`(각 1초 미만, numpy).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 한 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 보고 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V14, 통계 정보량, 토이 T1·T2, 주장 판정 C1–C20 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D19, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 본문과 부록 수치 재계산 (numpy, 수 초) |
| `analysis/toy_dtof_pileup.py` | 토이 T1: 단일 광자 LiDAR 파일업 (numpy) |
| `analysis/toy_itof_failures.py` | 토이 T2: iToF 실패 모드 (numpy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 한 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_dtof_pileup.py
python3 analysis/toy_itof_failures.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시는 찾지 못했습니다.
- 논문 그림은 그림 3 한 장만 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2611.08368"). 나머지 그림은 제3자 3D 자산, 데이터셋 렌더 프레임, 실제 장비 촬영을 포함해 뺐습니다.
- 공식 코드는 `github.com/WISION-Lab/visionsim`에 있고 GPL-3.0-or-later입니다(상업용 별도 라이선스 문구 포함). 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

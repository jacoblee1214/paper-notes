# AnyTeleop (arXiv:2307.04577, RSS 2023) 분석 노트

*AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System* (arXiv v3, 2024-05-16)을 표 재계산, 시드 단위 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2307.04577에서 받을 수 있습니다). 논문 그림은 도식 두 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **전체 시스템 재현은 이 맥에서 불가입니다.** 서버 본체가 공개되지 않았고 CuRobo 동작 생성은 CUDA를 전제합니다. 실로봇과 GPU 학습이 필요합니다.
- **리타게팅과 웹 뷰어는 CPU로 가능성이 있습니다.** 설치와 실행은 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 도식 두 장과 우리가 그린 그래프) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 시드 단위 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/crop_figures.py` | 논문 도식 두 장을 원문 PDF에서 자르기 (pymupdf, Pillow) |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 도식 두 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF에 저작권 표시는 찾지 못했습니다.
- 논문 도식 두 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2307.04577"). 로봇 사진, 사람 손 사진, 시뮬 화면은 넣지 않았습니다.
- 공식 코드는 `github.com/dexsuite/dex-retargeting`(MIT)과 `github.com/NVlabs/sim-web-visualizer`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

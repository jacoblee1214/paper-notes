# REACT (arXiv:2610.28523) 분석 노트

*VLM-Guided Contact Retargeting and Generative Partner Modeling for Humanoid-Human Interaction* (arXiv v1, 2026-10-20, CoRL 2026 포스터)의 표 수치를 다시 계산하고, 주장을 논문 표와 부록에 대조한 자료입니다. 공식 코드 저장소는 README만 있고 학습 코드는 아직 공개되지 않았습니다 (2026-11-23 확인).

## 구현 가능성 (요약)

- **전체 학습과 실로봇 시험은 일반 노트북에서 불가능합니다.** Isaac Sim(NVIDIA GPU), Inter-X, SMPL-X, Gemini API, Unitree G1이 필요합니다.
- **핵심 아이디어의 축소 토이는 CPU로 시험할 수 있습니다.** 이 노트에는 그 실험을 넣지 않았습니다.
- **코드가 공개되면** 보상 계수, 손실 가중치, 평가 프로토콜을 정적으로 대조할 수 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 결과, 한계, 버전과 라이선스 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 검증, 구현 가능성 판정 |
| `analysis/recompute.py` | 표 읽기, 재계산, 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
# 원문 PDF(arXiv:2610.28523)를 pdftotext -layout으로 옮긴 텍스트를 paper/paper.txt로 저장
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다 (arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 이 노트는 논문 그림을 쓰지 않았습니다. 표 수치는 인용과 재계산을 위해 옮겼고, 그래프, 스크립트, 분석 문서는 이 노트 작성자가 만들었습니다.
- 공식 코드 저장소 (`github.com/YutoShibata07/REACT-HHI`)는 README와 GIF만 있고 라이선스가 없습니다. 이 노트에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.

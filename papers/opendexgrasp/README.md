# OpenDexGrasp (arXiv:2609.18117, CoRL 2026 오럴) 분석 노트

*OpenDexGrasp: Open-vocabulary Task-Oriented Dexterous Grasping* (arXiv v4, 2026-11-10)를 표 재계산과 주장 대조로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive-distrib 1.0). 이 폴더의 그래프 세 장은 우리가 계산하고 그린 것입니다.

## 구현 가능성 (요약)

- **이 맥(Apple M4 Pro, 24 GiB, MPS, CUDA 없음)에서 논문 재현은 불가능합니다.** 공식 코드와 데이터가 아직 공개되지 않았습니다(저장소는 README 한 줄). 학습은 4B 비전-언어 모델과 2M 스텝을 전제하고, 실로봇이 필요합니다.
- **축소판 실험은 조건부로 가능합니다.** 보조 손실과 직접 생성의 효과를 작은 모델로 확인하는 설계입니다(노력 2–3주, 추정).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이, 라이선스 확인 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(가중 평균, SR 분모, 군집 근사 검정, 실로봇 검정, 속도) |
| `analysis/04_claim_verification.md` | 주장 대조(C1–C23), 측정 정의 점검, 저자 질문, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive-distrib 1.0 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다. 원문은 `curl -L -o paper/paper.pdf https://arxiv.org/pdf/2609.18117v4`로 받을 수 있습니다.
- 공식 저장소(https://github.com/opendexgrasp/OpenDexGrasp)는 플레이스홀더입니다. 코드를 받거나 실행하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

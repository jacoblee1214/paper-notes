# Dreamer 4: Training Agents Inside of Scalable World Models (arXiv:2509.24527) 분석 노트

*Training Agents Inside of Scalable World Models* (arXiv v1, 2025-09-29)를 표 재계산과 주장 검증으로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(arXiv non-exclusive 라이선스). 이 폴더의 그래프 두 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 규모 재현은 이 맥(Apple M4 Pro, 24 GiB, MPS, CUDA 없음)에서 불가능합니다.** 2B 모델은 TPU 수백 대에서 학습했고, 공식 코드와 설정이 공개되지 않았습니다.
- **CPU 토이는 가능합니다.** shortcut 손실과 PMPO 손실을 작은 데이터에서 구현해 방향을 확인할 수 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, 코드 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(Wilson 구간, 두 비율 검정, Fisher), 본문과 표의 일관성 |
| `analysis/04_claim_verification.md` | 주장 검증(C1–C18), 공식 코드 부재, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산 (scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 `arxiv.org/licenses/nonexclusive-distrib/1.0/`, 2026-11-21 확인). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드는 찾지 못했습니다(프로젝트 페이지 https://danijar.com/project/dreamer4/ 에 링크 없음).
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

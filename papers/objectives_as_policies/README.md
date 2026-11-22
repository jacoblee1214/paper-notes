# Objectives as Policies (OaP, arXiv:2611.18298) 분석 노트

*Objectives as Policies: Vision-Language Model Cost Programs for Contact-Rich Manipulation* (arXiv v1, 2026-11-13 제출)을 표 재계산, 공식 코드 정적 대조, 검정으로 분석한 자료입니다. 학회 표기는 없습니다. 논문 원문과 논문 그림은 넣지 않았습니다(arXiv non-exclusive 라이선스). 이 폴더의 그래프 두 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, CUDA 없음)에서 할 수 없습니다.** VLM 호출, CUDA 기반 MuJoCo-warp MPC, 장면 재구성, 실로봇이 필요합니다.
- **CPU MuJoCo 축소판은 조건부로 가능합니다.** 비용 항과 단계 검증기, MPPI를 옮기는 작업이며 노력은 1–2주로 추정했습니다(측정 안 함).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 빈칸 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 표 합계 검사, Fisher 검정, Wilson·Clopper-Pearson 구간, 검정력, 제어기 산술 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 일치(M)와 차이(D1–D10), 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/ZihaoLu001/OaP)는 Apache-2.0입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

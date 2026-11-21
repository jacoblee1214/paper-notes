# FTP-1 (arXiv:2606.13102, CoRL 2026 포스터) 분석 노트

*FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation* (arXiv v4, 2026-10-27)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 재현(학습, 시뮬, 실로봇)은 하지 않았습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.13102에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 논문 결과 재현은 불가능합니다.** 사전학습, 파인튜닝, UniVTAC 시뮬(Isaac Sim, NVIDIA GPU), 실로봇은 모두 불가입니다.
- **조건부:** 공식 사전학습 체크포인트의 단일 샘플 추론을 MPS로 옮기는 것(1–2주, 추정). 프로빙 재현.
- **재계산은 완료했습니다.** `analysis/recompute.py`(numpy, scipy, 수 초)로 표 1–4와 검정을 다시 계산했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv abs 페이지의 license 링크가 `creativecommons.org/licenses/by-sa/4.0/`(CC BY-SA 4.0)를 가리킵니다. 그래서 논문 그림은 게시본에 넣지 않았습니다.
- 공식 코드는 `github.com/michaelyuancb/ftp1-policy`에 있고 Apache-2.0입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 체크포인트, 데이터셋은 받지 않았습니다.

# GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation (arXiv:2410.06158) 분석 노트

*GR-2* (arXiv v1, 2024-10-08, 기술 보고서)를 본문 숫자 검산, 시도 수 가정 검정, 주장 검증으로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다. 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 38M 클립 사전학습, 실로봇 시연, CALVIN 학습은 GPU와 실로봇이 필요합니다. 공식 코드와 체크포인트는 공개되지 않았습니다.
- **재계산과 검정은 완료했습니다.** 시도 수가 없어서 실로봇 비율의 구간은 가정에 기댑니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 재현에 빠진 정보 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, 자료와 코드 표 |
| `analysis/03_analysis.md` | 본문 숫자 검산, 시도 수 가정 검정, CALVIN 구간, 빈 피킹과 규모 읽기 |
| `analysis/04_claim_verification.md` | 주장 C1–C11의 검증, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 재계산과 검정 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(abs 페이지의 license 링크, 2026-11-22 확인). PDF에는 저작권 표시가 없습니다. 논문 그림은 실로봇 사진과 영상 프레임이 많아 싣지 않았고, 그래프는 논문 본문과 그림의 값을 옮겨 우리가 그렸습니다.
- 공식 코드는 찾지 못했습니다(프로젝트 페이지 저장소는 웹 빌드만 있음).
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

# ForceMimic: Force-Centric Imitation Learning with Force-Motion Capture System for Contact-Rich Manipulation (arXiv:2410.07554, ICRA 2025) 분석 노트

ForceMimic (arXiv v3, 2025-03-03)을 보고 수치 재계산과 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 두 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 결과의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 손 장치 수집과 실로봇 실행은 하드웨어가 필요하고, 학습은 GPU와 처리된 데이터가 필요합니다.
- **CPU로 가능한 것:** 표 재계산과 검정(완료), 중력 보정과 힘 제어 로직의 합성 검증(시작 가능).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(구간, Fisher 검정, 검정력), 데이터와 힘 수치의 일관성 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 불일치와 빈칸, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 1초 안에 끝남) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 저장소(https://github.com/ForceMimic/hybridil, https://github.com/ForceMimic/forcecapture)는 MIT 라이선스입니다. 이 분석에서는 실행하거나 설치하지 않았고, 얕은 clone으로 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 데이터셋과 체크포인트는 받지 않았습니다.

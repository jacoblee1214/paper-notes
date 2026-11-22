# Wan: Open and Advanced Large-Scale Video Generative Models (arXiv:2503.20314) 분석 노트

Wan 기술 보고서(v2, 2025-04-19)를 구조, 표 재계산, 공식 추론 코드 정적 대조로 읽은 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **추론 14B는 이 맥(Apple M4 Pro, 약 24 GiB, CUDA 없음)에서 불가입니다.** bf16 가중치만 26.6 GiB입니다.
- **추론 1.3B는 조건부입니다.** 장치 지정과 autocast를 포팅해야 하고, 480p 5초 한 편은 연산량으로 보아 1–2시간 안팎으로 추정합니다(측정하지 않음).
- **학습은 불가입니다.** 공식 저장소에 학습 코드가 없습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 주장과 증거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산: VBench 총점, Wan-Bench 가중치 민감도, 승-패 차이의 신뢰구간, 파라미터와 토큰 수, 처리량 추정 |
| `analysis/04_code_verification.md` | 논문과 공식 추론 코드의 대조 (D1–D28), 구현 가능성 판정 |
| `analysis/recompute.py` → `recompute_results.json` | 재계산 스크립트와 결과 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/Wan-Video/Wan2.1)는 Apache-2.0 라이선스입니다. 이 분석에서는 실행하거나 설치하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

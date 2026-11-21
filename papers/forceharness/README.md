# ForceHarness: Turning Force into Action and Physics for Frozen Frontier Models (arXiv:2611.21099 v1) 분석 노트

*ForceHarness* (arXiv v1, 2026-11-16)를 표 재계산과 공식 코드(MIT) 정적 대조로 분석한 자료입니다. 학회 표기는 없습니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 우리가 그린 것입니다.

## 구현 가능성 (요약)

- **부분적으로 가능합니다.** 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서는 힘 추정 식과 프리미티브 루프를 옮길 수 있습니다. HiddenPhys 시뮬레이션은 패키지 설치와 모델 API 비용이 필요합니다.
- **불가능합니다.** ForceBench(Isaac Lab, NVIDIA GPU), 실로봇(Franka FR3), π0.5와 ForceVLA 학습입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 결과, 주장과 증거의 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표, 학회와 라이선스 판정 |
| `analysis/03_analysis.md` | 표 재계산(합계, 평균, Fisher, Wilson, 짝 McNemar), 분포 이동, 형식 비교 분모 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 차이 D1–D7, 구현 가능성 |
| `analysis/recompute.py` → `recompute_results.json` | 표 값의 재계산과 일관성 검사 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-21). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/showlab/ForceHarness)는 MIT 라이선스입니다. 이 분석에서는 실행하거나 설치하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

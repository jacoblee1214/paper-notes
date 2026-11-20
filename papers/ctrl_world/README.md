# Ctrl-World: A Controllable Generative World Model for Robot Manipulation (arXiv:2510.10125, ICLR 2026) 분석 노트

*Ctrl-World* (arXiv v3, 2026-03-01)을 표 재계산, 공식 코드 정적 대조, 작은 numpy 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 논문 수치를 우리가 다시 그린 것과 토이 결과입니다.

## 구현 가능성 (요약)

- **이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 논문 수치의 재현은 불가능합니다.** 학습(2×8 H100 2–3일)과 정책 평가(π0.5 JAX, 실로봇)는 불가능하고, 영상 지표와 정책 개선은 공식 코드에 구현이 없습니다.
- **추론 시연(replay 한 롤아웃)은 조건부로 가능합니다 (미검증).** 체크포인트와 SVD 약 16 GB 다운로드가 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(일관성, 대응 검정, 부트스트랩), 토이, 해석 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 차이(D1–D16), 빈칸, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 수 초) |
| `analysis/toy_rank_noise.py` | 토이: 20회 측정으로 쌍 순서를 맞힐 확률 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_rank_noise.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-20). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/Robert-gyj/Ctrl-World)는 MIT 라이선스입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

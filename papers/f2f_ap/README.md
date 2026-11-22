# F2F-AP (arXiv:2604.02408, CoRL 2026 포스터) 분석 노트

*F2F-AP: Flow-to-Future Asynchronous Policy for Real-time Dynamic Manipulation* (arXiv v4, 2026-10-15)를 표 재계산, 통계 검정, 공식 코드 정적 대조, 지연 토이로 분석한 자료입니다. 논문 원문 PDF와 그림은 넣지 않았습니다(arXiv:2604.02408에서 받을 수 있습니다). 모든 그래프는 직접 그렸습니다.

## 구현 가능성 (요약)

- **실로봇 재현(표 1, 2)은 불가입니다.** UR5e와 UMI 장치, 사람이 건네는 물체, 사족 로봇과 전신 제어기가 필요하고 사족 코드는 공개되지 않았습니다.
- **공식 학습은 CUDA와 deepspeed를 전제합니다.** Apple Silicon(MPS)에서는 설정을 고치지 않고는 돌릴 수 없고 학습 데이터도 공개되지 않았습니다.
- **공식 추론·시각화는 조건부입니다.** 새 환경 설치와 코드 실행이 필요하고 MPS 연산 지원은 측정하지 않았습니다.
- **핵심 가설의 단순 토이는 구현했고 완료했습니다.** `analysis/toy_latency_intercept.py`(numpy, 수 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표, 학회 확인 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, statsmodels, 수 초) |
| `analysis/toy_latency_intercept.py` | 토이: 지연 아래 1차원 포획 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 직접 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy statsmodels matplotlib
python3 analysis/recompute.py
python3 analysis/toy_latency_intercept.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문의 arXiv 라이선스는 arXiv 비독점 배포 라이선스(CC BY 아님)입니다. 그래서 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/Why-peace/F2F-AP`에 있습니다. LICENSE 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

# VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference (arXiv:2512.01031, 학회 표기 없음) 분석 노트

*VLASH* (arXiv v2, 2026-07-26)를 표 재계산, 그림 막대 역산, 공식 코드 정적 대조, 작은 numpy 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv 비독점 배포). 이 폴더의 그래프 네 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** π0.5 파인튜닝, `torch.compile` 지연 측정, LIBERO 평가, 실로봇은 CUDA GPU와 하드웨어가 필요하고, 공개 main 코드는 π0와 π0.5만 지원합니다.
- **MuJoCo 추종 과제의 축소판은 가능합니다.** 지연 무시, 정답 상태, 상태 굴림, 오프셋 학습을 비교합니다(처리량 미측정, 2–3일 추정).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, v1 대비 v2 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표, 학회와 라이선스 |
| `analysis/03_analysis.md` | 표 재계산(정수 가능성, 반응 지연 산술, 효율, 그림 막대 역산), 토이 2개 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조 D1–D14, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 1초 미만) |
| `analysis/digitize_bars.py`, `analysis/fig_digitized.json` | 그림 막대 높이와 에러바를 픽셀에서 읽는 스크립트와 결과 (PDF가 따로 필요) |
| `analysis/toy_attempts.py` | 토이 1: 성공/시도 지표의 성질 (numpy) |
| `analysis/toy_state_proxy.py` | 토이 2: 마지막 행동을 미래 상태로 쓸 때의 오차 (numpy) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_attempts.py
python3 analysis/toy_state_proxy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/mit-han-lab/vlash)는 Apache-2.0입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그림 막대에서 읽은 값은 근사(약 ±0.3%p)입니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

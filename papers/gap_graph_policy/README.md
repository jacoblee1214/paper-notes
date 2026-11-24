# GaP (arXiv:2607.05369, CoRL 2026 Spotlight) 분석 노트

*GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation (VA) Tasks* (arXiv v3, 2026-11-04)를 표 재계산, 통계 검정, 공식 코드 정적 대조, 채택 규칙 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2607.05369에서 받을 수 있습니다). 논문 그림은 세 장(그래프와 도식)만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, CUDA 없음)에서 불가능합니다.** 공식 코드는 RTX 4090급 GPU, Linux EGL, 외부 모델 API를 요구하고, 자기 학습 루프와 팝콘·USB-C·Wash Crates 그래프는 저장소에 없습니다.
- **재계산과 채택 규칙 토이는 끝냈습니다.** `analysis/recompute.py`(수 초)와 `analysis/toy_accept_rule.py`(수 초).
- **방법의 핵심(타입 그래프, 정적 검증, 채택 루프)의 축소 구현은 가능합니다.** 노력 2–4주(추정, 측정 전).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_accept_rule.py` | 토이: 갱신 채택 규칙의 잡음 (numpy, 수 초) |
| `analysis/plot_figures.py` | 우리가 그린 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 세 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_accept_rule.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 라이선스 링크 확인). 원문 PDF에서 저작권 표시는 찾지 못했습니다.
- 논문 그림 세 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2607.05369"). 실로봇 사진, 시뮬레이션 렌더, 제3자 로고가 들어간 그림은 넣지 않았고, 그림 2는 렌더 패널을 뺀 그래프 부분만 실었습니다.
- 공식 코드는 `github.com/graph-robots/graph-as-policy`(LICENSE는 Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

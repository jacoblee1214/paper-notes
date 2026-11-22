# FlowDAgger (arXiv:2607.08877, CoRL 2026 스포트라이트) 분석 노트

*FlowDAgger: Human-in-the-Loop Adaptation of Generative Robot Policies in Latent Space*를 수치 재계산, 통계 검정, 공식 코드 정적 대조로 읽은 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2607.08877에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **이 맥(Apple M4 Pro, MPS, CUDA 없음)에서 논문의 표를 재현할 수 없습니다.** 기준선, Cosmos-Policy, 실로봇 코드가 공개 저장소에 없고, 공식 코드는 JAX CUDA를 전제하며 체크포인트가 약 12.4 GB입니다.
- **방법의 핵심(작은 흐름 정합 정책, 행동 역변환, 잡음 정책 회귀)은 소형 연속 제어에서 직접 구현할 수 있습니다.** 계획 단계이고 2–4일로 추정합니다(측정 전).
- **재계산과 numpy toy는 `analysis/recompute.py`로 수 초에 다시 할 수 있습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그래프 네 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 동시기 연구, 공식 저장소 표 |
| `analysis/03_analysis.md` | 수치 재계산 V1–V16, numpy 실험 K1–K4, 주장 판정 C1–C10 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조 D1–D18, 구현 가능성 |
| `analysis/recompute.py`, `analysis/recompute_results.json` | 표와 그림 수치의 재계산, 검정, 역변환 toy |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 비독점 배포 라이선스(arxiv.org/licenses/nonexclusive-distrib/1.0)로 올라와 있습니다. CC BY 4.0이 아니므로 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/microsoft/FlowDAgger`에 MIT 라이선스로 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 프로젝트 페이지(`microsoft.github.io/FlowDAgger`)는 CC BY-SA 4.0으로 표기되어 있습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 재계산 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.

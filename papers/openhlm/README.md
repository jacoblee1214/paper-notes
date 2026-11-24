# OpenHLM (arXiv:2606.22174, CoRL 2026 Spotlight) 분석 노트

*OpenHLM: An Empirical Recipe for Whole-Body Humanoid Loco-Manipulation* (arXiv v2, 2026-11-02)을 표와 그림 수치 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2606.22174에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **실로봇 평가와 학습(4 × A800, 약 24 h)은 이 맥에서 불가능합니다.** Unitree G1, PICO4U, 카메라, 그리퍼가 필요하고 GPU 학습이 전제입니다.
- **코드 정적 대조와 수치 재계산은 끝냈습니다.** `analysis/recompute.py`(수 초, numpy, scipy)로 재현됩니다.
- **예시 시연 변환과 체크포인트 오프라인 점검은 조건부입니다.** 약 0.49 GB 예시 데이터와 7.47 GB 체크포인트 다운로드, 설치와 공식 코드 실행이 필요하고 Apple GPU(MPS) 동작은 시험하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 보고 수치, v1 → v2 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 본문 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 표와 그림 수치 재계산, 롤아웃 점수 복원, 검정 |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문의 arXiv 라이선스는 비독점 배포(`arxiv.org/licenses/nonexclusive-distrib/1.0/`)이며 CC BY가 아닙니다. 그래서 논문 그림은 넣지 않았고, 표와 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그림 막대 위 숫자를 읽은 값은 "그림 판독 값"으로 표시했습니다.
- 공식 코드는 `github.com/OpenHLM-project/OpenHLM`(Apache-2.0)에 있습니다. 이 노트에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 체크포인트와 데이터셋은 받지 않았고 크기는 Hugging Face API 메타데이터에서 읽었습니다.
- 그래프, 분석 문서와 스크립트는 이 노트의 작성자가 만들었습니다.

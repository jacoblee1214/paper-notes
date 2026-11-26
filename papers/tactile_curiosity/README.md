# TacEx (arXiv:2609.40134, CoRL 2026 포스터) 분석 노트

*Tactile Curiosity Drives Robot Interaction* (arXiv v3, 2026-10-13)을 그림에서 읽은 값의 재계산, 통계 검정, 공식 코드의 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 공식 코드는 넣지 않았습니다(원문은 arXiv:2609.40134에서 받을 수 있습니다). 논문 그림은 도식과 차트 네 장만 출처와 함께 넣었습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 불가능합니다.** LIBERO 확산 조종은 π0 가중치와 CUDA JAX가 전제이고, 무보상 탐색 환경은 공식 저장소의 고정 서브모듈에 없습니다.
- **공식 코드의 CPU 스모크 테스트는 조건부입니다.** 고정 의존성 설치가 필요합니다. 이 분석에서는 코드를 실행하지 않았습니다.
- **보너스 계산의 numpy 이식은 완료했습니다.** `analysis/toy_bonus_weights.py`(수 초). 소형 MuJoCo 환경에서의 재구현은 새 설치 없이 가능합니다(노력 2–3주 추정, 측정 전).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 네 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 저장소 표 |
| `analysis/03_analysis.md` | 그림 판독 값 재계산, 일관성 검사, 토이, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조, 라이선스, 구현 가능성 |
| `analysis/extract_figure_values.py` | 논문 PDF의 벡터 그래프에서 값을 읽음 (pymupdf, 입력은 arXiv v3 PDF) |
| `analysis/recompute.py` | 읽은 값으로 검정과 시드 수 추정 (scipy, 수 초) |
| `analysis/toy_bonus_weights.py` | 식 4와 코드의 보너스 계산 비교 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/crop_figures.py` | 논문 PDF에서 도식과 차트 네 장을 자름 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 네 장(© 저자, CC BY 4.0) |

## 재실행

논문 PDF를 `paper/paper.pdf`로 받습니다: `curl -L -o paper/paper.pdf https://arxiv.org/pdf/2609.40134v3`

```bash
pip install numpy scipy matplotlib pymupdf Pillow
python3 analysis/extract_figure_values.py
python3 analysis/recompute.py
python3 analysis/toy_bonus_weights.py
python3 analysis/plot_figures.py
python3 report/crop_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 논문 그림 네 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2609.40134 (v3)"). 시뮬레이션 렌더 프레임이 있는 부분은 싣지 않았습니다.
- 공식 코드는 `github.com/klemensiten/tacex`에 있고 LICENSE 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고, 정적으로만 읽었으며 실행하거나 설치하지 않았습니다.
- 그림에서 읽은 값은 저자가 표로 낸 수치가 아닙니다. 그래프, 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

# MESA (arXiv:2611.05567, CoRL 2026 포스터) 분석 노트

*MESA: An Evaluation Framework for Compositional, Semantic, and Spatial Generalization in Robotics* (arXiv v1, 2026-11-03)를 부록의 과제별 표 재계산, 통계 검정, 과제 부트스트랩, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.05567에서 받을 수 있습니다). 논문 그림은 세 장(그림 3, 4, 5)만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **정책 평가 수치(표 2, 3) 재현은 이 맥에서 불가입니다.** 기준선 학습이 H100 2장 21시간입니다.
- **평가 harness 설치와 MimicGen 데이터 생성은 조건부입니다.** 자산 5–13 GB 다운로드, 새 설치, 공식 코드 실행이 필요하고 macOS 렌더링은 확인하지 못했습니다.
- **표 재계산, 순위 안정성, 코드 정적 대조는 완료했습니다.** `analysis/recompute.py`(수 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 벤치마크 구성, 실험, 통계 보고 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V15, 주장 판정 C1–C13 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조 D1–D14, 구현 가능성 |
| `analysis/recompute.py` | 표 파싱과 재계산, 검정, 과제 부트스트랩 (numpy, scipy, 수 초) |
| `analysis/repo_static_stats.py` | 공식 저장소의 평가 BDDL 구조 통계 (표준 라이브러리, 저장소 복제본 필요) |
| `analysis/plot_figures.py`, `report/crop_figures.py` | 그래프 생성, 논문 그림 자르기 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 세 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py      # 입력: 논문 PDF에서 pdftotext -layout 으로 만든 paper/paper.txt
python3 analysis/plot_figures.py
```

`recompute.py`는 `paper/paper.txt`를 읽습니다. 원문을 arXiv에서 받아 `pdftotext -layout`으로 만들어 두세요.

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 세 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2611.05567"). 사진, 시뮬레이션 장면 프레임이 있는 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/pairlab/mesa-env`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

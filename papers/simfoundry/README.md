# SimFoundry (arXiv:2606.28276, CoRL 2026 포스터) 분석 노트

*SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation* (arXiv v4, 2026-08-05)를 표 재계산, 통계 검정, 시행 잡음 토이, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2606.28276에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **영상 → 장면 재구성, 시뮬 평가, 정책 학습, 실로봇은 Apple M4 Pro(CUDA 없음)에서 불가입니다.** Linux + NVIDIA GPU, 약 250 GB, Gemini 원격 호출이 필요하고 학습·지표·벤치마크 코드는 공개 저장소에 없습니다.
- **표 재계산, 검정, 시행 잡음 토이는 완료했습니다.** `analysis/recompute.py`(numpy, scipy, 수 초).
- **개념 재구현(물리 안정화, 술어 배치 샘플러, 재구성 지표)은 CPU로 가능합니다.** 각 1–3일(추정, 측정 전).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이(v1→v4) |
| `analysis/02_prior_work.md` | 선행 계보, 구성요소 출처, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V17, 일관성 검사 K1–K13, 주장 판정 C1–C11, 토이 T1 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D16, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산, 검정, 시행 잡음 토이 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY-SA 4.0으로 표시되어 있습니다(arXiv 라이선스 페이지 확인). 그러나 PDF 첫 쪽에 "(c) 2026 NVIDIA. All rights reserved." 고지가 있어 논문 그림은 쓰지 않았습니다.
- 공식 코드는 `github.com/NVlabs/SimFoundry`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 일부 의존성은 비상업이거나 지역 제한이 있습니다(`analysis/04_code_verification.md` §1).
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치, 예시 자산, 데이터셋은 받지 않았고 외부 API도 호출하지 않았습니다.

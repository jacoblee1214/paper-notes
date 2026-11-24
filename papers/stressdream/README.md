# StressDream (arXiv:2606.00267, CoRL 2026 포스터) 분석 노트

*StressDream: Steering Video World Models for Robust Policy Evaluation and Improvement* (arXiv v2, 2026-10-06)를 표·그림 수치 재계산, 공식 코드 정적 대조, 타이피컬 셋 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.00267에서 받을 수 있습니다). 논문 그림은 네 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 수치 재현은 일반 노트북에서 어렵습니다.** Vista는 VRAM 48–80 GB, Ctrl-World는 약 20 GB와 Qwen3-VL이 필요하고 정책 개선은 H100과 실로봇이 필요합니다.
- **알고리즘 축소판과 토이는 CPU로 가능합니다.** `analysis/toy_typical_set.py`(torch, 1초 미만)와 `analysis/recompute.py`(numpy, scipy, 수 초)로 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 네 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 코드 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V10, 일관성 K1–K8, 토이, 주장 판정 C1–C10 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D17, 구현 가능성 |
| `analysis/recompute.py` | 표·그림 수치 재계산과 코드 상수 기준값 |
| `analysis/toy_typical_set.py` | 토이: 타이피컬 셋 통계와 이동량 |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/crop_figures.py` | 논문 그림 네 장 잘라내기 (원문 PDF 필요) |
| `report/figures/` | 직접 그린 그래프 네 장과 논문 그림 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_typical_set.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 논문 그림 네 장(그림 2(a), 5, 8, 14)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.00267"). 사진, 시뮬레이터 프레임, 이모지 아이콘이 있는 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/CMU-IntentLab/StressDream`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 정적으로만 읽었으며 실행하거나 설치하지 않았습니다.
- 그림 판독 값과 가정한 시행 수는 문서 안에 표시했습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

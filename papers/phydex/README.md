# PhyDEX (arXiv:2611.26565) 분석 노트

*PhyDEX: A Benchmark for Physical Reasoning in Vision-Language-Action Models via Dynamic Extrinsic Dexterity* (Vo 외, 2026)를
논문의 표·그림·본문 수치와 대조해 분석한 자료입니다. 학회 표기는 확인할 수 없고, 공식 코드와 데이터는 공개되지 않았습니다(프로젝트 페이지 404).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 재현에 필요한 정보와 빠진 정보 |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소, GitHub 확인 결과 |
| `analysis/03_analysis.md` | 표 재계산, Fisher 검정, 일관성 검사 I1–I23, 식 (2) 점검 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C15, 재현 가능성, 구현 가능성 판정 |
| `analysis/recompute.py` | 표 재계산과 검정 (수 초, 표준 라이브러리와 scipy) |
| `analysis/plot_figures.py` | 논문 수치로 그린 그래프 |

## 재실행

```bash
python3 analysis/recompute.py      # recompute_results.json 생성
python3 analysis/plot_figures.py   # report/figures/own_*.png 생성 (matplotlib 필요)
```

## 출처와 라이선스

- 논문은 CC BY 4.0입니다(arXiv abs의 license 링크와 PDF XMP 기준). 그래도 이 저장소에는 **논문 그림을 넣지 않았습니다.** 그림 대부분에 시뮬 렌더, 데이터셋 자산, 실물 사진, 로고가 있기 때문입니다.
- `report/figures/own_*.png`는 논문의 표와 그림 수치로 직접 그린 그래프입니다.
- 분석 문서와 스크립트는 이 저장소의 작성자가 만들었습니다.
- 원문: https://arxiv.org/abs/2611.26565

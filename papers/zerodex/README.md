# ZeroDex (arXiv:2606.19340) 분석 노트

*ZeroDex: Zero-Shot Long-Horizon Dexterous Manipulation via Multi-View 3D-Grounded VLM Reasoning*
(Kim, Baik, Kim, Kim, Lee, Choi, Park, Choi, Joo, 서울대·고려대, 2026)을 표 재계산, 핵심 식 toy 구현,
arXiv v1 ↔ v3 대조로 분석한 자료입니다. 공식 코드는 아직 공개되지 않았습니다(2026-10-28 기준).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보와 구성 요소별 출처 |
| `analysis/03_derivations.md` | 다시점 합의의 성질과 toy, 두 앵커 정렬, 어포던스 visual hull, Table IV 추론 |
| `analysis/04_claim_verification.md` | 수치 재계산 C1–C9, v1 ↔ v3 대조, 재현 정보 체크리스트, 구현 가능성 |
| `analysis/table_check.py` | Table I–V 재계산과 버전 비교 |
| `analysis/grounding_toy.py` | Eq. 5–6 다시점 합의 vs 단일 RGB-D 역투영 toy |
| `analysis/alignment_toy.py` | Eq. 7 두 앵커 정렬 toy |
| `analysis/plot_headline.py` | 리포트 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/grounding_toy.py && python3 analysis/alignment_toy.py
cd analysis && python3 plot_headline.py
```

모두 CPU에서 몇 초 안에 끝납니다. 논문 PDF는 필요하지 않습니다(수치는 스크립트에 출처와 함께 적었습니다).

## 출처와 라이선스

- 논문은 **CC BY 4.0**입니다(arXiv abs 페이지 확인). `report/figures/paper_*.png`는 논문 v3에서 잘라낸 그림·표이며 © 저자, CC BY 4.0, 출처 arXiv:2606.19340입니다.
- `headline_success.png`는 논문 Table I의 수치로, `grounding_toy.png`는 직접 구현한 toy로 그렸습니다. toy의 VLM 잡음 모델과 카메라 배치는 가정입니다.
- 원문 PDF는 이 저장소에 넣지 않았습니다. 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

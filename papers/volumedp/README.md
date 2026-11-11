# VolumeDP (arXiv:2603.17720) 분석 노트

*VolumeDP: Modeling Volumetric Representation for Manipulation Policy Learning*
(Zhou 외, IROS 2026 채택)을 원문 대조, 표 재계산과 통계 검정, CPU 토이로 분석한 자료입니다.
공개 저장소(ManiSkill 포크, Apache-2.0)를 정적으로 읽어 논문과 대조했습니다. 코드 실행과 재현 실험은 하지 않았습니다.

**구현 가능성 요약:** 공개 코드에는 방법 구현이 없습니다. LIBERO 코드 링크는 404입니다. 공식 재현 경로(ManiSkill GPU 시뮬, LIBERO, 실물)는 이 맥에서 모두 열려 있지 않습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 요약, v1 대 v2 차이, 원문 안의 불일치 |
| `analysis/02_prior_work.md` | 선행연구, 기준선 출처 검증, GitHub 표 |
| `analysis/03_analysis.md` | 표 재계산, 실물 통계(Fisher, MH, Holm, 검정력), LIBERO Welch, 토이 결과, 공정성 판정 |
| `analysis/04_code_verification.md` | 논문 ↔ 공개 코드 대조(D1–D16), 구현 가능성 |
| `analysis/volumedp_check.py` | 표 재계산과 통계 (약 0.2초, scipy 사용) |
| `analysis/volumedp_toy.py` | 관측 표현 비교 토이 (약 2초, numpy 1스레드) |
| `analysis/plot_figures.py` | 직접 만든 그래프 세 장 |

## 재실행

```bash
python3 analysis/volumedp_check.py
python3 analysis/volumedp_toy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 이 저장소에는 **논문 그림, 논문 표 이미지, 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 표의 수치와 직접 만든 토이로 그렸습니다.
- 공개 코드는 Apache-2.0이며 클론만 했고 재배포하지 않습니다. 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

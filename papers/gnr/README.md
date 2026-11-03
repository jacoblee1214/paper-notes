# GNR: Generative Neural Retargeting (arXiv:2610.12440) 분석 노트

*Generative Neural Retargeting for Human-to-Robot Dexterous Manipulation*
(Gao, Yang et al., Meta Reality Labs Research, 2026)을 표·부록 수치 재계산, pass@S 곡선의 이질성 모델 적합,
manifold 가설 toy로 분석한 자료입니다. 공식 코드와 데이터가 공개되지 않아(2026-11-03 확인) 코드 대조 대신 주장 검증을 했고,
재현 실험은 하지 않았습니다. 프로젝트 페이지: https://yy-gx.github.io/GNR/

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 리타겟팅 세 방식 비교, 관련 GitHub 표 |
| `analysis/03_analysis.md` | Table 1 분해, pass@S 이질성 모델(꼬리 지수 a), 조기 종료 비용, manifold toy, "시연" 단위, 비용 회계 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C12, 재현 정보 체크리스트, 구현 가능성 |
| `analysis/table_check.py` | 표·본문·부록 수치 재계산 |
| `analysis/scaling_toy.py` | pass@S Beta-혼합 적합과 조기 종료 기대 표본 |
| `analysis/manifold_toy.py` | 생성기 대용(최근접 시드 + 국소 보정) 대 annealed CEM toy |
| `analysis/plot_figures.py` | 리포트 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/scaling_toy.py && python3 analysis/manifold_toy.py
cd analysis && python3 plot_figures.py
```

모두 CPU 단일 스레드에서 수 초에서 수십 초 안에 끝납니다. 논문 PDF는 필요하지 않습니다(수치는 스크립트에 출처와 함께 적었습니다).

## 출처와 라이선스

- 논문은 **CC BY-NC-ND 4.0**입니다(arXiv abs 페이지 확인). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- `report/figures/*.png`는 논문 표의 수치와 직접 만든 toy로 그린 그래프입니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

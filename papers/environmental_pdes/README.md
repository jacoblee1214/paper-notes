# Localization in Spatiotemporal Fields via Environmental PDEs (arXiv:2608.00272) 분석 노트

*Localization in Spatiotemporal Fields via Environmental PDEs* (Fuentes, Newaz, Cavalcanti, Bobadilla; IROS 2026 채택)를
논문 수치 재계산, 자세 관측가능성 계산, 1차 풍상의 수치 확산 검사, 이류–확산 위치추정 toy로 분석한 자료입니다.
공식 코드는 공개되지 않아(2026-11-10 확인) 코드 대조는 하지 않았습니다.

**구현할 수 있는지:** 공식 재현은 불가능합니다(코드, 지형·바람 자료, 수치 방법, 현장 데이터 미공개). 이류–확산 규모의 아이디어 재구현은 쉬운 편이고,
얕은 물 방정식 규모는 중간에서 어렵습니다. 실물 재현은 현장 장비가 필요합니다(`analysis/04_claim_verification.md` §4).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | 논문 구조, 장 모형, RBPF, 실험, 읽으면서 본 문제 |
| `analysis/02_prior_work.md` | 선행연구 계보, 가까운 대안, GitHub 표 |
| `analysis/03_analysis.md` | 수치 재계산, 관측가능성, CFL과 수치 확산, toy 결과 |
| `analysis/04_claim_verification.md` | 권리 판정, 주장 검증, 빠진 정보, 구현 가능성 |
| `analysis/table_check.py` | 논문 수치 재계산(Welch, 민감도, 검출력) |
| `analysis/pde_toy.py` | 2D 이류–확산 유한차분 + 입자 필터 toy |
| `analysis/obs_toy.py` | 자세 피셔 정보, 회전 판별력, 수치 확산 검사 |
| `analysis/extra_kappa_scheme.py` | 수치 방식(1차 풍상, Lax–Wendroff)에 따른 κ_eff와 필터 오차 |
| `analysis/extra_flat_bias.py` | 평평한 장에서 편향–장 수준 혼동과 편향 사전 분포의 영향 |
| `analysis/extra_rbpf_bias.py` | 편향 증강 표준 PF 대 RBPF(같은 입자 수·시드) |
| `analysis/extra_neural_field.py` | 작은 MLP 장 대리 모형을 측정 모형으로 쓴 입자 필터와 κ 복원 (torch 필요) |
| `analysis/plot_figures.py` | toy 그래프 |
| `report/crop_figures.py` | 논문 그림을 PDF에서 자르는 스크립트 |

## 재실행

```bash
pip install numpy scipy matplotlib pillow
python3 analysis/table_check.py && python3 analysis/pde_toy.py && python3 analysis/obs_toy.py
python3 analysis/plot_figures.py
```

`analysis/extra_neural_field.py`는 torch가 필요하고 CPU에서 약 20초 걸립니다(`pip install torch`).

모두 CPU에서 몇 초 안에 끝납니다. toy의 장, 잡음, 경계 조건, κ 값은 가정입니다.

## 출처와 라이선스

- 논문은 **CC BY 4.0**입니다(arXiv abs 페이지 확인). PDF에 다른 권리 표기가 없습니다. `report/figures/paper_*.png`는 논문에서 잘라낸 그림이며 © 저자, CC BY 4.0, 출처 arXiv:2608.00272입니다.
- 나머지 그래프는 논문 수치와 toy 시뮬레이션으로 직접 그렸습니다. 원문 PDF는 이 저장소에 넣지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

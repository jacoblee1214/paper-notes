# ResTacVLA (arXiv:2607.03387) 분석 노트

*Feeling the Unexpected: ResTacVLA for Contact-Rich Manipulation via Residual Tactile Representation*
(Zhang et al., IROS 2026)을 논문 표 재계산과, 잔차 촉각·surprise 게이트 가정의 numpy toy로 분석한 자료입니다.
공식 코드와 데이터는 공개되지 않았습니다("coming soon", 2026-11-09 확인).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 가까운 대안, 관련 GitHub |
| `analysis/03_analysis.md` | Table I–III 재계산(Fisher·CMH·단계 분해), Eq. 1의 σ 유도, 게이트·잔차 toy |
| `analysis/04_claim_verification.md` | 주장별 판정(C1–C14)과 구현 가능성 |
| `analysis/table_check.py` | 표 재계산 (1초 미만) |
| `analysis/gate_toy.py` | σ 게이트 대 예측 오차 게이트, 잔차 대 원시 촉각 toy (numpy 1스레드, 약 30초) |
| `analysis/plot_figures.py` | 리포트 그래프 네 장 |
| `analysis/extra_flow_gate_sim.py` | 자체 작성 MuJoCo 벽 충돌 장면에서 흐름 정책과 게이트 4종을 폐루프로 비교 (CPU, 약 2분) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/gate_toy.py
python3 analysis/extra_flow_gate_sim.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 표의 수치(`success_ci.png`, `phase_decomposition.png`)와 직접 만든 toy(`gate_toy.png`, `residual_vs_raw.png`)로 그렸습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

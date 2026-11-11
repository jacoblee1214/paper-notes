# Temporal Policy (arXiv:2607.29482) 분석 노트

*Temporal Policy: History-Initialized Action Generation for Robotic Learning from Demonstration*
(Miller & Jagersand, IROS 2026 채택)을 표 재계산, 결합 비용 항등식, CPU 토이로 분석한 자료입니다.
공식 코드(MIT, dmiller12/TemporalPolicy, e1e7ee5)를 읽어 논문과 대조했습니다. 재현 실험은 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림·표 제외) |
| `analysis/01_paper_analysis.md` | 논문 요약과 논문 안의 불일치 |
| `analysis/02_prior_work.md` | 선행연구 계보, 가까운 대안, GitHub 표 |
| `analysis/03_analysis.md` | 표 재계산, 결합 비용 항등식, 샘플러 토이 |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조, 구현 가능성, 권리 표기 |
| `analysis/table_check.py` | 표 재계산 (1초 미만) |
| `analysis/transport_toy.py` | 이력·잡음 원천 오라클 토이 (약 13초, numpy 1스레드) |
| `analysis/extra_flow_mode.py` | 학습된 이력 조건부 흐름의 모드 붕괴와 SDE 샘플러 (torch 필요, CPU 약 35초) |
| `analysis/plot_figures.py` | 직접 만든 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/transport_toy.py
python3 analysis/extra_flow_mode.py  # torch 필요
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 이 저장소에는 **논문 그림, 논문 표 이미지, 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 표의 수치와 직접 만든 토이로 그렸습니다.
- 공식 코드는 MIT이며 클론만 했고 재배포하지 않습니다. 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

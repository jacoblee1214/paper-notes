# TaF-Foundation (arXiv:2611.20718 v1) 분석 노트

*TaF-Foundation: Wrench-Grounded Cross-Sensor Tactile Representations* (arXiv v1, 2026-11-16)을 표 재계산과 통계 검정으로 읽은 자료입니다. 학회 표기는 없습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.20718에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **원 규모 재현은 이 맥에서 불가능합니다.** 데이터(1억 프레임)와 가중치가 공개되지 않았고, 사전학습은 GPU 여러 장, 시뮬과 실로봇은 NVIDIA GPU와 실로봇이 필요합니다.
- **방법의 소형 재구현은 조건부로 가능합니다.** 합성 센서에서 입력 어댑터와 세 손실을 구현하는 작업입니다. 노력은 1–2주로 추정합니다(측정 전).
- **재계산은 가능하고 이미 끝냈습니다.** `analysis/recompute.py`는 numpy와 scipy만 씁니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전과 학회 표기 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 기준선 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 검증, 권리 판정, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 표 재계산과 검정 (수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 비독점 배포 라이선스(nonexclusive-distrib 1.0)로 올라와 있습니다. CC BY가 아니므로 논문 그림을 재배포하지 않았습니다.
- 그래프와 표 재계산, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다. 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다.
- 공식 코드 저장소는 없습니다. 프로젝트 페이지 전용 저장소(`mrJiangjm/TaF-Foundation`)는 페이지 자산만 있고 라이선스가 없습니다. 이 자료는 그 저장소를 클론하거나 실행하지 않았습니다.
- 데이터와 가중치는 받지 않았습니다.

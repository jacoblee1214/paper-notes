# Sling2Sim2Real (arXiv:2607.23268) 분석 노트

*Sling2Sim2Real: One-Shot Elastic System Identification for Non-Destructive Slingshot Policy Learning* (Kang, Kim, Song, Park; IROS 2026 채택)을
표 재계산, 통계 검정, 식별성 계산, CPU toy로 분석한 자료입니다. 공식 코드는 공개되지 않아(2026-11-11 확인) 코드 대조는 하지 않았습니다.

**구현할 수 있는지:** 공식 재현(Isaac Sim, 실물 로봇과 고무줄)은 GPU와 실험 장비가 필요해 일반 노트북에서는 할 수 없습니다. 손실과 최적화 흐름은 CPU toy로 구현했습니다(`analysis/04_claim_verification.md` §4).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 읽으며 본 문제 |
| `analysis/02_prior_work.md` | 선행연구 계보, 가까운 대안, 공식 코드 검색 결과 |
| `analysis/03_analysis.md` | 표 재계산, Welch 검정, 유도, toy 결과 |
| `analysis/04_claim_verification.md` | 주장 검증, 누락된 정보, 구현 가능성 |
| `analysis/table_check.py` | 표 산술과 검정 |
| `analysis/sysid_toy.py` | 1자유도 탄성 밴드 toy: 피셔 정보, DE + 공분산 CEM, 착탄 전파 |
| `analysis/model_error_check.py` | 보정 오차가 착탄 오차로 옮겨지는 정도 |
| `analysis/plot_figures.py` | toy 그래프 |
| `report/crop_figures.py` | 논문 그림을 PDF에서 자르는 스크립트 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/sysid_toy.py      # 약 2분 (CPU, 1스레드)
python3 analysis/model_error_check.py
python3 analysis/plot_figures.py
```

toy의 모델, 잡음, 손실 가중치, 발사각은 가정입니다. 논문의 Isaac Sim 설정과 다릅니다.

## 출처와 라이선스

- 논문은 **CC BY 4.0**입니다(arXiv 페이지와 PDF 메타데이터). PDF에 다른 권리 표기가 없습니다. `report/figures/paper_fig*.png`는 논문에서 잘라낸 그림이며 © 저자, CC BY 4.0, 출처 arXiv:2607.23268입니다.
- 나머지 그래프는 논문 수치와 toy 시뮬레이션으로 직접 그렸습니다. 원문 PDF는 이 저장소에 넣지 않았습니다(1.6 MB).
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

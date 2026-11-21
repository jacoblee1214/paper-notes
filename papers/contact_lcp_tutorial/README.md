# A Tutorial on Complementarity-Based Contact Models for Robotics (arXiv:2611.26600) 분석 노트

arXiv v1(2026-11-19) 튜토리얼을 식 재유도와 작은 numpy 토이로 분석한 자료입니다. 학회 표기는 없습니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 네 장은 우리가 만든 것입니다.

## 요약

- 식(원뿔 투영, SOCFB 자코비안, 운동에너지 항등식, 정규화 NLS와 QCQP 동치)은 대체로 맞습니다.
- 볼록 완화(de Saxcé 항 제거)는 미끄럼 접촉을 μ|v_t⁺|만큼 들어 올리고, 접선 힘도 바꿉니다.
- 정규화는 정적 마찰을 없앱니다. 접선 힘은 W/(1+R)입니다.
- 두 문장은 과장되었거나 맞지 않습니다. 네 모서리 상자의 힘에서 "합만 유일"은 랭크 계산과 맞지 않고, 정규화의 최소 노름 선택은 극한에서만 성립합니다.
- 공식 코드와 프로젝트 페이지는 찾지 못했습니다. 검증은 주장별로 했고, 시뮬레이터 실험은 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 주장과 증거, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 자료 표 |
| `analysis/03_derivations.md` | 식 재유도와 수치 검증 V1–V10 |
| `analysis/04_claim_verification.md` | 주장별 판정 C1–C23, 구현 가능성, 저자에게 물을 질문 |
| `analysis/formulation_checks.py` | 토이 V1–V10 (numpy, scipy, 약 20초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/formulation_checks.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-21). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 논문에 연결된 코드는 찾지 못했습니다. MuJoCo, Newton, Simple 저장소는 메타데이터만 확인했고 clone하지 않았습니다.
- 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

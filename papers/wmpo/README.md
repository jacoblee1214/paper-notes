# WMPO (arXiv:2511.09515, ICLR 2026) 분석 노트

*WMPO: World Model-based Policy Optimization for Vision-Language-Action Models* (arXiv v1, 2025-11-12)을 표 재계산, 정수 검사, 통계 검정, 토이 모형으로 읽은 자료입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2511.09515에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- 논문 수치의 재현은 H100급 GPU가 필요해 이 노트의 환경에서는 불가능합니다.
- 공식 세계모델 추론 한 번은 가중치 다운로드 뒤에 조건부로 가능합니다.
- 축소판 재구현은 가능합니다 (추정 1–2주).

## 파일

| 경로 | 내용 |
|---|---|
| `01_paper_analysis.md` | 논문 분석 (방법, 실험, 버전과 학회, 주장과 근거의 간격) |
| `02_prior_work.md` | 학회 확인, 선행연구 계보, 공식 저장소 표 |
| `03_analysis.md` | 표 재계산, 정수 검사, 검정, 토이, 주장 판정 |
| `04_code_verification.md` | 공식 코드 대조 (D1–D9), 구현 가능성 |
| `recompute.py`, `recompute_results.json` | 표 재계산과 Fisher 검정 (수 초) |
| `toy_reward_exploit.py`, `toy_reward_exploit_results.json` | 토이: 보상 판정기 거짓 양성과 GRPO 쏠림 (numpy, 약 19초) |
| `plot_figures.py` | 그래프 두 장 생성 |
| `report_public.pdf` | 공개용 리포트 |
| `figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
python3 recompute.py
python3 toy_reward_exploit.py
python3 plot_figures.py   # figures/ 를 쓰려면 analysis 폴더 구조 필요
```

## 라이선스와 출처

- 논문: arXiv 페이지의 라이선스 링크는 CC BY 4.0입니다 (2026-11-20 확인). PDF에 권리 고지는 없습니다.
- 공식 코드: GitHub WM-PO/WMPO, Apache-2.0. 이 노트에는 코드를 넣지 않았고 실행하지 않았습니다.
- 그래프는 우리가 만든 것입니다. 표 값은 논문 표 1과 표 2에서 옮겼습니다.

# GR00T N1 (arXiv:2503.14734) 분석 노트

*GR00T N1: An Open Foundation Model for Generalist Humanoid Robots* (arXiv v2, 2025-03-27)를 표 재계산과 공식 코드(`n1-release` 태그) 정적 대조로 분석한 자료입니다. 학회 표기 없는 NVIDIA 기술 보고서입니다. 논문 원문과 논문 그림은 넣지 않았습니다. 이 폴더의 그래프 세 장은 논문 표 숫자로 우리가 그린 것입니다.

## 구현 가능성 (요약)

- **이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 논문 주장은 재현할 수 없습니다.** 사전학습(약 50,000 H100 시간)과 신경 궤적 생성(L40 약 105k 시간)은 자원상 불가능하고, 생성 파이프라인은 공개되지 않았습니다.
- 공식 체크포인트 추론은 CUDA와 flash-attn을 전제하므로 MPS 포팅 후에만 가능합니다. 전체 미세조정은 fp32 기준 약 35 GB가 필요해 불가능합니다.
- 이 폴더는 표 재계산, 쌍체 검정, 일관성 점검, 코드 대조, 부호 토이 결과만 담습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 부록, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, 공식 코드와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산, 일관성 점검, 쌍체 검정, 부호 토이 |
| `analysis/04_code_verification.md` | 논문과 공식 코드 대조, 구현 가능성 판정 |
| `analysis/tables.py` | 논문 표 숫자 (출처 표 번호 표기) |
| `analysis/recompute.py` | 재계산과 검정 (numpy, scipy, 1초 안팎) |
| `analysis/toy_flow_sign.py` | 식 (1) 부호와 오일러 단계 수 토이 (numpy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_flow_sign.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에서 CC BY 4.0으로 공개되어 있습니다 (abs 페이지 "view license", 2026-11-21 확인). 그러나 PDF에는 "© 2025 NVIDIA. All rights reserved." 표기가 있어 논문 그림은 싣지 않았습니다.
- 공식 코드 `NVIDIA/Isaac-GR00T`는 Apache-2.0입니다. `n1-release` 태그를 정적으로만 읽었고 실행하거나 설치하지 않았습니다.
- 모델 가중치 `nvidia/GR00T-N1-2B`는 NVIDIA One-Way Noncommercial 라이선스로 표기되어 있습니다. 가중치는 받지 않았습니다.
- 논문 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

# AnyDexRT (arXiv:2607.08341, CoRL 2026 포스터) 분석 노트

*AnyDexRT: Calibration-Efficient Dexterous Hand Retargeting with Few-Shot Human Guidance* (arXiv v2, 2026-11-05)를 표 재계산, 시드·작업자 군집 검정, 공식 코드 정적 대조, 작은 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2607.08341에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **방법(손가락별 MLP 매퍼, 네 손실, 최근접 이웃 역변환, 접촉 분류기)은 CPU로 작게 구현하고 학습할 수 있습니다.** 매퍼는 MLP 5개(입력 3차원)이고 공식 코드 기본값의 학습 스텝은 약 7,320개입니다(실행 시간은 측정하지 않은 추정).
- **논문 수치(표 1–4)와 실로봇 결과는 공개 자료만으로 재현할 수 없습니다.** 평가 코드, 사람 글러브 데이터가 공개되지 않았고 GMC·LMC의 정의식은 논문에 없습니다. 실로봇은 Flexiv Rizon 4, Wuji 손, Manus 글러브와 작업자 8명이 필요합니다.
- **공식 코드 실행은 조건부입니다.** 학습이 CUDA를 요구하고, 저장소에 라이선스 파일이 없습니다. 이 분석에서는 코드를 실행하거나 설치하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, v1 ↔ v2 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, 공개 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V, 일관성 점검 K, 토이, 주장 판정 C |
| `analysis/04_code_verification.md` | 논문 ↔ 공식 코드 대조 D1–D14, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_anchor_symmetry.py` | 토이: 앵커와 부분 Chamfer (torch, 약 2분) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_anchor_symmetry.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(nonexclusive-distrib/1.0)로 공개되어 있어 CC BY가 아닙니다. 논문 그림은 넣지 않았고 그래프는 표 값과 재계산으로 우리가 그렸습니다.
- 공식 코드는 `github.com/chenxi-wang/anydexrt`에 있고 라이선스 파일이 없습니다. 이 저장소에는 코드를 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

# Cosmos-Predict 2.5 (arXiv:2511.00062) 분석 노트

*World Simulation with Video Foundation Models for Physical AI* (NVIDIA, 기술 보고서, 학회 채택 표기 없음)를 표 재계산, 공식 코드 정적 대조, 일관성 검사로 분석한 자료입니다. 분석 대상은 v2 (2026-02-24)입니다. 논문 원문과 논문 그림은 넣지 않았습니다(PDF에 NVIDIA 저작권 표시). 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서는 재현할 수 없습니다.** 공식 코드는 Linux x86-64, Ampere 이상 NVIDIA GPU, CUDA 12.8.1을 요구합니다.
- **정적 대조와 표 재계산은 완료했습니다.** 실로봇(표 13) 재계산, 사람 평가의 N 민감도, 벤치마크 합산 검사, 주장 판정 C1–C28을 포함합니다.
- **두 공식 저장소는 Cosmos 3 안내가 붙은 유지보수 모드입니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, v1–v2 차이, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, 공식 저장소 표 |
| `analysis/03_analysis.md` | 표 13 재계산과 부호 검정, 사람 평가 N 민감도, 벤치마크 합산, 주장 판정 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조 (D1–D22), 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 원문 받기

```bash
curl -L -o paper/paper.pdf https://arxiv.org/pdf/2511.00062v2
```

## 출처와 라이선스

- 논문 라이선스: arXiv 초록 페이지의 라이선스 링크는 CC BY 4.0입니다. PDF 첫 쪽에는 "© 2026 NVIDIA. All rights reserved."가 있어서 논문 그림은 쓰지 않았습니다.
- 공식 코드(https://github.com/nvidia-cosmos/cosmos-predict2.5, https://github.com/nvidia-cosmos/cosmos-transfer2.5)는 Apache-2.0입니다. 실행하거나 복사하지 않고 정적으로만 읽었습니다. 가중치는 NVIDIA Open Model License이며 받지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.

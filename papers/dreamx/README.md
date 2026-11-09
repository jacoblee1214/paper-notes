# DreamX (arXiv:2611.05377) 분석 노트

*DreamX: Video Diffusion Model is a Universal Latent Action Learner*
(Jisoo Kim, Chaehyun Kim, Junsu Kim, Jiho Park, Seongchan Kim, Eunbeen Hong, Soowon Son, Fu-En Yang, Seungryong Kim, KAIST · NVIDIA, 2026)을
논문 표 재계산, 통계 검정, 사람·로봇 latent 정렬에 관한 선형 toy로 분석한 자료입니다.
공식 코드는 아직 공개되지 않아(2026-11-09 확인, [cvlab-kaist/DreamX](https://github.com/cvlab-kaist/DreamX) @ `056bd46`은 README만) 코드 대조는 하지 않았습니다.
학습·추론 재현(GPU)은 하지 않았고, 가중치와 데이터셋은 받지 않았습니다.

**구현 가능성:** 공식 재현은 불가능합니다(코드·가중치 미공개, 사전학습 연산량 미보고). 공개 백본(Cosmos-Predict2.5 2B)에 query와 LoRA를 붙이는
구조라 재구현은 중간 난이도이지만, 행동 branch 구조와 행동 공간 통일 방식이 논문에 없고 CUDA GPU 여러 장이 필요합니다.
24 GiB 통합 메모리 맥에서는 학습이 불가능하고, 백본 단독 추론은 수정이 필요하며 검증하지 않았습니다(`analysis/04_claim_verification.md` §4).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | LAM 계보, 구성 요소별 출처, 관련 공개 저장소 표 |
| `analysis/03_analysis.md` | 정렬 toy, LARYBench 읽기, "300×" 산술, 실물·CALVIN 통계, Table 8 읽기 |
| `analysis/04_claim_verification.md` | 주장 C1–C13 재계산·판정, 재현 정보 체크리스트, 구현 가능성 |
| `analysis/table_check.py` | 표·본문 수치 재계산, Wilson 구간, Fisher 검정 |
| `analysis/alignment_toy.py` | 재구성 손실만으로 사람·로봇 latent가 정렬되는지 보는 선형 toy |
| `analysis/plot_figures.py` | 리포트 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/alignment_toy.py
python3 analysis/plot_figures.py
```

모두 CPU 단일 스레드로 1분 안에 끝납니다. 논문 PDF와 가중치는 필요하지 않습니다(수치는 스크립트에 출처와 함께 적었습니다).

## 출처와 라이선스

- 논문은 **CC BY 4.0**입니다(arXiv abs 페이지 확인, PDF 안에 다른 권리 고지 없음). `report/figures/paper_fig*.png`는 논문에서 잘라낸 그림이며 © 저자, CC BY 4.0, 출처 arXiv:2611.05377입니다.
- 나머지 그래프는 논문 수치와 toy 결과로 직접 그렸습니다.
- 원문 PDF는 이 저장소에 넣지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

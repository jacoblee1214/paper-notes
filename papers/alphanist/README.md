# AlphaNIST (arXiv:2610.13354) 분석 노트

*AlphaNIST: Emergent Assembly Skills from Binary Task Rewards* (Zhang, Akinola, Scalise, Guaman Castro, Yin et al., NVIDIA · University of Washington, 2026)을
그림 픽셀 판독, 본문 수치 재계산, 구성 요소 원본(SGS arXiv:2610.12465, OmniReset 공개 구현 UW-Lab/UWLab `2b40afb`) 대조, 샘플러·보상 toy로 분석한 자료입니다.
공식 코드는 아직 공개되지 않았고, 학습 재현(GPU)은 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, 관련 GitHub 저장소 표 |
| `analysis/03_analysis.md` | 샘플링 커널(식 4 vs SGS), 성공 시 종료 + 보조 보상의 유인, reset 혼합 분포, 계산량 해부 |
| `analysis/04_claim_verification.md` | 주장별 재계산·대조(C1–C16), 빠진 설정값, 구현 가능성 |
| `analysis/fig_digitize.py` | 그림 5(a) heatmap·그림 6 곡선 픽셀 판독 |
| `analysis/table_check.py` | 본문·그림·Table I 수치 재계산 |
| `analysis/sampling_toy.py`, `reward_toy.py` | 샘플러·보상 구조 toy (numpy) |
| `analysis/plot_figures.py` | 리포트 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib pillow
python3 analysis/sampling_toy.py && python3 analysis/reward_toy.py && python3 analysis/table_check.py
cd analysis && python3 plot_figures.py
```

모두 CPU에서 수 초 안에 끝납니다. 그림 판독을 다시 하려면 원문 PDF(25 MB)가 필요합니다:
`mkdir -p paper && curl -L -o paper/paper.pdf https://arxiv.org/pdf/2610.13354v1 && python3 analysis/fig_digitize.py` (pdftoppm 필요).

## 출처와 라이선스

- 논문은 **CC BY 4.0**입니다(arXiv abs 페이지 확인). `report/figures/paper_fig*.png`는 논문에서 잘라낸 그림이며 © 저자, CC BY 4.0, 출처 arXiv:2610.13354입니다.
- `checkpoints.png`는 논문 그림 5·6을 판독한 값으로, `kernels.png`와 `reward_delay.png`는 논문 식 4·5·7과 SGS 식 1–2로 직접 그렸습니다.
- 원문 PDF와 다른 저장소의 코드는 넣지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

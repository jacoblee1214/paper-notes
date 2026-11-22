# MimicPlay (arXiv:2302.12422, CoRL 2023) 분석 노트

*MimicPlay: Long-Horizon Imitation Learning by Watching Human Play* (arXiv v2, 2023-10-13)을 논문 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2302.12422에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **실로봇 재현은 이 맥에서 불가능합니다.** Franka 제어 스택과 사람 play 수집 장치가 공개되지 않았습니다.
- **공식 학습은 CUDA를 전제합니다.** `.cuda()` 32곳이라 MPS 이식이 필요하고, 학습 설정도 논문과 다릅니다.
- **공식 시뮬 평가는 조건부입니다.** 평가 설정은 논문과 같습니다. 설치와 체크포인트 다운로드가 필요합니다.
- **축소판(잠재 계획, GMM 헤드, KL 정렬 효과)은 가능합니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 공개되어 있습니다. arXiv abs 페이지의 라이선스 링크는 arXiv 비독점 배포 라이선스(nonexclusive-distrib/1.0)이며 CC BY가 아닙니다. 그래서 논문 그림과 원문 PDF는 넣지 않았습니다.
- 공식 코드는 `github.com/j96w/MimicPlay`에 있습니다(MIT). 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 체크포인트와 데이터셋은 받지 않았습니다.

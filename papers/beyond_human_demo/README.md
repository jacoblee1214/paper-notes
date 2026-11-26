# GLIDE (arXiv:2609.24996, CoRL 2026 포스터) 분석 노트

*Learning Beyond What Humans Can Demonstrate* (Song, Mittal, Jain, arXiv v1, 2026-09-21)를 표 재계산, 정확한 이항 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2609.24996에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **실로봇 재현(표 1, 2)은 이 맥(Apple M4 Pro, CUDA 없음)에서 불가입니다.** YAM 양팔, Meta Quest, RealSense 3대, CRAFT 손이 필요합니다.
- **π0.5 미세조정은 GPU 서버가 필요합니다.** 저장소에 학습 설정이 없습니다.
- **가드 필터의 구현과 가상 입력 시험은 CPU에서 가능합니다.** 일부 상수 산술은 `analysis/toy_selection_and_filter.py`로 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 판본, 학회, 라이선스 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 선택 편향 토이, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 구현 가능성 |
| `analysis/recompute.py` | 표 1·2 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_selection_and_filter.py` | 최댓값 선택 편향 토이와 코드 상수 산술 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_selection_and_filter.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 논문 그림은 실로봇 사진과 사람이 나온 사진을 포함해 넣지 않았습니다.
- 공식 코드는 `github.com/Yuchen-Song/glide_code`(Apache-2.0, 커밋 `8ebb33a6dca12a2091facccdc45a1a6ffbb50e66`)에 있고, 데이터는 Hugging Face `yuchensong/glide_data`(CC BY 4.0)에 있습니다. 이 저장소에는 코드와 데이터를 넣지 않았고, 코드는 실행하거나 설치하지 않았으며 데이터는 받지 않았습니다(메타데이터만 읽음).
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

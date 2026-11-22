# EgoWAM (arXiv:2607.08436, CoRL 2026 포스터) 분석 노트

*EgoWAM: World Action Models Beyond Pixels with In-the-Wild Egocentric Human Data* (arXiv v2, 2026-10-13)를 표 재계산, 통계 검정, 공식 코드 정적 대조, 작은 기하 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2607.08436에서 받을 수 있습니다). 논문 그림은 두 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **실로봇 평가와 공식 학습은 이 맥(CUDA 없음)에서 불가능합니다.** ARX5 양팔, Aria 안경, 데이터 접근 계정, GPU 2–3일이 필요합니다.
- **3D 흐름 목표 재생성은 불가능합니다.** 추출 스크립트가 부르는 모듈이 릴리스에 없습니다.
- **표 재계산, 정적 코드 대조, 일인칭 3D 흐름 안정화 토이는 가능하고 끝냈습니다.** `analysis/recompute.py`(수 초), `analysis/toy_egomotion.py`(약 1초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 판본 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표·그림 수치 재계산과 검정 (numpy, scipy) |
| `analysis/toy_egomotion.py` | 토이: 머리 운동과 3D 흐름 목표 (numpy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_egomotion.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 권리 고지는 찾지 못했습니다.
- 논문 그림 두 장(그림 3, 그림 9)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2607.08436"). 사진과 데이터셋 프레임이 든 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/GaTech-RL2/EgoWAM`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 그림 3의 막대 값은 프로젝트 페이지의 공개 데이터와 논문 그림 라벨로 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

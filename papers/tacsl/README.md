# TacSL (arXiv:2408.06506) 분석 노트

*TacSL: A Library for Visuotactile Sensor Simulation and Learning* (arXiv v2, 2025-03-08; IEEE Transactions on Robotics 41, 2025, DOI 10.1109/TRO.2025.3547267)을 보고 수치 재계산, 일관성 검사, 접촉 식 검산으로 읽은 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 라이선스가 비배타적 배포 허락입니다). 공식 코드는 넣지 않았고, 실행하거나 설치하지 않았습니다.

## 구현 가능성 (요약)

- **공식 코드(Isaac Gym TacSL)와 Isaac Lab 센서 실행은 맥에서 불가능합니다.** 둘 다 CUDA와 NVIDIA GPU를 전제합니다. 공식 코드는 지원이 중단되었습니다.
- **재현하지 않았습니다.** 논문 표 수치를 다시 계산하고 검정했습니다.
- **CPU에서 할 수 있는 검산은 했습니다.** 접촉 식 검산(`analysis/toy_implicit_contact.py`, 수 초)과 재계산(`analysis/recompute.py`, 수 초)입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 수치, arXiv 버전 차이, 학회지 확인 |
| `analysis/02_prior_work.md` | 선행연구 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산과 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 I–IX 재계산과 검정 (scipy, 수 초) |
| `analysis/toy_implicit_contact.py` | 접촉 식 대수 검산 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (논문 수치로 그림) |
| `report/figures/` | 자체 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_implicit_contact.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문 라이선스: arXiv 페이지의 "view license" 링크는 `arxiv.org/licenses/nonexclusive-distrib/1.0/`입니다(비배타적 배포 허락, CC BY 아님). 그래서 논문 그림은 싣지 않았습니다.
- 학회지 확인: Crossref에서 DOI 10.1109/TRO.2025.3547267의 제목, 저널명, 권, 쪽, 연도를 확인했습니다. 저널판 본문은 열람하지 못했으므로 arXiv v2 기준으로 분석했습니다.
- 공식 코드의 `tacsl` 브랜치(`isaac-sim/IsaacGymEnvs`)는 BSD 3 라이선스(루트)와 독점 표기 파일(센서 모듈)이 함께 있습니다. 이 저장소는 코드를 복사하지 않았습니다.
- 그래프, 검산 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다. 표 값은 인용과 재계산을 위해 옮겼습니다.

# MimicGen (arXiv:2310.17596, CoRL 2023) 분석 노트

*MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations* (arXiv v1, 2023-10-26)를 표 재계산, 공식 코드 정적 대조, 작은 numpy 검사로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 네 장은 우리가 표 값으로 그린 것입니다.

## 구현 가능성 (요약)

- **논문 표 전체의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 학습은 GPU를 전제하고, Factory 과제와 Mobile Kitchen은 공개 코드가 없으며, 실로봇은 코드가 없습니다.
- **로봇 시뮬레이션 과제 하나의 생성 파이프라인 확인은 조건부로 가능합니다.** 설치와 렌더링 확인이 필요하고, 시간은 측정 전입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 부록 요약, 버전 확인 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(실로봇 구간, 시뮬 ± 문제, 로봇 전이), 표·본문 일치 검사, 토이 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 차이 목록, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 값 옮김, 일치 검사, 검정, 구간 (numpy, scipy, 1초 미만) |
| `analysis/recompute_results.json` | 위 스크립트의 결과 |
| `analysis/toy_transform_check.py` | 변환 식과 코드 연산의 수치 검사 (numpy, 1초 미만) |
| `analysis/toy_transform_check_results.json` | 위 스크립트의 결과 |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_transform_check.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/NVlabs/mimicgen)는 NVIDIA Source Code License(비상업 연구·평가용)입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 데이터셋(CC-BY 4.0, README 기준)은 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 검사 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.

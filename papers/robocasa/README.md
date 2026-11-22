# RoboCasa (arXiv:2406.02523, RSS 2024) 분석 노트

*RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots* (arXiv v1, 2024-06-04)를 표 재계산, 공식 코드 정적 대조, 일관성 점검으로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 CC BY 4.0이지만 그림 대부분이 제3자 자산·화면입니다). 이 폴더의 그래프 세 장은 우리가 표 값으로 그린 것입니다.

## 구현 가능성 (요약)

- **논문의 주요 학습 실험은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 재현할 수 없습니다.** BC 학습은 GPU를 전제로 하고, 복합 과제 미세 조정과 실로봇 실험은 코드와 하드웨어가 필요합니다.
- **원자 과제 한 개의 생성 성공률 측정은 조건부로 가능합니다.** 설치와 약 10 GB 에셋이 필요하고 시간은 측정 전입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계, 버전 확인 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 과제 평균 재계산, 검정, 실로봇 근사, 본문·표 일관성 점검, 주장 판정 |
| `analysis/04_code_verification.md` | 논문과 공식 코드(2024 스냅샷과 HEAD)의 대조, 차이 목록, 구현 가능성 |
| `analysis/recompute.py` | 과제 값 옮김, 평균·구간·검정·부트스트랩 (numpy, scipy, 1초 미만) |
| `analysis/recompute_results.json` | 위 스크립트의 결과 |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/robocasa/robocasa)는 LICENSE 파일상 MIT입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 데이터셋과 에셋은 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 검사 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.

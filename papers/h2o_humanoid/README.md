# H2O: Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation (arXiv:2403.04436, IROS 2024) 분석 노트

*H2O* (arXiv v1, 2024-03-07)를 표 재계산과 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치와 실물 시연의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 공식 코드는 Isaac Gym Preview 4(Linux, NVIDIA GPU)를 전제하고, 평가 스크립트와 필터링 스크립트가 없습니다. 실물에는 Unitree H1과 MoCap이 필요합니다.
- **리타게팅 스크립트는 조건부입니다.** AMASS와 SMPL 다운로드, 새 설치, 실행 승인이 필요합니다.
- **표 재계산과 보상 커널 비교는 가능하고 끝냈습니다.** `analysis/recompute.py`(수 초). 소형 MuJoCo 축소판은 새 설치 없이 시작할 수 있고 노력은 3–5주(추정, 측정 전)입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전과 학회, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산 V1–V10, 일관성 검사 K1–K12, 주장 판정 C1–C9 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조 D1–D17, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산, 보상 커널 토이, 시드 민감도 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-23). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/LeCAR-Lab/human2humanoid)는 README와 LICENSE 파일 기준 CC BY-NC 4.0(상업 이용 금지)입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

# Refined Policy Distillation (arXiv:2503.05833, IROS 2026 채택) 분석 노트

*Refined Policy Distillation: From VLA Generalists to RL Experts*를 논문 곡선 판독, 유도, 공식 코드 정적 대조, numpy 토이로 분석한 자료입니다. 논문 원문과 그림은 넣지 않았습니다(라이선스).

## 구현 가능성 (요약)

- **공식 재현:** NVIDIA GPU가 있어야 합니다. 시뮬이 GPU로 고정되어 있고, 교사 VLA 서버(`agents` 2025 상태)의 버전 고정이 필요합니다. OpenVLA 교사는 다중 GPU가 필요합니다.
- **이 분석:** 정적 읽기와 CPU 토이만 했습니다. 재현은 하지 않았습니다.
- **손실 항만 옮기기:** 쉽습니다. 교사 서버와 GPU 시뮬이 실제 노력입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 곡선 판독, v1 대 v2 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표, 관련 저장소 |
| `analysis/03_analysis.md` | 곡선 판독 요약, 검정력, 교사 항 기울기와 고정점 편향 유도, 토이 결과 |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조 (D 항목), 구현 가능성 |
| `analysis/toy_teacher_pull.py` | 교사 항 토이: 편향, 스케줄, 무관 교사 대조 (numpy, 약 3초) |
| `analysis/fig_reading.json` | 논문 Fig. 3, 4, 5, 7의 육안 판독값 (±10%) |
| `analysis/reading_stats.py` | 판독값에서 근사 Welch 검정과 검정력 최소 차이 계산 |
| `analysis/plot_figures.py` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
OMP_NUM_THREADS=1 python3 analysis/toy_teacher_pull.py
OMP_NUM_THREADS=1 python3 analysis/reading_stats.py
OMP_NUM_THREADS=1 python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC 라이선스가 아닙니다(arXiv 비독점 배포). 이 저장소에는 논문 그림과 원문 PDF를 넣지 않았습니다. 원문은 arXiv:2503.05833에서 받을 수 있습니다.
- 공식 코드 Refined-Policy-Distillation/RPD는 Apache-2.0입니다. 코드는 옮기지 않았고 파일과 줄 번호로만 인용했습니다. 실행하지 않았습니다.
- 교사 서버 RobotControlStack/vlagents(구 juelg/agents)는 Apache-2.0입니다. 소스는 옮기지 않았습니다.
- 그래프와 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다. 토이의 파라미터는 설계 가정이며 논문 값에 맞추지 않았습니다.

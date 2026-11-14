# FACTR 2 (arXiv:2606.12406, CoRL 2026) 분석 노트

*FACTR 2: Learning External Force Sensing for Commodity Robot Arms Improves Policy Learning* (CoRL 2026 구두, arXiv v2)을 표 재계산, 정밀도 점검, 검정, 공식 NEXT 코드의 정적 대조, 소규모 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(arXiv non-exclusive 배포 조건이며 CC BY가 아닙니다).

## 구현 가능성 (요약)

- **NEXT 추정기:** 공식 구현(Apache-2.0)이 있고, 학습과 추론은 CPU로 됩니다(추론 약 0.8 ms, 무작위 가중치 측정). 저가 팔에서 자유 운동 10분 데이터가 있어야 합니다.
- **FIRST와 정책 학습:** 공식 코드가 없습니다. 임계값(T_high, T_low, F)이 논문에 없어서 같은 라벨을 만들 수 없습니다. 정책 학습은 과제당 GPU 12–24시간이 필요합니다.
- **실로봇 실험(표 1 Franka 비교, 사용자 연구, Piper 피드백):** 실물 장비가 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계, v1과 v2 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, 관련 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, 검정, 정밀도 점검, 토이 결과 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 차이, 주장 판정, 구현 가능성 |
| `analysis/table_check.py` | 표 재계산과 검정 (scipy, 약 2초) |
| `analysis/toy_residual_history.py` | 마찰 이력과 외부 토크 식별 토이 (numpy, 약 2초, 인수 W = 0, 0.5, 2) |
| `analysis/toy_upsampling.py` | 단계별 업샘플링 토이 (numpy, 약 10초) |
| `analysis/cpu_timing.py` | LSTM 추론 시간 측정 (torch, 약 30초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/toy_residual_history.py 0
python3 analysis/toy_residual_history.py 0.5
python3 analysis/toy_residual_history.py 2
python3 analysis/toy_upsampling.py
python3 analysis/plot_figures.py
```

`cpu_timing.py`는 torch가 필요합니다. 공식 코드(`factr2_next`)는 실행하지 않았습니다. 정적 읽기만 했습니다. 공식 코드를 보려면 `git clone --depth 1 https://github.com/philiphan0109/factr2_next.git`로 받습니다(분석 기준 HEAD `0df02c5`).

## 출처와 라이선스

- 논문은 arXiv non-exclusive 배포 조건입니다(arXiv abs 페이지 확인, 2026-11-14). 논문 그림과 원문 PDF는 이 저장소에 넣지 않았습니다. 원문은 arXiv:2606.12406에서 받을 수 있습니다.
- 논문의 표 수치는 인용과 재계산을 위해 옮겼습니다. 문장과 그림은 옮기지 않았습니다.
- 그래프와 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다. 토이의 파라미터는 설계 가정이며 논문 값에 맞추지 않았습니다.

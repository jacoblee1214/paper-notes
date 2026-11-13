# GoalDex (arXiv:2611.02171, CoRL 2026) 분석 노트

*GoalDex: Learning Image-Goal-Conditioned Policies for Dexterous In-Hand Manipulation* (CoRL 2026, arXiv v1)을 표 재계산, 통계 검정, 대칭 물체 토이, 허용오차 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(라이선스 CC BY-NC-ND 4.0).

## 구현 가능성 (요약)

- **공식 재현: 이 맥에서는 불가.** 공식 코드는 아직 없습니다("Code soon", 2026-11-12 확인). 학습에는 NVIDIA GPU가 있는 리눅스, Isaac Lab, 3D Gaussian Splatting 자산이 필요합니다.
- **실로봇 재현:** Allegro 16-DOF 손과 손목 카메라, 30 Hz 정책이 필요합니다.
- **재구현 노력:** 구조는 수백 줄입니다. 병목은 다물체 렌더링과 교사 학습입니다(추정).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, 관련 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, 검정, 검정력, 토이 결과 |
| `analysis/04_claim_verification.md` | 주장별 판정, 구현 가능성, 라이선스 |
| `analysis/table_check.py` | 표 재계산과 검정 (scipy, 약 2초) |
| `analysis/toy_symmetry.py` | 대칭 물체에서 목표 모호성 (numpy, 약 2초) |
| `analysis/toy_tolerance.py` | 허용오차와 예측 실패율 모형 (numpy, 약 6초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `analysis/extra_goal_alignment.py` | 손 모델(MuJoCo)에서 목표 정렬 손실만 바꾼 소규모 비교. 손 모델은 별도로 받아야 함(아래 명령) |
| `analysis/extra_goal_alignment_results.json` | 위 비교의 집계 결과(CS, 검정, 임베딩 진단) |

## 재실행

```bash
pip install numpy scipy matplotlib torch mujoco
python3 analysis/table_check.py
python3 analysis/toy_symmetry.py
python3 analysis/toy_tolerance.py
python3 analysis/plot_figures.py
```

### 손 모델 비교 (선택, CPU 약 20–25분)

손 모델은 저장소에 넣지 않았습니다. 이 폴더 안에서 다음 명령으로 받습니다(라이선스 BSD-3-Clause, commit 370c43965323c0944b433b202aac5058b5f907a9, 폴더 2.5 MB).

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git data/menagerie
git -C data/menagerie sparse-checkout set wonik_allegro
python3 analysis/extra_goal_alignment.py
```

## 출처와 라이선스

- 논문은 CC BY-NC-ND 4.0입니다(arXiv abs 페이지 확인). 논문 그림과 원문 PDF는 이 저장소에 넣지 않았습니다. 원문은 arXiv:2611.02171v1에서 받을 수 있습니다.
- 논문의 표 수치는 인용과 재계산을 위해 옮겼습니다. 문장과 그림은 옮기지 않았습니다.
- 그래프와 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다. 토이의 파라미터는 설계 가정이며 논문 값에 맞추지 않았습니다.

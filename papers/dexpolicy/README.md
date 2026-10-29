# DexPolicy (arXiv:2610.00360) 분석 노트

*DexPolicy: Scheduled Exploration for Trajectory-Guided Dexterous Manipulation*
(Wang, Qian, Li, Zhang et al., Peking University · AI2 Robotics, 2026)을 표 재계산, toy 검증,
[공식 코드](https://github.com/AIGeeksGroup/DexPolicy)(`98bf668`)와 기반 환경 ViViDex(`zerchen/vividex_mujoco`) 대조로 분석한 자료입니다.
학습 재현은 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, 관련 GitHub 표 |
| `analysis/03_analysis.md` | 최적 σ의 닫힌 꼴, 학습 σ의 Adam 이동 상한, GRPO advantage 대조, 잡음 스케줄 toy, 통계 |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조(D1–D11), 수치 재계산(C1–C8), 구현 가능성 |
| `analysis/table_check.py` | Table I–VIII과 본문 수치 재계산 |
| `analysis/exploration_toy.py` | 잡음 스케줄 toy (numpy PPO-clip, 문맥 밴딧) |
| `analysis/grpo_credit_toy.py` | 논문 Eq. 5와 공개 코드 GRPO advantage 대조 |
| `analysis/plot_figures.py` | 리포트 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/exploration_toy.py && python3 analysis/grpo_credit_toy.py
cd analysis && python3 plot_figures.py
```

모두 CPU에서 각 20초 안에 끝납니다. 논문 PDF나 공식 코드는 필요하지 않습니다(수치는 스크립트에 출처와 함께 적었습니다).

## 출처와 라이선스

- 논문은 **CC BY-NC-SA 4.0**입니다(arXiv abs 페이지 확인). CC BY가 아니라서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- `report/figures/*.png`는 논문 수치, 공식 코드(MIT)의 스케줄 정의, 직접 만든 toy로 그렸습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

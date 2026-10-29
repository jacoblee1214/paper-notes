# ThinkWorld (arXiv:2610.37666) 분석 노트

*ThinkWorld: World Imagination Reinforces Chain-of-Dynamics Reasoning*
(Kim, Huang, Kim, Chen, Wang, Kim, Yang, NVIDIA · KAIST, 2026)을 논문 표 재계산, 공개 리더보드(LIBERO-PRO, LIBERO-Plus) 대조,
부록 보상 식의 toy 구현으로 분석한 자료입니다. 코드와 가중치가 공개되지 않아 코드 대조와 학습 재현은 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, GitHub 정리 |
| `analysis/03_analysis.md` | 배포 구조와 Table 3 해부, Eq. 6 보상의 신호 폭, 형식 오류 벌칙과 그룹 표준화, 실물·시뮬레이션 통계 |
| `analysis/04_claim_verification.md` | 주장별 재계산(C1–C14), 공개 자료와 다른 점, 구현 가능성 |
| `analysis/table_check.py` | 표·그림·리더보드 재계산 |
| `analysis/reward_toy.py` | Eq. 6 보상과 GRPO 표준화 toy |
| `analysis/plot_figures.py` | 리포트 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/reward_toy.py && python3 analysis/plot_figures.py
```

모두 CPU에서 몇 초 안에 끝납니다. 논문 PDF는 필요하지 않습니다(수치는 출처와 함께 스크립트에 적었습니다).

## 출처와 라이선스

- 논문은 arXiv abs 페이지 기준 **CC BY 4.0**이지만, PDF 양식에 "© 2026 NVIDIA. All rights reserved." 문구가 함께 있습니다. 권리 표기가 충돌하므로 **논문 그림은 이 저장소에 넣지 않았습니다.** 그림은 [프로젝트 페이지](https://thinkworld-project.github.io)에서 볼 수 있습니다.
- `real_robot_ci.png`, `component_gain.png`는 논문 표의 수치로, `reward_weights.png`는 부록 A.6의 보상 식으로 직접 그렸습니다.
- 리더보드 대조에 쓴 값은 `Zxy-MLlab/LIBERO-PRO`(`eafdb80`), `sylvestf/LIBERO-plus`(`4976dc3`) README에서 옮겼습니다.
- 원문 PDF는 이 저장소에 넣지 않았습니다. 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

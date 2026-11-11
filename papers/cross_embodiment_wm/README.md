# Cross-Embodiment World Models (arXiv:2511.01177) 분석 노트

*Scaling Cross-Embodiment World Models for Dexterous Manipulation* (IROS 2026 채택)을 표 재계산과 장난감 실험으로 분석한 자료입니다.
공식 코드는 아직 공개되지 않았습니다(2026-11-10 확인).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림·표 제외) |
| `analysis/01_paper_analysis.md` | 방법, 실험, 한계, v1 → v3 차이 |
| `analysis/02_prior_work.md` | 선행연구, 가까운 대안, GitHub 확인 |
| `analysis/03_analysis.md` | Table I 재계산과 유의성, 유도, 그림 판독, 장난감 결과 |
| `analysis/04_claim_verification.md` | 주장별 판정, 빠진 정보, 구현 가능성, 권리 |
| `analysis/table_check.py` | Table I 재계산 (수 초) |
| `analysis/particle_toy.py` | 2차원 장난감 (numpy 1스레드, 약 2분 30초) |
| `analysis/extra_particle_3d.py` | MuJoCo 손 3종의 입자 밀도와 합·평균 집계 검사 (torch, mujoco 필요, CPU 약 2.5분) |
| `analysis/plot_figures.py` | 직접 만든 그래프 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/particle_toy.py
python3 analysis/extra_particle_3d.py  # torch, mujoco 필요
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY-NC-SA 4.0입니다. 이 저장소에는 **논문 그림, 논문 표 이미지, 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 표의 수치와 직접 만든 장난감 실험으로 그렸습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

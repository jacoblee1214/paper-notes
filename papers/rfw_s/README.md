# RFW-S: Everyday Deformable Objects with Independent Simulation Verification (arXiv:2611.23056 v2) 분석 노트

*RFW-S* (arXiv v2, 2026-11-18)를 공개 파생 CSV로 표 재계산하고, 공식 코드를 정적으로 대조한 분석입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 전체 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 100K 자산 생성에 GPU와 생성 모델이 필요하고, 물리 사전은 유료 API를 씁니다. 실로봇과 사람 평가도 필요합니다.
- **공식 수치 재계산은 공개 CSV로 완료했습니다**(57개 검사 중 56개 일치, 나머지는 반올림).
- **축소판(보정과 검증을 분리하는 MuJoCo 실험)은 CPU로 시작할 수 있습니다**(1–2주 추정, 논문 수치 재현은 아님).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 버전 비교 |
| `analysis/02_prior_work.md` | 선행 연구 계보, 구성 요소별 출처, GitHub와 데이터 자료 표 |
| `analysis/03_analysis.md` | 표 재계산과 일관성 검사, 재현성 점검, 파지 쌍별 검정 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 불일치와 빈칸, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 공개 파생 CSV로 표 재계산 (numpy, scipy). 저장소 `repo/paper/analysis_data/`를 읽음 |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

공식 저장소의 파생 CSV가 필요합니다(`repo/paper/analysis_data/`, 저장소를 얕게 clone). 시뮬레이터는 실행하지 않습니다.

```bash
pip install numpy scipy matplotlib
python3 -I analysis/recompute.py
python3 -I analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/RoboFlywheel-com/RoboFlywheel-Soft)는 Apache-2.0입니다. 제3자 가중치와 데이터는 각자의 라이선스를 따릅니다. 이 분석에서는 코드를 실행하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

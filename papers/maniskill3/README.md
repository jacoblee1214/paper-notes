# ManiSkill3: GPU Parallelized Robotics Simulation and Rendering (arXiv:2410.00425, RSS 2025) 분석 노트

*ManiSkill3* (arXiv v2, 2025-05-30)의 보고 수치를 표와 본문만으로 다시 계산하고, 공식 저장소(mani-skill/ManiSkill, Apache-2.0)를 정적으로 읽어 대조한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다. 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 할 수 없습니다.** 처리량, 메모리 비교, PPO 학습 시간, sim2real, SIMPLER 상관은 CUDA GPU와 실물 로봇을 씁니다.
- **재계산과 검사는 가능하고 끝냈습니다.** 실험 표의 신뢰구간과 검정, 실물 성공률의 신뢰구간, 상관의 점 개수 가정, 메모리 비율 일관성, 리플레이 버퍼 산수.
- **설치 없이 할 수 있는 것:** 설정 대조와 소형 CPU 토이.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 주장, 실험 수치, 주장과 증거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 학회 확인, GitHub·프로젝트 자료 표 |
| `analysis/03_analysis.md` | 수치 재계산(신뢰구간, 검정, 상관, 메모리 비율, 버퍼 산수) |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 정적 대조, 불일치, 질문 목록 |
| `analysis/recompute.py` | 재계산과 일관성 검사 (scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에서 CC BY 4.0으로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다. 그림에 제3자 장면과 실물 사진이 있기 때문입니다.
- 공식 코드(https://github.com/mani-skill/ManiSkill)는 Apache-2.0입니다. 이 분석에서는 클론해서 정적으로만 읽었고 설치하거나 실행하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

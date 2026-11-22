# Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data (arXiv:2608.02580, CoRL 2026) 분석 노트

*Ego2Robot* (arXiv v1, 2026-08-03; CoRL 2026 스포트라이트)를 표 재계산, 공식 코드 정적 대조, 작은 numpy 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 네 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치(표 1, 2, 5, 그림 3, 4)의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 8 x A100 사전학습이 필요하고, 학습 코드·체크포인트·합성 데이터가 공개되지 않았고, 실로봇이 필요합니다.
- **합성 파이프라인의 순수 기하·신호처리(행동 정렬, 필터, 베이스 후보 점수)는 CPU에서 구현·검증할 수 있습니다.** IK와 MuJoCo 렌더링은 설치 뒤 조건부이고, SAM 3, ProPainter, DA3, WiLoR, RoboTwin(CuRobo)은 CUDA가 필요합니다. 노력은 추정이며 측정하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 버전 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(V1–V15), 이항·쌍체 검정, 노출 비율, 절제와 실로봇 점검, 토이 K1–K3 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조(일치 항목, 차이 D1–D16), 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 1초대) |
| `analysis/toy_camera_frame.py` | 토이: 식 (10)의 카메라 프레임 행동 (numpy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py          # 표 5는 paper/paper.txt 가 필요합니다(원문 PDF에서 pdftotext -layout)
python3 analysis/toy_camera_frame.py
python3 analysis/plot_figures.py
```

`recompute.py`는 원문 텍스트(`paper/paper.txt`)를 읽습니다. 논문은 `https://arxiv.org/abs/2608.02580` 에서 받아 `pdftotext -layout`으로 변환하면 됩니다. 이 폴더에는 원문을 넣지 않았으므로, 결과 `recompute_results.json`을 그대로 참고해도 됩니다.

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/www-Ye/ego2robot_blog, 커밋 244a4ee)에는 최상위 라이선스 파일이 없습니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

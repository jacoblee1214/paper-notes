# ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining (arXiv:2606.17200, CoRL 2026) 분석 노트

*ACE-Ego-0* (arXiv v1, 2026-06-15)를 표 재계산, 공식 코드 정적 대조, 작은 numpy 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 네 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 공식 평가는 CUDA와 FlashAttention을 가정하고, 체크포인트는 약 12.3 GB, SFT 데이터는 약 53.4 GB, SFT는 GPU 8장 규모입니다. 사전학습과 사람 영상 파이프라인은 코드가 공개되어 있지 않습니다.
- **소형 대리 실험(보조 손실, 카메라 좌표 행동, 청킹 샘플러)은 가능합니다.** 노력은 1–2주로 추정합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산 V1–V12, 불일치 후보 K1–K8, 볼록 껍질 토이 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조 D1–D14, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 수 초) |
| `analysis/toy_hull.py` | 볼록 껍질 면적비 토이 (numpy, 약 15초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_hull.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-26). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/ACERobotics-VLA/ACE-Ego-0)는 MIT 라이선스입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

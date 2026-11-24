# Box Trajectories as Compact State Representations for World Action Models (arXiv:2610.17526, CoRL 2026) 분석 노트

*BoxWAM* (arXiv v1, 2026-10-13, 유일한 판본)을 표 재계산, 공식 코드 정적 대조로 분석한 자료입니다. 실험은 재현하지 않았습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** RoboTwin 포크가 CUDA 컴파일(CuRobo, pytorch3d)을 요구하고, π0.5 설정은 JAX, 3.35B 파라미터, 체크포인트 12.4 GB, 데이터 약 120 GB가 필요하며, 실로봇은 하드웨어와 정답 상자 파이프라인이 필요합니다.
- **방법의 핵심(130M 예측기 + 선형 헤드 + 한 방향 어텐션)을 작게 다시 구현하는 것은 가능합니다.** 합성 환경에서 상자 보조 예측과 어텐션 설정을 비교할 수 있습니다(노력 추정 2–4일에서 1–2주).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 보고된 결과, 한계, 주장과 증거의 틈 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(평균, 정수 가능성, 쌍체·Fisher 검정, 검출 한계) |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조(D1–D17), 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 약 2초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |
| `analysis/bench_mps.py` | 랜덤 가중치 ViT-L 인코더와 예측기의 한 스텝 시간 측정 (torch, MPS가 있으면 1분 안팎) |
| `report/crop_figures.py` | 논문 PDF에서 도식·차트 패널 세 장 자르기 (pymupdf, Pillow) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
pip install torch                       # 처리량 측정에만 필요
python3 analysis/bench_mps.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0 라이선스로 공개되어 있습니다(abs 페이지 라이선스 링크 `creativecommons.org/licenses/by/4.0/`, 2026-11-24 확인). PDF에서 저작권 표시는 찾지 못했습니다. 이 폴더의 `report/figures/paper_fig*.png` 세 장은 논문 그림 1, 2, 6에서 도식·차트 패널만 잘라 낸 것입니다(© Qinxi Yu 외, CC BY 4.0, 출처 arXiv:2610.17526). 실로봇 사진, 시뮬레이션 화면, 데이터셋 프레임은 넣지 않았습니다. `own_*.png`는 이 저장소 작성자가 표 값으로 그린 그래프입니다.
- 공식 코드(https://github.com/quincy-u/box-wam, MIT)는 정적으로만 읽었고 이 폴더에 넣지 않았습니다. 프로젝트 페이지는 https://box-wam.github.io 입니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

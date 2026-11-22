# LAPT-Scale: Scaling Embodied Policies with Latent Action Pre-Training on Human Videos (arXiv:2610.13732, CoRL 2026) 분석 노트

*LAPT-Scale* (arXiv v1, 2026-10-09)을 보고 수치 재계산, 검정, 작은 CPU 토이로 분석한 자료입니다. 공식 코드가 없어서 코드 대조 대신 주장 검증(C1–C15)을 썼습니다. 논문 원문은 넣지 않았고, 논문 그림은 그림 4 한 장만 출처를 적어 넣었습니다. 이 폴더의 나머지 그래프 네 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 결과의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 한 번의 전체 파이프라인이 약 9.6K H200 GPU시간이고, 정책 기반 모델, 250시간 로봇 데이터, 자체 수집 영상이 비공개이며 코드가 없습니다.
- **LAM의 구조와 손실은 구현할 수 있습니다.** PyTorch 400–600줄, 3–5일(추정). 소형 규모의 학습과 합성 영상 검증은 가능하고, LPIPS·DINOv2·RAFT 가중치가 필요한 절제는 다운로드 뒤의 조건부입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(V1–V5), 합성 토이(K1–K2) |
| `analysis/04_claim_verification.md` | 주장 검증(C1–C15), 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산, k/n 가능성, Fisher·Wilson, 추세, 검정력 (numpy, scipy, 약 10초) |
| `analysis/toy_psnr_motion.py` | 합성 영상의 PSNR 둔감성, 코드 수 대 변위 분해능 (numpy, 약 5초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_psnr_motion.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0(`http://creativecommons.org/licenses/by/4.0/`)으로 공개되어 있습니다(abs 페이지 확인, 2026-11-23). `report/figures/paper_fig4.png`는 논문 그림 4를 잘라 쓴 것입니다. 그림: © Zhiyuan Xiang 외, CC BY 4.0, 출처 arXiv:2610.13732. 논문의 나머지 그림은 데이터셋 프레임과 실로봇 사진을 포함해 넣지 않았습니다.
- 공식 코드와 프로젝트 페이지는 찾지 못했습니다(2026-11-23).
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

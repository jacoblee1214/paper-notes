# HuMI: Humanoid Whole-Body Manipulation from Humanoid-Free Demonstrations (arXiv:2602.06643, CoRL 2026) 분석 노트

*HuMI* (arXiv v3, 2026-11-06)를 보고 수치 재계산, 검정, 공식 코드 정적 대조, 작은 CPU 토이로 분석한 자료입니다. 논문 원문과 공식 코드는 넣지 않았고, 논문 그림은 그림 4와 7(그래프) 두 장만 출처를 적어 넣었습니다. 나머지 그래프 다섯 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 결과의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 실로봇(Unitree G1, Vive 추적기)이 필요하고, 저수준 학습은 Isaac Lab + CUDA(환경 스텝 약 4.9e9), 고수준 학습은 CUDA·DDP·bf16을 전제합니다.
- **가능한 것:** 공개 소형 데이터로 하는 오프라인 IK 점검(설치·다운로드 뒤)과 핵심 메커니즘의 소형 MuJoCo 재구현.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이, 주장과 증거의 틈 |
| `analysis/02_prior_work.md` | 선행 계보, GitHub와 자료 표, 구성 요소별 출처 |
| `analysis/03_analysis.md` | 표 재계산(K1–K13), 토이(V1) |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조(D1–D13), 주장 대조(C1–C8), 구현 가능성 |
| `analysis/recompute.py` | 표 재계산, Fisher·Wilson·Newcombe, 검정력 모의 (numpy, scipy, 약 4초) |
| `analysis/toy_chunk_anchor.py` | 토이: 덩어리 기준 자세 (numpy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 다섯 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_chunk_anchor.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0(`http://creativecommons.org/licenses/by/4.0/`)으로 공개되어 있습니다(abs 페이지 확인, 2026-11-23). `report/figures/paper_fig4.png`, `paper_fig7.png`는 논문 그림 4와 7을 잘라 쓴 것입니다. 그림: © Ruiqian Nai 외, CC BY 4.0, 출처 arXiv:2602.06643. 논문의 나머지 그림은 로봇·카메라 사진을 포함해 넣지 않았습니다.
- 공식 코드: https://github.com/Richard-coder-Nai/HuMI (MIT, 커밋 37a85a696f7558710bf0bd4c5a1641d37265a3e6를 정적으로만 읽음). 프로젝트 페이지: https://humanoid-manipulation-interface.github.io/
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

# Contact-UAN (arXiv:2610.18991, CoRL 2026 포스터) 분석 노트

*Contact-UAN: Unsupervised Actuator Networks as a Residual World Model for Humanoid Control* (arXiv v1, 2026-10-13)을 그림 판독, 산술 재계산, 공식 코드의 정적 대조, 소프트 접촉 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다. 논문 그래프는 세 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **실로봇 재현(달리기, 힘 제어, 점프)은 이 맥에서 불가입니다.** Unitree G1, 동력계, 코스가 필요하고 하드웨어 데이터와 사전학습 정책이 공개되어 있지 않습니다.
- **공식 파이프라인 전체는 사실상 불가입니다.** 하드웨어 데이터가 없고, 하류 정책 학습은 CUDA GPU를 전제합니다.
- **정적 대조, 그림 판독과 재계산, 소프트 접촉 토이는 가능하고 끝냈습니다.** `analysis/toy_contact_sensitivity.py`(약 1초, mujoco), `analysis/recompute.py`(수 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그래프 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 코드 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/digitize_figures.py` | 논문 벡터 그림에서 값 읽기 (PyMuPDF, 원문 PDF 필요) |
| `analysis/recompute.py` | 표와 그림 수치 재계산, 검정 (numpy, scipy) |
| `analysis/toy_contact_sensitivity.py` | 토이: 소프트 접촉의 높이·기울기 감도 (mujoco) |
| `analysis/bench_g1_cpu.py` | G1 MJCF의 CPU 스텝 속도 (공식 저장소의 MJCF 필요) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/crop_figures.py` | 논문 그래프 세 장 자르기 (원문 PDF 필요) |
| `report/figures/` | 우리가 그린 그래프 네 장과 논문 그래프 세 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf pillow mujoco
curl -L -o paper/paper.pdf https://arxiv.org/pdf/2610.18991v1   # 원문 PDF (digitize_figures.py, crop_figures.py가 읽음)
python3 analysis/digitize_figures.py
python3 analysis/recompute.py
python3 analysis/toy_contact_sensitivity.py
python3 analysis/plot_figures.py
python3 report/crop_figures.py
```

`analysis/bench_g1_cpu.py`는 공식 저장소의 MJCF를 읽습니다: `git clone --depth 1 https://github.com/nolie-rolie/contact-uan repo && git -C repo checkout 55de136a33f44b7f22190c5f9de882266268b5ab`. 그 뒤 `python3 analysis/bench_g1_cpu.py`로 실행합니다.

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 license 링크 확인). 원문 PDF에서 저작권 고지를 찾지 못했습니다.
- 논문 그래프 세 장(그림 5, 7, 8)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2610.18991"). 사진과 시뮬레이션 화면이 있는 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/nolie-rolie/contact-uan`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 그 저장소의 `third-party/openhomie`는 CC BY-NC 4.0 고지가 있습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

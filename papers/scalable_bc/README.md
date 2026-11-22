# ABC (arXiv:2606.27375, CoRL 2026 포스터) 분석 노트

*Scalable Behavior Cloning with Open Data, Training, and Evaluation* (arXiv v1, 2026-06-25)을 표 재계산, 통계 검정, 공식 코드 정적 대조, CPU 토이로 분석한 자료입니다. 논문 원문 PDF와 공식 코드는 넣지 않았습니다(arXiv:2606.27375, github.com/amazon-far/abc에서 받을 수 있습니다). 논문 그림은 쓰지 않았고 우리가 그린 그래프만 넣었습니다.

## 구현 가능성 (요약)

- **사전학습(DiT 200k 스텝, VLA 전체 학습)과 실로봇 재현은 개인 장비로 불가능합니다.** H200 수천 GPU 시간, 1 TB 이상 데이터, 24 GiB를 넘는 미세조정 메모리가 필요합니다.
- **공개 DiT 체크포인트(8.1 GB)의 시뮬 평가는 조건부입니다.** 새 환경, 다운로드, 공식 코드 실행이 필요하고 속도는 측정 전입니다.
- **구조 재구현과 다중 draw 토이는 CPU에서 가능합니다.** `analysis/toy_multidraw.py`(약 10초, torch)로 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V16, 토이, 주장 판정 C1–C17 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조 D1–D18, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정, 검정력 (numpy, scipy, 수 초) |
| `analysis/toy_multidraw.py` | 토이: 다중 diffusion draw의 기울기 분산 (torch CPU, 약 10초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_multidraw.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF의 저작권 표시는 찾지 못했습니다. 이 노트에는 논문 그림을 넣지 않았습니다.
- 공식 코드는 `github.com/amazon-far/abc`(Apache-2.0)에 있습니다. 이 노트는 코드를 정적으로만 읽었고 실행, 설치하지 않았으며 코드를 포함하지 않습니다. DINOv3와 Gemma 가중치는 각각 별도 약관을 따릅니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 노트 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

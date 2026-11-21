# GRAIL (arXiv:2606.05160, CoRL 2026 구두) 분석 노트

*GRAIL: Generating Humanoid Loco-Manipulation from World Models* (arXiv v2, 2026-10-19)을 수치 재계산과 공식 코드 정적 읽기로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **이 맥(CUDA 없음)에서 논문 결과 재현은 불가합니다.** 공식 파이프라인은 Linux GPU 호스트와 NVIDIA Docker를 전제하고, 학습은 GPU 64장과 약 30시간이 필요합니다.
- **잔차 어댑터의 PyTorch 구조 검사는 가능합니다**(추정 2–4주, 성능 주장은 재현하지 않음).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계, 본문 확인 사항, v1→v2 차이, 질문 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, 가까운 대안, GitHub 표 |
| `analysis/03_analysis.md` | 수치 재계산, 구간, 분모 검사, 일관성 검사 |
| `analysis/04_code_verification.md` | 논문 대 공식 코드 대조(D1–D19), 구현 가능성 |
| `analysis/recompute.py` | 본문 수치 재계산과 검정 (scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포이고, PDF에 NVIDIA 저작권 표기가 있습니다. 논문 그림과 원문 PDF는 넣지 않았습니다. 원문은 arXiv:2606.05160에서 받을 수 있습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드(NVlabs/GRAIL, NVIDIA License, 비상업 연구·평가)는 정적으로만 읽었고 재배포하지 않았습니다. 데이터(CC BY-NC 4.0)와 체크포인트는 받지 않았습니다.

# T(R,O) Grasp++ (arXiv:2610.25410, CoRL 2026 포스터) 분석 노트

*T(R,O) Grasp++: Cross-Embodiment Dexterous Grasping in Cluttered Environments at Scale* (arXiv v1, 2026-10-18)을 표 재계산, 통계 검정, 공식 코드 정적 대조, 토이 두 개로 분석한 자료입니다. 논문 원문 PDF와 공식 코드는 넣지 않았습니다(arXiv:2610.25410과 GitHub에서 받을 수 있습니다). 논문 그림은 그림 8 한 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **시뮬 성공률, 학습, 실로봇 재현은 이 맥(Apple M4 Pro, MPS, CUDA 없음)에서 불가입니다.** Isaac Gym(NVIDIA GPU), 8 x A100급 학습과 6.19 TiB 데이터, 실로봇이 필요합니다.
- **공개 체크포인트의 추론과 충돌 통계는 조건부입니다.** 새 설치, 공식 코드 실행, 체크포인트와 검증 장면 다운로드가 필요하고 맥 호환은 확인하지 못했습니다.
- **표 재계산, 코드 정적 대조, 토이는 끝냈습니다.** `analysis/recompute.py`(수 초), `analysis/toy_checks.py`(약 7초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 8만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V15, 토이 T1–T2, 주장 판정 C1–C15 |
| `analysis/04_code_verification.md` | 공식 코드 대조(일치 K1–K10, 차이 D1–D17), 라이선스, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 표 재계산과 검정 (numpy, scipy; 그림 8 막대 읽기는 원문 PDF가 있을 때만 갱신) |
| `analysis/toy_checks.py`, `toy_checks_results.json` | 토이: rotvec 확산 성질, 환경 충돌 규칙 민감도 (numpy, scipy) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 8(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_checks.py
python3 analysis/plot_figures.py
```

`recompute.py`의 그림 8 막대 읽기(V11)는 `pymupdf`, `Pillow`와 원문 PDF(`paper/paper.pdf`)가 있어야 다시 계산하고, 없으면 저장된 결과를 유지합니다.

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 논문 그림 8은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2610.25410"). 시뮬 렌더와 실로봇 사진이 있는 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/Barrybarry-Smith/TROGraspV2`에 있습니다. LICENSE 파일이 없고 GitHub 라이선스 값도 비어 있습니다. 이 저장소에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다. 데이터(Hugging Face `Barrybarry-Smith/TROV2Dataset`)는 받지 않았고 데이터 카드가 라이선스 미선언이라고 밝힙니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

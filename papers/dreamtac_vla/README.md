# DreamTacVLA (arXiv:2512.23864, CoRL 2026 포스터) 분석 노트

*Learning to Feel the Future: DreamTacVLA for Contact-Rich Manipulation* (arXiv v5, 2026-10-27)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 공식 코드는 넣지 않았습니다(논문은 arXiv:2512.23864에서 받을 수 있습니다). 논문 그림은 그림 4 한 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **공식 결과의 재현(표 1 실로봇, 표 4 IsaacSim)은 이 맥(Apple M4 Pro, MPS, CUDA 없음)에서 불가입니다.** Dobot X-Trainer와 DIGIT 센서, IsaacSim과 cuRobo(NVIDIA GPU)가 필요하고 세계 모델 사전학습 코드는 공개 저장소에 없습니다.
- **공식 코드 학습은 사실상 어렵습니다.** `.cuda()`가 하드코딩되어 있고 V-JEPA 2 ViT-L 가중치가 공개 목록에 없습니다.
- **수치 재계산과 정적 코드 대조는 CPU에서 수 초에 끝납니다.** `analysis/recompute.py`.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림은 그림 4만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 공개 자료 표 |
| `analysis/03_analysis.md` | 재계산 V1–V12, 일관성 검사 K1–K17, 주장 판정 C1–C14 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D17, 라이선스, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 4(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 4는 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2512.23864"). 실로봇 사진과 촉각 영상 프레임이 들어 있는 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/michaelyeah7/learning-to-feel-the-future`에 있습니다(LICENSE 파일은 MIT와 Dobot 부분 MIT). 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 가중치와 데이터셋은 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

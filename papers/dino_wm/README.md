# DINO-WM (arXiv:2411.04983, ICML 2025) 분석 노트

*DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning* (arXiv v2, 2025-02-01)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2411.04983에서 받을 수 있습니다). 논문 그림은 한 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **전체 재현은 이 맥에서 불가능합니다.** 학습은 GPU와 SLURM을 전제하고, 인과 마스크가 CUDA로 고정되어 있습니다. Rope, Granular 시뮬레이션은 Docker와 CUDA를 씁니다.
- **공식 체크포인트로 PushT 계획을 확인하는 것은 조건부입니다.** 체크포인트 다운로드, 설치, CUDA 한 곳 수정이 필요합니다.
- **작은 PyTorch 축소판은 가능합니다.** 노력은 추정 1–2주입니다(측정 전).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 한 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, v1과 v2의 차이 |
| `analysis/02_prior_work.md` | 선행 계보, ICML 2025 근거, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (표준 라이브러리, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |
| `report/figures/` | 그래프 두 장(우리가 그림)과 논문 그림 2(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install matplotlib numpy
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 논문 그림 2는 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2411.04983 (v2)"). 그림 속 작은 프레임은 저자의 시뮬레이터 장면입니다.
- 공식 코드는 `github.com/gaoyuezhou/dino_wm`에 있습니다(MIT). 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.
- 체크포인트와 데이터는 받지 않았습니다.

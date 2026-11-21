# DreamerV3 (arXiv:2301.04104) 분석 노트

*Mastering Diverse Domains through World Models* (arXiv v2, 2024-04-17; 저널판 Nature 640, 647–653, 2025)을 공식 저장소의 실행별 점수로 다시 계산하고, 공식 코드를 정적으로 대조한 자료입니다. 재현 실험은 하지 않았습니다.

## 구현 가능성 (요약)

- **논문 전체 재현은 이 맥(Apple M4 Pro, CUDA 없음)에서 불가능합니다.** Atari, DMLab, ProcGen, Minecraft, DMC 학습은 GPU와 수 일에서 수 주 규모입니다.
- **표 재계산은 완료했습니다.** `analysis/recompute.py`(numpy, scipy, 수 초)가 공식 저장소 `scores/`를 읽습니다.
- **공식 코드(JAX)는 CUDA 버전으로 고정되어 있습니다.** CPU 실행과 소형 재구현은 설치 승인 후 가능하며, 아직 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림은 넣지 않음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, 공식 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, 집계 창 민감도, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 불일치 목록, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 재계산과 검정 결과 |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `paper/paper.txt` | 원문 PDF의 `pdftotext -layout` 결과 (PDF 자체는 넣지 않음) |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
git clone --depth 1 https://github.com/danijar/dreamerv3.git repo   # 점수 파일 scores/ 가 필요
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

`repo/`는 공식 저장소의 얕은 복제본입니다. 이 폴더에는 넣지 않았습니다.

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 "view license" 링크 확인). PDF에서 저작권 표시는 찾지 못했습니다. 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/danijar/dreamerv3`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프와 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.

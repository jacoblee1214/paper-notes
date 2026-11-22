# SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics (arXiv:2506.01844) 분석 노트

SmolVLA(arXiv v1, 2025-06-02)를 표 재계산, 공식 LeRobot 정적 대조, 통계 검정으로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프는 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **사전학습과 미세조정 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 논문 규모는 4 GPU, 200k 스텝, 약 30k GPU 시간입니다.
- **공개 가중치의 추론 지연과 메모리 측정은 조건부로 가능합니다.** 논문은 이 수치를 주지 않습니다.
- **토이 검증(흐름 매칭 규약, 어텐션 구조, 비동기 큐)은 CPU로 가능합니다.**

## 파일

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 어블레이션, 주장과 증거의 틈 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(V1–V8), 어블레이션 검정, 실로봇 민감도, 비동기 검정 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조(D1–D28), 구현 가능성, 질문 |
| `analysis/recompute.py` → `recompute_results.json` | 표 평균, 이항 검정, Welch 검정, 데이터셋 개수 (numpy, scipy) |
| `analysis/plot_figures.py` | 그래프 한 장 (matplotlib). 실행하려면 `report/figures/` 경로가 필요 |
| `report/figures/own_ablation_ci.png` | 어블레이션 신뢰구간 그래프 (우리가 만듦) |

`recompute.py`는 원문 텍스트(`paper/paper.txt`, `pdftotext -layout`으로 만든 것)를 읽습니다. 원문 PDF는 arXiv에서 받으십시오: https://arxiv.org/pdf/2506.01844

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/huggingface/lerobot)는 Apache-2.0입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

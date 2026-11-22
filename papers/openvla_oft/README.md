# OpenVLA-OFT (arXiv:2502.19645, RSS 2025) 분석 노트

*Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success* (arXiv v2, 2025-04-28)을 표 재계산, 통계 검정, 공식 코드의 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2502.19645에서 받을 수 있습니다). 논문 그림은 쓰지 않았고, 그래프 세 장은 표 값에서 다시 그렸습니다.

## 구현 가능성 (요약)

- **미세조정, 사전학습, 실로봇 재현은 이 맥에서 불가입니다.** CUDA와 대용량 GPU 메모리가 필요합니다.
- **LIBERO 체크포인트의 추론 한 청크는 조건부입니다.** 공식 코드는 CUDA 장치를 전제하므로 포팅과 측정 뒤에 판정할 수 있습니다.
- **표 재계산, 검정, 일관성 검사는 가능하고 완료했습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 판본 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소의 출처, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 시도 수 점검, 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF의 저작권 표시는 찾지 못했습니다.
- 공식 코드는 `github.com/moojink/openvla-oft`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 모델 가중치의 라이선스는 HF 모델 카드(MIT)와 기반 모델의 Llama 2 조건 확인이 필요하다는 점이 다릅니다. 가중치는 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

# FlashVLA (arXiv:2608.27384, CoRL 2026 포스터) 분석 노트

*FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference* (arXiv v1, 2026-08-27)을 표 재계산, 공식 코드의 정적 대조, 소형 numpy 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **구조는 구현할 수 있습니다.** 스태거드 버퍼, 청크 단위 인과 마스크, 다중 버퍼 공동 학습은 소형 PyTorch 모델로 옮길 수 있습니다(추정 2–4일).
- **수치 재현은 이 맥에서 불가합니다.** 지연 측정은 CUDA 장치와 CUDA Graph를 전제로 하고, 학습은 8×H200 설정입니다.
- **실로봇 재현은 장비와 데이터가 필요합니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계, 특이점 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 표 재계산, 일관성 검사, 근사 검정, 지연 산술, 토이 결과 |
| `analysis/04_code_verification.md` | 논문과 코드의 대조(D1–D10), 저자에게 물을 질문, 구현 가능성 |
| `analysis/table_check.py` | 표 재계산과 근사 검정 (scipy, 수 초) |
| `analysis/toy_streaming.py` | 지연 모델과 비동기 경계 토이 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/toy_streaming.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY 4.0입니다(arXiv abs 페이지 확인). 이 자료에는 논문 그림과 원문 PDF를 넣지 않았습니다. 원문은 arXiv:2608.27384에서 받을 수 있습니다.
- 공식 코드(`z-lab/flashvla`)는 Apache-2.0입니다. 이 분석에서는 정적으로만 읽었고 실행하지 않았습니다. 가중치와 데이터는 받지 않았습니다.
- 논문 표와 본문의 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

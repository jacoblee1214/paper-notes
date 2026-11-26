# Robot-Factored World Models via Robot Rendering (arXiv:2607.22535, CoRL 2026) 분석 노트

*RoFacto* (arXiv v2, 2026-11-02)를 표 재계산, 공식 코드 정적 대조, 작은 토이로 분석한 자료입니다. 논문 원문은 넣지 않았고, 논문 그림은 그림 1의 블록 다이어그램 한 장만 출처를 밝히고 실었습니다. 나머지 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** Wan2.1 14B 가중치만 bf16 약 26 GiB이고 학습 설정은 GPU 4장입니다. DROID 명목화 코드와 평가 코드는 공식 저장소에 없습니다.
- 표 재계산과 토이는 CPU로 끝냈습니다. 소형 축소판은 조건부로 가능합니다(노력 1–2주 추정).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 수치, 주장과 증거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산 V1–V10, 해석, 토이 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조(D1–D12), 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 1초 미만) |
| `analysis/toy_leakage.py` | 토이: 조건 종류별 결과 예측 (torch, CPU, 약 16초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/crop_figures.py` | 논문 그림 1의 블록 다이어그램 잘라내기 (pymupdf, 원문 PDF 필요) |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_leakage.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv abs 페이지의 license 링크가 CC BY 4.0입니다(확인 2026-11-26). 그림 1의 블록 다이어그램은 © 저자(Kim, Kim, Cha, Joo), CC BY 4.0, 출처 arXiv:2607.22535 그림 1 일부입니다.
- 공식 코드(https://github.com/bjkim95/rofacto)는 Apache-2.0입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

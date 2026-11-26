# BitVLA (arXiv:2506.07530, CoRL 2026 스포트라이트) 분석 노트

*BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation* (arXiv v2, 2026-03-01)을 표 재계산, 통계 검정, 공개 코드 정적 대조, 선형층 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2506.07530에서 받을 수 있습니다). 논문 그림은 다섯 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 수치 재현은 이 맥(Apple M4 Pro, 24 GiB, MPS)에서 불가입니다.** 로봇 사전학습은 5,376 H800 GPU시간이고 증류·사전학습 코드는 공개되어 있지 않으며, 실로봇은 장비가 필요합니다.
- **소형 재구현은 가능합니다.** W1.58A8 선형층과 층별 증류를 작은 PyTorch 모델로 옮기는 일은 1–2일(추정)입니다. 선형층 토이는 `analysis/toy_quant.py`로 확인했습니다.
- **공개 체크포인트 추론은 조건부입니다.** 5.64 GB 다운로드와 포크 설치, MPS 이식이 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 다섯 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, v1 대 v2 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 메모리 구성, 일관성 항목, 주장 판정, 토이 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D16, 구현 가능성 판정 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 약 10초) |
| `analysis/toy_quant.py` | 토이: 식 (1)–(4) 선형층과 삼진 통계 (numpy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 다섯 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_quant.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF의 저작권 표시는 찾지 못했습니다.
- 논문 그림 다섯 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2506.07530"). 실로봇 사진은 넣지 않았습니다.
- 공식 코드는 `github.com/ustcwhy/BitVLA`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 정적 대조는 커밋 `a8d62dfe398bfcce01283278cd716d4d6eae3c5a` 기준입니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다. 메모리 구성은 공개 허브의 config와 파일 크기 메타데이터로 계산했습니다.

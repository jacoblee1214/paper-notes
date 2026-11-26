# ActionCodec (arXiv:2602.15397, CoRL 2026 포스터) 분석 노트

*ActionCodec: What Makes for Good Action Tokenizers* (arXiv v2, 2026-10-08)를 표 재계산, 통계 검정, 합성 데이터 토이, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2602.15397에서 받을 수 있습니다). 논문 그림은 네 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 수치 재현은 이 맥(Apple M4 Pro, CUDA 없음)에서 불가입니다.** 설계 검증, 표 1, SimplerEnv는 GPU, 실로봇은 로봇이 필요합니다.
- **공개 토크나이저(약 45M) 추론은 조건부입니다.** 설치와 다운로드, 저장소 코드 실행이 필요합니다.
- **작은 VQ와 OR 측정의 축소판은 가능합니다.** `analysis/toy_or_tradeoff.py`(numpy, 약 10초)가 그 출발점입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 네 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 점검, 토이 결과, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_or_tradeoff.py` | 토이: OR과 재구성 오차의 절충 (numpy, 약 10초) |
| `analysis/plot_figures.py` | 우리가 그린 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 네 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_or_tradeoff.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 네 장(그림 1의 학습 곡선, 그림 2, 그림 3, 그림 7)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2602.15397"). 사진, 장면 위 어텐션 지도, 로봇 사진, 로고가 든 부분은 넣지 않았습니다.
- 공식 코드는 `github.com/ZibinDong/actioncodec`에 있습니다. README는 MIT라고 적지만 LICENSE 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

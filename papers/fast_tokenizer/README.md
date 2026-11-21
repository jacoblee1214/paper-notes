# FAST (arXiv:2501.09747) 분석 노트

*FAST: Efficient Action Tokenization for Vision-Language-Action Models* (arXiv v1, 2025-01-16)을 표 재계산, 토이 실험, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2501.09747에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **토크나이저 검증은 이 맥에서 가능합니다.** CPU에서 돌고, 설치 뒤에 실행해야 합니다.
- **π0-FAST 추론은 조건부입니다.** 속도는 측정하지 않았습니다.
- **학습과 시뮬 평가는 불가능합니다.** GPU가 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 설정, 결과 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 코드와 관련 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검출력, 토이 결과, 논문 내부 불일치, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조 (HF 토크나이저, openpi) |
| `analysis/recompute.py` | Table I, Table II 재계산과 검출력 (수 초) |
| `analysis/toy_fast_spectrum.py` | 토이: 주파수에 따른 0 아닌 DCT 계수 수와 재구성 오차 (1–2초, numpy) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_fast_spectrum.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF에서 추가 권리 표시를 찾지 못했습니다.
- 공식 코드는 `github.com/Physical-Intelligence/openpi`(Apache-2.0)와 Hugging Face `physical-intelligence/fast`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다(HF 설정 파일 네 개만 정적으로 읽었습니다).

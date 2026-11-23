# T-Rex (arXiv:2606.17055, CoRL 2026 포스터) 분석 노트

*T-Rex: Tactile-Reactive Dexterous Manipulation* (arXiv v2, 2026-06-18)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.17055에서 받을 수 있습니다). 논문 그림은 저자가 그린 그래프 네 장만 출처와 함께 넣었습니다.

## 구현 가능성 (요약)

- 실로봇 평가, 사전·중간학습, 공식 사후학습은 일반 노트북(Apple M4 Pro 24 GiB 기준)에서 불가입니다.
- 공개 체크포인트 추론은 조건부입니다(MPS 이식과 다운로드 필요, 속도 미측정).
- 소형 VQ-VAE 재학습과 계단식 흐름 매칭 축소판은 가능합니다. 표 재계산은 끝냈습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 버전 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 네 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 네 장(그림 2, 4, 5, 6)은 "© 저자, CC BY 4.0, 출처 arXiv:2606.17055"로 표기했고, 사진과 구조도는 넣지 않았습니다.
- 공식 코드는 `github.com/ZhuoyangLiu2005/T-Rex`(MIT)에 있습니다. 코드는 넣지 않았고 실행하거나 설치하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.

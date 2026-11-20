# Fast-WAM (arXiv:2603.16666, NeurIPS 2026 poster) 분석 노트

*Fast-WAM: Do World Action Models Need Test-time Future Imagination?* (arXiv v2, 2026-03-23)을 표 재계산, 통계 검정, 공식 코드 정적 대조, numpy 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2603.16666에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프는 모두 표 수치로 다시 그렸습니다.

## 구현 가능성 (요약)

- **공식 결과 재현은 CUDA GPU가 필요해서 이 환경에서 불가입니다.** 학습은 8–64 GPU를, 평가는 CUDA와 시뮬레이터를 전제합니다.
- **부분 가능합니다.** 5B 백본의 추론 지연 측정은 포팅과 큰 가중치 다운로드가 필요합니다. 수 M 파라미터 축소 재구현은 가능하지만 논문 수치는 재현되지 않습니다.
- **토이와 표 재계산은 CPU에서 됩니다.** `analysis/recompute.py`(수 초), `analysis/toy_imagination.py`(1초 미만).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계, 버전 차이 (v1과 v2 본문 동일) |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, 공식 저장소와 체크포인트 표, 학회 확인 |
| `analysis/03_analysis.md` | 표 재계산, 정수 복원, 쌍체·비쌍체 검정, 검출력, 토이 결과 |
| `analysis/04_code_verification.md` | 공식 코드(커밋 7faa711) 정적 대조 D1–D16, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정. 입력은 논문 텍스트 (`pdftotext -layout`으로 만든 파일) |
| `analysis/toy_imagination.py` | 토이: 상상의 정보 효과, 미래 예측 부분공간의 표현 효과 (numpy) |
| `analysis/plot_figures.py` | 그래프 세 장 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
pdftotext -layout paper.pdf paper/paper.txt        # arXiv PDF에서 만든 텍스트
python3 analysis/recompute.py
python3 analysis/toy_imagination.py
python3 analysis/plot_figures.py                   # analysis/recompute_results.json 과 toy_imagination_results.json 필요
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 라이선스 링크 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 공식 코드는 `github.com/yuantianyuan01/FastWAM`에 있습니다. `LICENSE`는 MIT입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 체크포인트는 Hugging Face `yuanty/fastwam`에 있고 라이선스 필드가 비어 있습니다. 데이터(LIBERO 전처리)는 CC BY 4.0입니다. 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 학회: NeurIPS 2026 포스터. 근거는 공식 virtual 포스터 페이지입니다.

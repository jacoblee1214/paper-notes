# Dust (Q Labs, 2026) 분석 노트

*Dust: Pretraining Transformers Without Backpropagation* (Dahal, Mandal, Gülbahar, Vegesna; Q Labs Research, 2026-10,
https://qlabs.sh/research/dust)를 부록 표 재계산, 연산량 환산, [공식 코드](https://github.com/qlabs-eng/dust)(커밋 `b20f7c0`) 대조,
numpy toy(추정기, cosine 법칙, credit decay의 편향-분산)로 분석한 자료입니다. 학습 재현은 GPU가 필요해서 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 0차 최적화·node perturbation 계보, 구성 요소별 출처 |
| `analysis/03_analysis.md` | 학습 구간, 연산량 환산, 극한 재적합 민감도, 16k에서의 cosine, cosine 법칙 유도와 toy |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조(D1–D8), 수치 V1–V4, 주장별 판정(C1–C9), 구현 가능성 |
| `analysis/table_check.py` | 본문 수치 재계산, power law 재적합, EGGROLL 외삽, 코드 배분 대조, 연산량 환산 |
| `analysis/dust_toy.py` | Dust 추정기 numpy 구현과 cosine 법칙·편향 실험 |
| `analysis/plot_figures.py` | 리포트 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
git clone https://github.com/qlabs-eng/dust code_official && git -C code_official checkout b20f7c0   # table_check.py가 dust.py의 배분을 읽음
python3 analysis/table_check.py && python3 analysis/dust_toy.py && python3 analysis/plot_figures.py
```

`table_check.py`는 수 초, `dust_toy.py`는 CPU에서 약 4분이 걸립니다.

## 출처와 라이선스

- 논문 페이지에 라이선스 표기가 없어서 논문 그림은 넣지 않았습니다. 수치는 논문 부록 표에서 인용했습니다.
- 그래프는 모두 부록 표와 공식 코드 설정으로 직접 계산해 그렸습니다.
- 공식 코드(MIT)는 이 저장소에 넣지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

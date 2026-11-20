# LingBot-VA (arXiv:2601.21998) 분석 노트

*Causal World Modeling for Robot Control* (arXiv v2, 2026-03-22)을 표 재계산, 부록 성공 규칙 재검사, 공식 저장소 정적 대조, numpy 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2601.21998에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프는 모두 표 수치로 다시 그렸습니다.

## 구현 가능성 (요약)

- **정식 재현은 불가입니다.** 학습은 8 GPU 이상, 평가는 CUDA와 시뮬레이터를 전제합니다.
- **공식 체크포인트 추론은 조건부입니다.** MPS 포팅과 수 GB 가중치가 필요합니다.
- **축소 재구현과 토이는 가능합니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, 공식 저장소와 체크포인트 표, 학회 판정 |
| `analysis/03_analysis.md` | 표 재계산, 쌍체 검정, 실세계 성공 규칙 재검사, 본문과 표의 불일치, 토이 결과 |
| `analysis/04_code_verification.md` | 공식 코드(커밋 7c6ffa9) 정적 대조 D1–D20, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정. 입력은 논문 텍스트 |
| `analysis/realworld_check.py` | 실세계 부록 표를 읽어 성공 규칙 검사 |
| `analysis/toy_noisy_history.py` | 토이: 잡음 증강의 정밀도와 강건성 교환 (numpy) |
| `analysis/plot_figures.py` | 그래프 세 장 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
pdftotext -layout paper.pdf paper/paper.txt        # arXiv PDF에서 만든 텍스트
python3 analysis/recompute.py
python3 analysis/realworld_check.py
python3 analysis/toy_noisy_history.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 라이선스 링크 확인).
- 공식 코드는 `github.com/robbyant/lingbot-va`에 있고 Apache-2.0입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 체크포인트와 데이터의 라이선스는 확인하지 못했습니다. 받지 않았습니다.
- 학회: RSS 2026 (본 학회 (RSS 2026 논문집 p015, roboticsproceedings.org/rss22 확인)).

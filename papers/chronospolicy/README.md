# Chronos Policy (arXiv:2610.13929) 분석 노트

*Chronos Policy: Learning Robot Policies on Physical Time* (Wang, Clark, Song, Stanford, 2026)을
논문 표, 본문, [프로젝트 페이지](https://changhaowang.github.io/chronospolicy/)의 결과 데이터(`data.js`, `dex-data.js`)를 대조하고
numpy toy로 time-aware RoPE와 학습/평가 시간 범위를 확인하며 분석한 자료입니다. 코드는 아직 공개되지 않았습니다(2026-10-28 기준).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보와 구성 요소별 출처 |
| `analysis/03_analysis.md` | RoPE 성질, 시간 커널, 학습/평가 시간 범위, 시드 역산과 검정 |
| `analysis/04_claim_verification.md` | 수치 정합성(V1–V5), 주장별 판정(C1–C9), 실험 설계 점검(D1–D9), 구현 가능성 |
| `analysis/table_check.py` | 표 ↔ 웹 데이터 대조, 본문 수치 재계산, 시드별 성공 횟수 역산, 검정 |
| `analysis/timing_toy.py` | time-aware RoPE와 시간 범위 numpy toy |
| `analysis/plot_headline.py` | 리포트 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
mkdir -p paper/site
curl -L -o paper/paper.pdf https://arxiv.org/pdf/2610.13929 && pdftotext -layout paper/paper.pdf paper/paper.txt
for f in data.js dex-data.js; do curl -L -o paper/site/$f https://changhaowang.github.io/chronospolicy/$f; done
python3 analysis/table_check.py && python3 analysis/timing_toy.py && python3 analysis/plot_headline.py
```

모두 CPU에서 수 초 안에 끝납니다. 웹 데이터 파일은 프로젝트 페이지가 바뀌면 달라질 수 있습니다(2026-10-28에 받은 값 기준).

## 출처와 라이선스

- 논문은 **CC BY 4.0**입니다(arXiv abs 페이지 확인). `report/figures/paper_fig*.png`는 논문에서 잘라낸 그림이며 © 저자, CC BY 4.0, 출처 arXiv:2610.13929입니다.
- `headline_ablation.png`, `standard_tasks.png`는 논문 Table 3·4와 프로젝트 페이지 데이터로, `timing_coverage.png`는 논문 부록의 설정으로 직접 그렸습니다.
- 원문 PDF와 프로젝트 페이지 파일은 이 저장소에 넣지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

# AnyRank (arXiv:2610.14603, CoRL 2026 포스터) 분석 노트

*AnyRank: Reliable Anytime Policy Ranking via Sequential Testing* (arXiv v2, 2026-10-16)을 표 재계산, 공식 코드 정적 대조, 자체 e-process 토이로 분석한 자료입니다. 논문 원문 PDF와 공식 코드는 넣지 않았습니다(arXiv:2610.14603, https://github.com/irom-princeton/AnyRank).

## 구현 가능성 (요약)

- **알고리즘(그래프 위험 재배분, 표본 선택, 쌍별 e-process)은 CPU만으로 구현·시험할 수 있습니다.** `analysis/toy_anyrank.py`(약 4초, numpy)로 확인했습니다.
- **논문의 실험(세계모델 사전, 실로봇 수집)은 GPU와 로봇이 필요해 재현하지 않았습니다.**
- **공식 코드는 실행하지 않았습니다.** 핵심 쌍별 검정은 별도 패키지(README 기준 CC BY-NC 4.0)에 의존합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 한 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 코드 저장소 표 |
| `analysis/03_analysis.md` | 수치 재계산 V1–V11과 토이 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조 D1–D15, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산 (numpy, scipy, pandas, 수 초) |
| `analysis/toy_anyrank.py` | 자체 e-process 토이 (numpy, pandas, 약 4초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 한 장(© 저자, CC BY 4.0) |

## 재실행

공식 저장소를 `repo/`에 받아 두고(데이터 CSV만 읽습니다), 아래를 실행합니다.

```bash
git clone --depth 1 https://github.com/irom-princeton/AnyRank repo
pip install numpy scipy pandas matplotlib
python3 analysis/recompute.py
python3 analysis/toy_anyrank.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF의 저작권 표시는 찾지 못했습니다.
- 논문 그림은 그림 4(막대 차트) 한 장만 실었고 "© 저자, CC BY 4.0, 출처 arXiv:2610.14603"을 표기했습니다.
- 공식 코드는 MIT입니다. 이 저장소에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

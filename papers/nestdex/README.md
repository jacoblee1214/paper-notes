# NestDex (arXiv:2608.13362) 분석 노트

*NestDex: Nested Policy Learning with Copilot Assisted Teleoperation for Dexterous Manipulation*
(Zhao, Tang, Ba, Zhi, University of Sydney · Vanderbilt University, 2026)을 논문 표 재계산, 프로젝트 사이트 공개 데이터 재분석,
temporal ensembling toy로 분석한 자료입니다. 공식 코드는 공개되지 않아(2026-11-05 확인) 코드 대조는 하지 않았습니다.

**구현할 수 있는지:** 공식 재현은 불가능합니다(코드·학습된 정책·시연 데이터 미공개, 전용 하드웨어 필요).
아이디어(클러치 진행도 → 실행 index, 앞 질의·뒤 재생, proprio chunk 정책 + ACT식 앙상블, 손 VAE)는 수백 줄로 다시 만들 수 있고,
장벽은 GPU가 아니라 손 기술 시연 장비입니다(`analysis/04_claim_verification.md` §4).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, 관련 GitHub 저장소 |
| `analysis/03_analysis.md` | 시간 분해, 클러치 동역학, 앙상블 가중치와 toy, 공개 데이터 재분석, 학습 규모, Table III 통계 |
| `analysis/04_claim_verification.md` | 주장 검증(C1–C9), 논문에 없는 재현 정보, 구현 가능성 |
| `analysis/table_check.py` | Table II·III 통계(Fisher, Holm, Mantel–Haenszel)와 공개 데이터 재분석 |
| `analysis/ensemble_toy.py` | Eq. 2 temporal ensembling 가중치와 접촉 반응 toy |
| `analysis/extra_clutch_toy.py` | 1차원 닫기 toy: 클러치 속도 상한과 chunk 길이의 접촉 반응 |
| `analysis/extra_effort_timing.py` | 공개 노력 곡선의 시간 해상 AUC와 블록 순서 점검 |
| `analysis/plot_figures.py` | 리포트 그래프 네 장 |

## 재실행

공개 데이터 파일은 이 저장소에 넣지 않았습니다(웹사이트 저장소에 라이선스 표기가 없음). 먼저 받아 둡니다.

```bash
pip install numpy scipy matplotlib
mkdir -p paper && curl -L -o paper/inner-policy-effort.json https://raw.githubusercontent.com/cheese-zj/nestdex-site/63f96e6/public/data/inner-policy-effort.json
python3 analysis/table_check.py && python3 analysis/ensemble_toy.py && python3 analysis/plot_figures.py
```

모두 CPU에서 몇 초 안에 끝납니다. 논문 수치는 스크립트에 출처(표·식 번호)와 함께 적었습니다.

## 출처와 라이선스

- 논문은 **CC BY 4.0**입니다(arXiv abs 페이지 확인, PDF 안에 다른 권리 표기 없음). `report/figures/paper_fig*.png`는 논문에서 잘라낸 그림이며 © 저자, CC BY 4.0, 출처 arXiv:2608.13362입니다.
- `effort_public_data.png`와 `table_check_results.json`의 C5 항목은 프로젝트 사이트의 공개 데이터(`cheese-zj/nestdex-site` @ `63f96e6`, `public/data/inner-policy-effort.json`)에서 계산한 값입니다. 원 데이터는 위 주소에서 받습니다.
- 나머지 그래프는 논문 수치와 toy 시뮬레이션으로 직접 그렸습니다. 원문 PDF는 이 저장소에 넣지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

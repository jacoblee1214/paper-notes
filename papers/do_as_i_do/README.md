# Do as I Do (arXiv:2606.19333) 분석 노트

*Do as I Do: Dexterous Manipulation Data from Everyday Human Videos* (Paliwal et al., UC Berkeley, 2026)를
논문과 공개 코드([malik-group/do-as-i-do@824591b](https://github.com/malik-group/do-as-i-do))를 대조해 분석한 자료입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보 |
| `analysis/03_derivations.md` | Eq. 1 = EMA 필터라는 유도, 정렬 식, 최적화기의 실제 형태 |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조 (D1–D10) |
| `analysis/guidance_toy.py` | Eq. 1 성질의 CPU toy 검증 |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/guidance_toy.py            # toy 검증 + 그림
# whisking 결과 확인: 공식 레포를 repo/ 에 clone 한 뒤
git clone https://github.com/malik-group/do-as-i-do repo && python3 analysis/whisk_tracking_check.py
```

## 출처와 라이선스

- 논문 그림(`report/figures/paper_fig*.png`)의 출처는 Paliwal et al., arXiv:2606.19333이며, 원 논문 라이선스는 CC BY 4.0입니다.
- 그 밖의 분석 문서, 그림, 코드는 이 저장소 작성자가 만들었습니다.
- 공식 코드와 서브모듈(SAM License, CC BY-NC-ND 등)은 포함하지 않았습니다.

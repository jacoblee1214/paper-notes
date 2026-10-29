# Mana (arXiv:2606.13677) 분석 노트

*Mana: Dexterous Manipulation of Articulated Tools* (Yin, Shi, Abbeel, Liu, UC Berkeley · CMU · Stanford · Amazon FAR, 2026)을
Table 1·2와 Fig. 7 재계산, 손가락 힘 용량·적분기 안정성·랜덤화 크기 toy, 공개된 Lightning Grasp 코드(`zhaohengyin/lightning-grasp` @ `af43818`)
대조로 분석한 자료입니다. Mana 자체 코드는 공개되지 않았습니다(2026-10-29 확인).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, 관련 GitHub 저장소 |
| `analysis/03_analysis.md` | Table 1 통계, 단계 연쇄 vs Table 2, Fig. 7 판독, 물리 toy, 보상 스케일 |
| `analysis/04_claim_verification.md` | 주장별 판정(C1–C13), Lightning Grasp 코드 대조(D1–D7), 구현 가능성 |
| `analysis/table_check.py` | Table 1·2·Fig. 7 재계산 |
| `analysis/physics_toy.py` | 손가락 힘 용량, 적분기 안정성, 랜덤화 환산 (numpy, 수 초) |
| `analysis/plot_figures.py` | 리포트 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/physics_toy.py
cd analysis && python3 plot_figures.py
```

모두 CPU에서 몇 초 안에 끝납니다. 수치는 출처와 함께 스크립트에 적었고, 논문 PDF는 필요하지 않습니다.

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.** 그림과 영상은 [프로젝트 페이지](https://zhaohengyin.github.io/mana)에서 볼 수 있습니다.
- 그래프는 논문 표와 그림에서 읽은 수치(`table1_ci.png`, `chain_vs_table2.png`, `fig7_ci.png`)와 직접 만든 toy(`physics.png`)로 그렸습니다.
- Lightning Grasp 코드는 비상업 라이선스라 이 저장소에 포함하지 않았습니다. 대조 결과만 적었습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

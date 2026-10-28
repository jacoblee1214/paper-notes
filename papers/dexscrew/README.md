# DexScrew (arXiv:2512.02011) 분석 노트

*Learning Dexterous Manipulation Skills from Imperfect Simulations* (Hsieh et al., UC Berkeley, 2025–2026)를
논문과 공개 코드([dexscrew@3bde4e3](https://github.com/x-robotics-lab/dexscrew),
[skill-teleop@dec60d6](https://github.com/x-robotics-lab/skill-teleop),
[minbc@b64c53f](https://github.com/x-robotics-lab/minbc/tree/b64c53f59ccb47230df16b3da31def8f16694557))를 대조해 분석한 자료입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보 (Hora → penspin → DexScrew) |
| `analysis/03_analysis.md` | 표 재구성과 유의성 재계산, privileged 버그의 영향, 회전-하강 결합 |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조 (D1–D9) |
| `analysis/stats_check.py` | Table I/II 재구성과 검정 (`python3 analysis/stats_check.py`) |
| `analysis/priv_layout_check.py` | privileged info 배치 재현 (공식 레포를 `dexscrew/`에 clone한 뒤 실행) |

## 재실행

```bash
pip install numpy scipy matplotlib pyyaml
python3 analysis/stats_check.py && python3 analysis/plot_stats.py
git clone https://github.com/x-robotics-lab/dexscrew && python3 analysis/priv_layout_check.py screwdriver
```

## 출처와 라이선스

- 논문 그림(`report/figures/paper_fig*.png`)의 출처는 Hsieh et al., arXiv:2512.02011이며, 원 논문 라이선스는 CC BY 4.0입니다.
- 공식 코드(MIT)는 포함하지 않았습니다. 분석 문서, 그림, 스크립트는 이 저장소 작성자가 만들었습니다.

# Contact-Grounded Policy (arXiv:2603.05687) 분석 노트

*Contact-Grounded Policy: Dexterous Visuotactile Policy with Generative Contact Grounding*
(Xu et al., Purdue · Meta Reality Labs Research · UW–Madison, RSS 2026)을 논문 표 재계산과, 핵심 사상(Eq. 1)의
물리적 유도 및 3-DoF 손가락 toy로 분석한 자료입니다. 공식 코드는 공개되지 않았습니다(2026-10-28 확인).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보와 구성 요소별 출처 |
| `analysis/03_derivations.md` | contact-consistency mapping의 준정적 유도, toy 결과, Fig. 13 재해석, 실행 시간 구조 |
| `analysis/04_claim_verification.md` | 주장별 판정(C1–C13)과 구현 가능성 |
| `analysis/table_check.py` | Table II–VI과 그림 수치 재계산 |
| `analysis/mapping_toy.py` | 3-DoF 평면 손가락 + 관절 PD + 촉각 taxel toy (numpy, CPU 약 2분) |
| `analysis/plot_figures.py` | 리포트 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/mapping_toy.py
cd analysis && python3 plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 표의 수치(`success_ci.png`), 논문 그림에서 읽은 값과 Table III(`noise_vs_state_only.png`), 직접 만든 toy(`mapping_toy.png`)로 그렸습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

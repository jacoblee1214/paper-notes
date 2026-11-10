# AdaVLA (arXiv:2608.29208) 분석 노트

*AdaVLA: Adaptive Step Flow Matching for Training-free Acceleration of Vision-Language-Action Models*
(Han, Han, Yi, IROS 2026 채택)을 표 재계산, 지연 산술, 곡률 기준 흐름 toy로 분석한 자료입니다.
공식 코드는 찾지 못했습니다(2026-11-10 확인).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림·표 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations, v1 → v2 차이 |
| `analysis/02_prior_work.md` | 선행연구 계보, 가까운 대안, 관련 GitHub 저장소 |
| `analysis/03_analysis.md` | 표 재계산과 유의성, 지연 산술, 역산 NFE, 식 9 유도, toy 결과 |
| `analysis/04_claim_verification.md` | 주장별 판정, 빠진 정보, 권리 표기, 구현 가능성 |
| `analysis/table_check.py` | 표 재계산 (수 초) |
| `analysis/pr_check.py` | 참여율 정의 검산, 스케일 불변성, 계산 비용 (수 초) |
| `analysis/flow_toy.py` | 2차원 가우시안 혼합 흐름 toy (numpy 1스레드, 약 40초) |
| `analysis/plot_figures.py` | 직접 만든 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/pr_check.py && python3 analysis/flow_toy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스, 1쪽에 IEEE 저작권 문구). 그래서 이 저장소에는 **논문 그림, 논문 표 이미지, 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 표의 수치와 직접 만든 toy로 그렸습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

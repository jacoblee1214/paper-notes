# Real-Time Force Regulation for Whole-Hand Dexterous Grasping (arXiv:2609.30082) 분석 노트

*Real-Time Force Regulation for Whole-Hand Dexterous Grasping* (Kim, Alexiev, Lin, Kim, Kim, Lee,
서울대 · MIT · 연세대, 2026)을 그림·표 재계산, 힘 분배 QP 구조 toy, 자세 오차 toy, 같은 연구실의 공개 저장소
[rophi-lab/RoPhiTutorial](https://github.com/rophi-lab/RoPhiTutorial)(`85aaa4c`) 대조로 분석한 자료입니다.
논문 공식 코드는 공개되지 않았습니다(2026-10-29 확인, 프로젝트 페이지 "Code (Soon)").

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, GitHub 정리 |
| `analysis/03_analysis.md` | QP 비유일성, actuation-consistency 제약의 차원 의존성, 계산 속도, 하중 대 마찰 용량, 자세 오차와 접점 추정 |
| `analysis/04_code_verification.md` | 같은 연구실 코드 대조(D1–D9), 수치 재계산(C1–C8), 구현 가능성 |
| `analysis/table_check.py` | Fig. 3·4·6·7, Table I, 본문 수치 재계산 (`fig_digitize_results.json` 필요) |
| `analysis/fig_digitize.py` | Fig. 4 곡선 픽셀 판독 (원문 PDF가 `paper/paper.pdf`에 있어야 함) |
| `analysis/qp_toy.py` | 공개 코드 형식의 접촉력 QP를 OSQP식 ADMM으로 푼 toy (20 DoF 손 모델) |
| `analysis/pose_error_toy.py` | 자세 오프셋 아래 기하 접점 재현율·오검출·법선 오차 |
| `analysis/plot_figures.py` | 리포트 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib pillow
python3 analysis/table_check.py && python3 analysis/qp_toy.py && python3 analysis/pose_error_toy.py
cd analysis && python3 plot_figures.py
# Fig. 4 판독을 다시 하려면: mkdir -p paper && curl -L -o paper/paper.pdf https://arxiv.org/pdf/2609.30082v1 && python3 analysis/fig_digitize.py
```

모두 CPU에서 수 초 안에 끝납니다.

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 그림·표의 수치(`fig3_counts_ci.png`)와 직접 만든 toy(`nullspace_eps.png`, `pose_error_contacts.png`)로 그렸습니다.
- 대조한 공개 저장소는 논문 공식 코드가 아니라 같은 연구실의 튜토리얼입니다. 코드는 이 저장소에 넣지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

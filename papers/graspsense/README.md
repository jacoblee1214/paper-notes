# GraspSense (arXiv:2604.05697) 분석 노트

*GraspSense: Physically Grounded Grasp and Grip Planning for a Dexterous Robotic Hand via Language-Guided Perception and Force Maps*
(Semenyakina, Snegirev et al., Skoltech, 2026)을 표 재계산, 힘 지도 재순위화 toy, 저자 추정 저장소
[eliizavetas/GraspSense](https://github.com/eliizavetas/GraspSense)(`16dc5fb`) 대조로 분석한 자료입니다. 재현 실험(GPU)은 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, 관련 GitHub 표 |
| `analysis/03_analysis.md` | 표 정합성(여유 = 1/λ 등), 재순위화의 재료 불변성, 마찰 정역학, 조회 방식 차이, 허용 하중의 크기 |
| `analysis/04_code_verification.md` | 논문 ↔ 저자 저장소 대조(D1–D9), 수치 재계산(C1–C9, T1–T3), 구현 가능성 |
| `analysis/table_check.py` | Table I–IV 재계산과 정합성 점검 |
| `analysis/force_map_toy.py` | 힘 지도 재순위화·조회 방식·두께 환산 toy (numpy) |
| `analysis/plot_figures.py` | 리포트 그래프 세 장 |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/table_check.py && python3 analysis/force_map_toy.py
cd analysis && python3 plot_figures.py
```

모두 CPU에서 몇 초 안에 끝납니다. 논문 PDF나 저자 코드는 필요하지 않습니다(수치는 스크립트에 출처와 함께 적었습니다).

## 출처와 라이선스

- 논문은 **CC BY-NC-ND 4.0**입니다(arXiv abs 페이지 확인). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- `report/figures/*.png`는 논문 표의 수치와 직접 만든 toy로 그린 그래프입니다.
- 저자 저장소(MIT)의 코드는 이 저장소에 넣지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

# Stiffness Copilot (arXiv:2603.14068) 분석 노트

*Stiffness Copilot: An Impedance Policy for Contact-Rich Teleoperation* (Yeping Wang 외, IROS 2026 채택)을
사용자 연구 통계 재계산, 라벨 식의 닫힌 해 검산, 접촉 모델·식별 가능성 numpy toy로 분석한 자료입니다.
방법 코드와 가중치는 공개되지 않았습니다. 웹 저장소(stiffness-copilot/stiffness-copilot.github.io)에는 뷰어와 내보내기 스크립트만 있고 라이선스가 없습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 분석 (Abstract → Method → Experiments → Limitations), v1 ↔ v2 메모 |
| `analysis/02_prior_work.md` | 선행연구 계보, 가까운 대안, 관련 GitHub 표 |
| `analysis/03_analysis.md` | 사용자 연구 재계산(Welch·Holm·Friedman·MWU), 라벨 식 유도, 접촉 toy 해석식, 식별 toy |
| `analysis/04_claim_verification.md` | 주장별 판정(C1–C18), 빠진 정보, 권리 표기, 구현 가능성 |
| `analysis/user_study_check.py` | Table I과 Fig. 3 판독값으로 통계 재계산 (수 초) |
| `analysis/contact_toy.py` | 2D 임피던스 접촉 toy: 방향별 대 고정 강성, 법선 오차 해석식 (수 초) |
| `analysis/label_toy.py` | 섹터 로그 평균 라벨의 닫힌 해, 공분산 대비, ε 민감도, 식별 toy (수 초) |
| `analysis/plot_figures.py` | 리포트 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/user_study_check.py
python3 analysis/contact_toy.py
python3 analysis/label_toy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 Table I과 Fig. 3에서 읽은 수치와 직접 만든 toy로 그렸습니다. Fig. 3의 참가자별 성공 횟수는 막대에서 읽은 값입니다.
- 웹 저장소의 데이터는 조건·과제당 대표 시행 한 번이라 통계에 쓰지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

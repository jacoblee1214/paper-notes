# Visible Touch 분석 노트 (arXiv:2609.14156, CoRL 2026 Spotlight)

*Visible Touch: Rendering Contact for Visuomotor Policies* (arXiv v2, 2026-10-07)을 표 재계산, 시드 단위 검정, numpy 토이, 공식 저장소 정적 대조로 읽은 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(라이선스 CC BY 4.0이지만 공개본에서는 우리 그래프만 씁니다).

## 구현 가능성 (요약)

- **전면 재현: 이 맥에서는 불가.** 공식 저장소(프로젝트 페이지가 링크, 개인 계정, 라이선스 없음)는 센서 몰드 생성 도구뿐입니다. 오버레이 코드, LIBERO와 π0.5 실험 코드, PCB 레이아웃은 없습니다. π0.5 실험은 H200급 GPU와 실로봇이 필요합니다.
- **부분 구현: 가능.** 투영 오버레이는 논문 식으로 numpy에서 구현할 수 있습니다. 몰드 도구는 CAD 환경이 있으면 돌릴 수 있습니다.
- **불일치:** 센서 자석 치수가 논문 BOM(Ø2 mm)과 저장소 설정(Ø3 mm)에서 다릅니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 비판점, v1과 v2의 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, 시드 단위 검정, 실제 실험 검정, 일관성 검사, 토이 결과 |
| `analysis/04_code_verification.md` | 논문과 공식 저장소 대조, 구현 가능성, 라이선스 |
| `analysis/table_check.py` | 표 재계산과 검정 (scipy, 수 초) |
| `analysis/toy_overlay.py` | 토이 A(2D 화살표의 힘 복원)와 토이 B(공유 항으로 생기는 상관) (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/toy_overlay.py
python3 analysis/plot_figures.py
```

원문 PDF는 다음 명령으로 받을 수 있습니다(라이선스 CC BY 4.0): `curl -L -o paper.pdf https://arxiv.org/pdf/2609.14156v2`

## 출처와 라이선스

- 논문은 CC BY 4.0입니다(arXiv abs 페이지 확인, 2026-11-15). PDF 본문에 권리 표시는 없습니다. 공개본에는 논문 그림을 넣지 않았습니다.
- 논문의 표 수치는 인용과 재계산을 위해 옮겼습니다. 문장과 그림은 옮기지 않았습니다.
- 그래프와 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다. 토이의 가정(40개 과제, 무작위 성공률 분포, 기하 파라미터)은 각 스크립트와 분석 문서에 적혀 있습니다.
- 공식 저장소는 읽기만 했습니다. 실행, 설치, 가중치와 데이터 다운로드는 하지 않았습니다.

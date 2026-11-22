# GELLO (arXiv:2309.13037, IROS 2024) 분석 노트

*GELLO: A General, Low-Cost, and Intuitive Teleoperation Framework for Robot Manipulators* (arXiv v2, 2024-07-18)를 사용자 연구 표 재계산, 통계 검정, 공식 코드 정적 대조로 읽은 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2309.13037에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **전체 재현은 이 맥에서 불가능합니다.** 장치 제작, 실로봇, 사람 사용자 연구가 필요합니다.
- **공식 소프트웨어의 CPU 실행은 승인 후 가능합니다.** 이번 작업에서는 실행하지 않았습니다.
- **관절 보정 수식 검사는 가능합니다.** 공식 설정 값을 numpy로 옮겨 검사할 수 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 (v1 대 v2) |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, 통계 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비배타 배포 라이선스(`arxiv.org/licenses/nonexclusive-distrib/1.0/`)입니다. CC BY가 아니므로 논문 그림은 넣지 않았습니다.
- 학회 정보는 Crossref에 등록된 IEEE 메타데이터(DOI `10.1109/IROS58592.2024.10801581`)에서 확인했습니다.
- 공식 소프트웨어(`wuphilipp/gello_software`)와 하드웨어(`wuphilipp/gello_mechanical`)는 MIT입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값(성공 수, 실패 기호 개수)은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.

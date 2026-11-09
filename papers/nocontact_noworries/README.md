# NoContactNoWorries (arXiv:2606.24450v2) 분석 노트

Patil, Das, Bhosale, Roy의 "NoContactNoWorries: Estimating Contact through Vision and Proprioception for In-Hand Dexterous Manipulation"(IROS 2026 채택)을 표 재계산, 검정, 두 개의 numpy toy로 읽은 분석입니다.
공식 코드가 공개되지 않아(2026-11-09 확인) 코드 대조는 하지 않았고, 주장 검증으로 대체했습니다.

**구현할 수 있는지:** 공식 재현은 불가능합니다(모델·학습·데이터 코드 미공개, 시뮬레이션은 NVIDIA GPU, 실물은 LEAP Hand와 FSR 센서). 접촉 헤드 구조는 수백 줄로 옮길 수 있고, 어려운 부분은 학습 데이터입니다(`analysis/04_claim_verification.md` §4).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 읽기 (방법, 실험, 결과, 한계, 비판적 해석), v1→v2 차이 |
| `analysis/02_prior_work.md` | 선행연구 계보, 관련 저장소와 GitHub 표 |
| `analysis/03_analysis.md` | Table I–III 재계산과 검정, 유도, 감각·제어 toy |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C20, 재현 정보 누락, 구현 가능성 |
| `analysis/table_check.py` | Welch t, Holm, 부호 검정, 검출력(MDE), ± 가정 민감도 → `table_check_results.json` |
| `analysis/sensing_toy.py` | 관절 오차와 가려진 시각의 접촉 정보, 속도 결합 대조 → `sensing_toy_results.json` |
| `analysis/control_toy.py` | 위양성·위음성·디바운스에 따른 정지 성공률, F1 → FPR 변환 → `control_toy_results.json` |
| `analysis/plot_figures.py` | 그래프 네 장 (논문 그림이 아니라 수치로 직접 그림) |

재실행 (모두 CPU, 수 초):

```bash
python3 analysis/table_check.py
python3 analysis/sensing_toy.py
python3 analysis/control_toy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 페이지에 CC BY-SA 4.0으로 표기되어 있습니다. 논문 그림과 원문 PDF는 이 패키지에 넣지 않았습니다. 그림은 논문 수치와 toy로 직접 그렸습니다.
- 수치는 논문 Table I–III에서 옮겼고, 재계산 스크립트에 값과 출처(표 번호)를 적었습니다.
- 공식 코드 저장소는 없습니다. 프로젝트 페이지 저장소(웹사이트만)와 제3자 WIP 구현(README와 LICENSE만)을 확인했습니다.
- toy의 δ, α, 접촉 비율, 가림 확률은 가정입니다. 결론은 경향으로 읽어야 합니다.

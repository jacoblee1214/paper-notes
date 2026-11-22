# ZETA (arXiv:2609.02546, CoRL 2026 스포트라이트) 분석 노트

*ZETA: A Controlled Study of Zero-Shot Cross-Embodiment VLA Transfer for Tabletop Manipulation* (arXiv v3, 2026-10-11)를 부록 표 재계산, 통계 점검, 주장 검증으로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 라이선스가 CC BY-NC-ND 4.0입니다). 원문은 arXiv:2609.02546에서 받을 수 있습니다.

## 구현 가능성 (요약)

- **논문 전체의 재현은 불가능합니다.** 공식 코드, 데이터, 가중치가 공개되지 않았고(GitHub는 README만), 설정이 H800 16장 사전학습과 IsaacSim 렌더링, 실로봇 7종을 전제합니다.
- **재계산과 좌표계 불변성 검산은 완료했습니다.** `analysis/recompute.py`(numpy, scipy, 1초)와 `analysis/toy_frame_invariance.py`(1초 미만).
- **자체 작성 소형 합성 실험은 가능합니다**(torch, CPU 또는 MPS, 분 단위).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (공개용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 버전(v1 → v3 차이), 학회, 권리, 설계, 결과 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, 자료 표 |
| `analysis/03_analysis.md` | 재계산 V1–V11, 일관성 K1–K12 |
| `analysis/04_claim_verification.md` | 주장 C1–C15, 코드 공개 시 확인 W1–W10, 구현 가능성 |
| `analysis/parse_tables.py`, `recompute.py` | 논문 부록 표 읽기와 재계산 (numpy, scipy) |
| `analysis/toy_frame_invariance.py` | 표 1 표현의 좌표계 불변성 검산 |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

`parse_tables.py`는 논문 PDF에서 만든 텍스트(`paper/paper.txt`)를 읽습니다. 원문을 받아 텍스트를 만든 뒤 실행합니다.

```bash
pip install numpy scipy matplotlib
mkdir -p paper && curl -L https://arxiv.org/pdf/2609.02546 -o paper/paper.pdf
pdftotext -layout paper/paper.pdf paper/paper.txt
cd analysis
python3 recompute.py
python3 toy_frame_invariance.py
python3 plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv:2609.02546 v3입니다. arXiv 라이선스는 CC BY-NC-ND 4.0입니다. 논문 그림은 싣지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트와 분석 문서는 이 저장소 작성자가 만들었습니다.
- 공식 저장소는 README뿐이라 코드를 읽지 못했습니다. 가중치와 데이터는 받지 않았습니다.

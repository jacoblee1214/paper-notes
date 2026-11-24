# GenAN (arXiv:2604.09487, CoRL 2026 포스터) 분석 노트

*Sim-to-Real Transfer for Muscle-Actuated Robots via Generalized Actuator Networks* (arXiv v4, 2026-10-15)를 표 재계산, 통계 검정, 공식 코드 정적 대조, 장난감 실험으로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2604.09487에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **핵심 아이디어의 축소 재구현과 식 검증은 CPU로 가능합니다.** `analysis/toy_position_vs_torque_loss.py`(약 20초).
- **공식 코드로 학습·평가는 조건부입니다.** python 3.10 새 환경, JAX, MJX, skrl 포크, Git LFS 데이터가 필요하고 정책 학습은 GPU 전제입니다.
- **실물 재현은 PAMY2 로봇과 공 추적 장비가 필요합니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V13, 장난감 K1–K3, 주장 판정 C1–C14 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D27, 라이선스, 구현 가능성 |
| `analysis/extract_figure_values.py` | 논문 PDF 벡터 그림에서 값 읽기 (PyMuPDF) |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 1초 미만) |
| `analysis/toy_position_vs_torque_loss.py` | 식 2 확인, 질량행렬 이방성, 위치 손실 대 토크 손실 장난감 (mujoco, torch) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib mujoco torch pymupdf
python3 analysis/extract_figure_values.py paper.pdf   # arXiv:2604.09487v4 PDF 경로
python3 analysis/recompute.py
python3 analysis/toy_position_vs_torque_loss.py       # 자체 4링크 사슬. 저자 MJCF 폴더가 있으면 --assets <폴더>
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv non-exclusive 라이선스(arxiv.org/licenses/nonexclusive-distrib/1.0)이고 CC BY가 아니므로 논문 그림은 싣지 않았습니다. 이 저장소의 그래프는 작성자가 다시 그린 것입니다.
- 공식 코드는 `github.com/intelligent-soft-robots/genan`에 있고 LICENSE 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 가중치와 데이터셋은 받지 않았습니다.

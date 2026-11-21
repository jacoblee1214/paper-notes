# ACT (ALOHA) (arXiv:2304.13705) 분석 노트

*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (Zhao, Kumar, Levine, Finn; RSS 2023)을
논문 수치 재계산과 공식 코드(MIT) 정적 대조로 읽은 자료입니다. 실로봇과 학습 실험은 하지 않았습니다.

## 구현 가능성 (요약)

- 실로봇 재현은 불가능합니다 (팔 4개, 웹캠 4대, ROS 1 환경 필요).
- 시뮬레이션 재현(MuJoCo Transfer Cube)은 데이터 생성은 CPU로 가능하지만 학습은 GPU가 필요합니다. 맥에서 학습 시간은 측정하지 않았습니다.
- 공식 코드는 `.cuda()`가 하드코딩되어 있어 맥(MPS)으로 옮기려면 수정이 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림과 개인용 항목 제외) |
| `analysis/01_paper_analysis.md` | Abstract → 장비 → 방법 → 실험 → 한계, 버전·라이선스·학회 확인 |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, 공식 코드와 프로젝트 페이지 표 |
| `analysis/03_analysis.md` | 표 I·II 재계산 (Wilson CI, Fisher 검정, Holm), 본문 수치 대조 (K 항목) |
| `analysis/04_code_verification.md` | 논문 ↔ 공식 코드 대조 (D1–D23) |
| `analysis/recompute.py` | 재계산 스크립트 (논문 표 값을 코드에 직접 넣음, 실행 없이 값 확인 가능) |
| `analysis/plot_figures.py` | 실세계 성공률 그래프 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py      # → analysis/recompute_results.json
python3 analysis/plot_figures.py   # → report/figures/act_real_ci.png
```

## 출처와 라이선스

- 논문 라이선스는 arXiv 비독점 배포(`nonexclusive-distrib/1.0`)이며 CC BY가 아닙니다. 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.** 원문은 https://arxiv.org/abs/2304.13705 에서 받을 수 있습니다.
- `report/figures/act_real_ci.png`는 논문 표 I·II의 수치로 직접 그린 그래프입니다.
- 공식 코드(`tonyzhaozh/act`, `tonyzhaozh/aloha`)는 MIT 라이선스이며 이 저장소에 복제하지 않았습니다. 대조는 각 커밋(742c753, 06369f0)을 읽어 했습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

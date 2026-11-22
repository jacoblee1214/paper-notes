# Mobile ALOHA (arXiv:2401.02117) 분석 노트

*Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation* (Fu, Zhao, Finn; CoRL 2024)을
논문 표 수치 재계산과 공식 코드(MIT) 정적 대조로 읽은 자료입니다. 실로봇과 학습 실험은 하지 않았습니다.

## 구현 가능성 (요약)

- 실로봇 재현은 불가능합니다 (이동 기지, 팔 4개, 카메라 3대, ROS 1 필요).
- 학습 재현은 데이터를 받고 MPS 이식 후 시간을 잰 뒤 판단합니다. 논문은 학습 장치와 시간을 쓰지 않았습니다.
- 공식 코드는 `.cuda()`가 47곳 하드코딩되어 있어 맥(MPS)으로 옮기려면 수정이 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림과 개인용 항목 제외) |
| `analysis/01_paper_analysis.md` | 논문 분석: 버전, 학회 (PMLR v270 확인), 장비, 방법, 실험, 한계 |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, 공식 코드와 프로젝트 페이지 표 |
| `analysis/03_analysis.md` | 표 1–4 재계산 (Wilson 구간, Fisher·Holm 검정, 정수 사슬 검사), 본문 대조, 불일치 목록 |
| `analysis/04_code_verification.md` | 논문 ↔ 공식 코드 대조 (D1–D21), 숨은 상수, 구현 가능성 |
| `analysis/recompute.py` | 재계산 스크립트 (논문 표 값을 코드에 직접 넣음) |
| `analysis/plot_figures.py` | 표 1 성공률 그래프 (논문 수치로 직접 그린 것) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py      # → analysis/recompute_results.json
python3 analysis/plot_figures.py   # → report/figures/mobile_aloha_cotrain.png
```

## 출처와 라이선스

- 논문 라이선스는 arXiv 비독점 배포(`nonexclusive-distrib/1.0`)이며 CC BY가 아닙니다. 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.** 원문은 https://arxiv.org/abs/2401.02117 에서 받을 수 있습니다.
- `report/figures/mobile_aloha_cotrain.png`는 논문 표 1의 수치로 직접 그린 그래프입니다.
- 공식 코드(`MarkFzp/mobile-aloha`, `MarkFzp/act-plus-plus`)는 MIT 라이선스이며 이 저장소에 복제하지 않았습니다. 대조는 각 커밋(0e40324, 26bab07)을 읽어 했습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

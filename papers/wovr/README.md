# WoVR (arXiv:2602.13977, CoRL 2026 포스터) 분석 노트

*WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL* (arXiv v3, 2026-11-05)을 표 재계산, 통계 검정, 토이 실험, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2602.13977에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **이 맥에서 논문 수치 재현은 불가능합니다.** 세계모델 RL 학습은 GPU 여러 장이 필요하고, 실로봇 데이터는 공개되지 않았습니다.
- **공식 세계모델 추론 한 번은 조건부입니다.** 가중치 다운로드와 MPS 포팅이 필요하고, 측정 전입니다.
- **축소 재구현은 가능합니다.** 토이 `analysis/toy_kir_depth.py`(numpy, 1초 안팎)로 롤아웃 깊이와 가짜 성공의 관계를 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 통계 검정, 일관성 검사, 토이 결과, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_kir_depth.py` | 토이: 롤아웃 깊이와 가짜 성공 (numpy, 1초 안팎) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_kir_depth.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비배타적 배포 라이선스(`http://arxiv.org/licenses/nonexclusive-distrib/1.0/`)로 공개되어 있습니다. CC 라이선스가 아니므로 논문 그림은 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드는 `github.com/RLinf/RLinf`(Apache-2.0)와 `github.com/youmo445/diffsynth-studio-rlinf`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.

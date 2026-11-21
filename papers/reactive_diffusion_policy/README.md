# Reactive Diffusion Policy (arXiv:2503.02881, RSS 2025) 분석 노트

*Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation* (arXiv v3, 2025-04-23)을 표 재계산과 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2503.02881에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **실로봇 재현(표 II–IV)은 이 맥에서 불가입니다.** Flexiv 로봇 두 대, 촉각 센서, Quest 3가 필요합니다.
- **공식 코드 학습은 조건부입니다.** 설정에 CUDA 장치가 박혀 있고, 데이터와 체크포인트는 Hugging Face에서 받아야 합니다.
- **AT 구조의 작은 재구현은 가능합니다.** 아직 만들지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문 라이선스는 arXiv 비독점(`nonexclusive-distrib/1.0`)입니다. CC BY가 아니므로 논문 그림과 원문 PDF는 이 저장소에 넣지 않았습니다.
- 공식 코드는 `github.com/xiaoxiaoxh/reactive_diffusion_policy`에 있습니다. 최상위 LICENSE는 MIT이며, 같은 파일에 Diffusion Policy와 UMI의 MIT 고지가 함께 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

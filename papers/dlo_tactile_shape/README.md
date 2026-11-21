# DLO 촉각 형상 추정 (arXiv:2611.22990) 분석 노트

*Physics-Informed Tactile-based Shape Estimation of Deformable Linear Objects* (arXiv v1, 2026-11-17)을 표 재계산, 에피소드 수준 검정, 코드 정적 대조로 분석한 자료입니다. 학회 표기는 없습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.22990에서 받을 수 있습니다). 논문 그림은 네 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **이 맥에서 공식 학습은 불가능합니다.** 학습 경로가 CUDA를 전제하고, 데이터가 약 40 GB입니다(추정).
- **추론 파리티 점검은 조건부입니다.** 저장소에 8프레임 픽스처가 있지만 체크포인트(137 MB)는 저장소에 없습니다. Sparsh 인코더는 CC-BY-NC 4.0입니다.
- **자체 재구현은 2–3주 추정입니다(측정 전).**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 네 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 정보 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조(정적 읽기), 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 우리가 그린 그래프 두 장 생성 |
| `report/figures/` | 그래프 두 장(우리가 그림)과 논문 그림 네 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문 라이선스는 arXiv abs 페이지의 license 링크로 확인했습니다(CC BY 4.0). 원문 PDF에서 저작권 문구를 찾지 못했습니다.
- 논문 그림 네 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2611.22990"). 실로봇 사진과 촉각 데이터셋 프레임이 있는 그림 1과 3은 넣지 않았습니다.
- 공식 코드는 `github.com/rotematari/dlofeel-state`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.

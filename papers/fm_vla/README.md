# FM-VLA (arXiv:2607.18231, CoRL 2026 포스터) 분석 노트

*FM-VLA: Force-based Memory for Vision-Language-Action Models in Contact-Rich Manipulation* (arXiv v1, 2026-07-20)을 표·그림 수치 재계산과 주장 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2607.18231에서 받을 수 있습니다). 논문 그림은 싣지 않았고 그래프는 모두 표의 값으로 새로 그렸습니다.

## 구현 가능성 (요약)

- **표 1의 재현은 어렵습니다.** 공식 저장소에 코드 파일이 없고, 양팔 로봇, 시연 750개, A100 8장 규모의 후학습이 필요합니다.
- **힘 기억 모듈의 축소판은 구현할 수 있습니다.** 소형 정책과 합성 접촉 과제로 1–2주로 추정합니다(측정 전). 논문 Stage 1 레시피를 CPU로 그대로 돌리면 약 160–230시간입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사, 합성 toy |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C21, 구현 가능성 판정 |
| `analysis/recompute.py` | 표·그림 수치 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_count_probe.py` | 합성 횟수 프로브 (torch, CPU, 약 2분) |
| `analysis/toy_force_vae_cost.py` | Force-VAE 크기와 CPU 한 스텝 시간 (torch, 1분 이내) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_count_probe.py
python3 analysis/toy_force_vae_cost.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv non-exclusive distribution license로 배포됩니다(arXiv 라이선스 링크 확인, CC BY 아님). 그래서 논문 그림은 싣지 않았습니다.
- 공식 저장소(github.com/qft-333/FM-VLA)는 README와 데모 영상만 있어 코드를 읽지 못했고, 아무것도 실행하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트와 분석 문서는 이 저장소 작성자가 만들었습니다.

# ViTaL (arXiv:2606.14981, CoRL 2026 포스터) 분석 노트

*Inference-time Policy Steering via Vision and Touch* (ViTaL)을 수치 재계산, 통계 검정, 공식 코드 정적 대조로 읽은 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.14981에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **이 맥에서 논문 결과를 재현할 수 없습니다.** 실로봇 실험, CUDA 전제 학습, 시각 보상 서버, 기본 정책 가중치가 필요하고 공개되지 않았거나 이 환경에서 돌릴 수 없습니다.
- **알고리즘 구조는 소형 토이로 옮길 수 있습니다.** 새 설치 없이 numpy로 가능합니다(계획 단계).
- **재계산은 `analysis/recompute.py`로 수 초에 다시 할 수 있습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그래프 두 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조 (정적 읽기), 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표와 그림 수치의 재계산과 검정 (numpy, scipy) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC0 1.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). CC BY 4.0이 아닙니다. 그래서 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/yilin-wu98/vital`에 있습니다. 구성 요소별로 MIT(latent-tactile, anytouch-reward)와 Apache-2.0(liquid-tactile-control)입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 데이터셋은 Hugging Face에 CC BY 4.0으로 공개되어 있습니다. 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 재계산 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.

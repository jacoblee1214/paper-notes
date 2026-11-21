# Adaptive Compliance Policy (arXiv:2410.09309) 분석 노트

*Adaptive Compliance Policy: Learning Approximate Compliance for Diffusion Guided Control* (arXiv v2, 2025-03-07, ICRA 2025로 적혀 있으나 공식 목록 미확인)을 표 재계산과 공식 코드 정적 읽기로 검토한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2410.09309에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **실로봇 재현(뒤집기, 닦기)은 이 맥에서 불가능합니다.** UR5e, 힘 센서, C++ 컨트롤러가 필요합니다.
- **공식 학습은 GPU를 전제합니다.** 맥에는 CUDA가 없습니다.
- **맥에서 가능한 것은 레이블 생성 토이와 정적 검산입니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 검정, "50% 이상" 점검 |
| `analysis/04_code_verification.md` | 공식 코드와 논문의 차이 D1–D10 (정적 읽기) |
| `analysis/recompute.py` | 표 재계산과 검정 (표준 라이브러리만 사용) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
python3 analysis/recompute.py
python3 analysis/plot_figures.py   # matplotlib 필요
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 라이선스 링크 확인). PDF에서 저작권 표시는 찾지 못했습니다.
- 논문 그림은 게시하지 않았습니다. 그래프는 논문 표 값으로 우리가 다시 그렸습니다.
- 공식 코드는 `github.com/yifan-hou/adaptive_compliance_policy`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.

# Targeting World Models to Compromise Robot Learning Pipelines (arXiv:2606.09499, CoRL 2026 스포트라이트) 분석 노트

*Targeting World Models to Compromise Robot Learning Pipelines* (Rathbun 외, arXiv v2, 2026-10-07)를 표 재계산, 일관성 검사, 공개 포크 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.09499에서 받을 수 있습니다). 논문 그림은 두 장(그림 4, 5)만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 CUDA가 없는 환경(Apple 실리콘 맥 등)에서 불가능합니다.** Cosmos-Predict2.5 2B 미세조정과 공격 최적화는 CUDA GPU(저자는 96GB급)를 쓰고 데이터와 자산은 공개되지 않았습니다.
- **공개 포크는 텍스트 조건 공격(VPH)만 있고 핵심 PGD 모듈이 빠져 있습니다.** VTH, PPO, r2Dreamer 코드는 없습니다.
- **재계산과 일관성 검사는 완료했습니다.** 작은 행동 조건 월드 모델에서 같은 위협 모델의 축소판을 만드는 것은 CPU로 가능해 보이지만 구현하지 않았고 노력 추정은 측정 전입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 자료 표 |
| `analysis/03_analysis.md` | 재계산(V), 일관성 검사(K), 주장 판정(C) |
| `analysis/04_code_verification.md` | 공개 포크 정적 대조(D), 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/recompute_results.json` | 재계산 결과 |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시는 찾지 못했습니다.
- 논문 그림 두 장(그림 4, 5)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.09499"). 사진, 렌더 프레임, 생성 영상, 제3자 아이콘이 있는 그림은 넣지 않았습니다.
- 코드는 논문이 링크하지 않았고, 저자 계정으로 보이는 공개 포크 `github.com/EthanRath/cosmos-predict2.5`(Apache-2.0, `gamma-branch` 커밋 `8a924ea`)를 정적으로만 읽었습니다. 이 저장소에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

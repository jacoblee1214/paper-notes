# IRASim (arXiv:2406.14540, ICCV 2025) 분석 노트

*IRASim: A Fine-Grained World Model for Robot Manipulation* (arXiv v2, 2025-07-29)을 공식 코드의 정적 대조, 표 재계산, 통계 검정으로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2406.14540에서 받을 수 있습니다). 논문 그림은 세 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **학습과 정책·계획 실험은 이 맥에서 불가입니다.** 데이터셋당 약 2,370 GPU 시간이 필요하고, 정책 평가와 계획 실험은 공식 코드가 없습니다.
- **아키텍처 재구현과 추론 검사는 부분적으로 가능합니다.** 공식 코드는 CUDA를 전제하므로 MPS 포팅이 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 버전 차이, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조 (차이 D1–D17), 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/crop_figures.py` | 논문 그림 세 장을 원문 PDF에서 자르는 스크립트 (pymupdf) |
| `report/figures/` | 그래프 네 장 (우리가 그림)과 논문 그림 세 장 (© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다 (arXiv 페이지의 "view license" 링크 확인). 원문 PDF에는 ©나 회사 저작권 표시가 없습니다.
- 논문 그림 세 장은 저작자 표기와 함께 실었습니다 ("© 저자, CC BY 4.0, 출처 arXiv:2406.14540"). 데이터셋 프레임과 사진이 있는 패널은 뺐습니다.
- 공식 코드는 `github.com/bytedance/IRASim`에 있으며 Apache-2.0입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

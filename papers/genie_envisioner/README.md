# Genie Envisioner (arXiv:2508.05635, ICLR 2026) 분석 노트

*Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation*를 arXiv v3 기준으로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2508.05635에서 받을 수 있습니다). 논문 그림은 Figure 8 한 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **전체 재현은 이 맥에서 불가능합니다.** 학습은 A100 수십 장과 수 일이 필요하고, 실로봇은 필요합니다.
- **추론 일부는 조건부입니다.** LTX 2B 추론은 MPS에서 bf16을 쓸 수 없어 포팅이 필요하고 메모리가 빠듯합니다.
- **표 재계산과 일관성 검사는 완료했습니다.** `analysis/recompute.py`는 수 초 걸립니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 한 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 권리 점검 |
| `analysis/04_code_verification.md` | 공식 코드 대조 (D1–D15), 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장과 논문 Figure 8 (© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다 (arXiv abs 페이지의 license 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 Figure 8은 저작자 표기와 함께 실었습니다 ("© 저자, CC BY 4.0, 출처 arXiv:2508.05635").
- 공식 코드는 `github.com/AgibotTech/Genie-Envisioner-V1`에 있습니다. README는 CC BY-NC-SA 4.0을 적고 LICENSE 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

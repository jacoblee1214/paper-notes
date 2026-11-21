# What Makes Video World Model Latents Action-Relevant (arXiv:2606.07687, CoRL 2026 포스터) 분석 노트

Jewon Yeom 외 (서울대학교)의 논문을 원자료 CSV 재계산, 공식 코드 정적 대조, 논문 내부 일관성 검사로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.07687에서 받을 수 있습니다). 논문 그림은 두 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **학습과 폐루프 재현은 이 맥에서 불가입니다.** 매칭 대조와 DIFF 학습은 CUDA 기본 설정이고, π0.5 폐루프는 GPU와 시뮬레이터가 필요합니다.
- **프리즈 프로브는 조건부입니다.** LIBERO 데이터와 V-JEPA 2 체크포인트를 받고 특징을 추출하면 가능할 수 있습니다(MPS 속도 미측정).
- **재계산과 정적 대조는 완료했습니다.** `analysis/recompute.py`는 공식 저장소의 결과 CSV를 읽어 표 2, 표 4, 표 10, 표 20을 다시 계산합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장과 우리 그래프 두 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, v1과 v2 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 관련 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 표 재계산과 검정 (numpy, scipy; 공식 저장소 CSV 필요) |
| `analysis/plot_figures.py` | 우리가 그린 그래프 두 장 |
| `report/figures/` | 우리가 그린 그래프 두 장과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

```bash
git clone --depth 1 https://github.com/k1seul/corl2026-action-relevant-latents repo
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 license 링크 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 논문 그림 두 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.07687"). 이 그림들은 그래프와 도식이며 로봇 사진이나 데이터셋 프레임이 없습니다. 데이터셋 프레임이 있는 폐루프 도식은 넣지 않았습니다.
- 공식 코드는 `github.com/k1seul/corl2026-action-relevant-latents`에 있으며 MIT 라이선스입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

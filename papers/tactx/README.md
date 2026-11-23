# TACTX (arXiv:2606.31236, CoRL 2026 포스터) 분석 노트

*TACTX: Learning Shared Tactile Representations Across Diverse Sensors* (arXiv v1, 2026-06-30)를 표 재계산, 공식 코드 정적 대조, 합성 센서 토이로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2606.31236에서 받을 수 있고, 논문 라이선스가 arXiv 비독점 배포입니다). 그래프는 이 저장소 작성자가 그렸습니다.

## 구현 가능성 (요약)

- **실로봇 정책 평가(표 1, 2, 8)는 이 맥에서 불가입니다.** Franka, GELLO, 촉각 센서 세 종과 배치 코드가 필요합니다.
- **표현 학습은 쌍 데이터(2,670 궤적)와 인코더 체크포인트가 공개되어 있지 않아 공개 자료만으로 이어 가지 못합니다.**
- **표 재계산과 합성 센서 토이는 가능하고 완료했습니다.** `analysis/recompute.py`(수 초), `analysis/toy_transitive.py`(torch CPU, 2–3분 안팎).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그래프 세 장) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 데이터 표 |
| `analysis/03_analysis.md` | 재계산 V1–V10, 일관성 검사 K1–K14, 토이, 주장 판정 C1–C13 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조 D1–D17, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 수치 재계산과 검정 (numpy, scipy) |
| `analysis/toy_transitive.py` | 토이: 합성 센서 쌍 정렬 (torch CPU) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_transitive.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(`arxiv.org/licenses/nonexclusive-distrib/1.0/`)입니다. CC BY가 아니므로 논문 그림은 넣지 않았습니다. 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다.
- 공식 코드는 `github.com/tactx-project/TactX`에 있고 LICENSE 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다. 분석한 커밋은 `9381ee0d1912695003945436bf3d2f4f8361e672`입니다.
- 정렬 손실이 의존하는 SimCLR(MIT)의 `nt_xent.py`는 고정 커밋의 텍스트를 읽었습니다.
- 데이터(Hugging Face `TactX/TactX`, 약 130 GB)와 가중치는 받지 않았습니다. 표 수치와 카드의 메타데이터만 확인했습니다.
- 분석 문서, 스크립트, 그래프, 토이는 이 저장소 작성자가 만들었습니다.

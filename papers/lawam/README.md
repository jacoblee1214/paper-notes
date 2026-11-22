# LaWAM (arXiv:2606.15768, CoRL 2026 포스터) 분석 노트

*LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies* (arXiv v2, 2026-10-25)를 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다 (arXiv:2606.15768에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **사전학습, 후학습, 실로봇은 이 맥에서 불가입니다.** 수천 H100-시간과 CUDA, 비공개 데이터 혼합이 전제입니다.
- **공개 체크포인트 추론은 조건부입니다.** 약 10 GB 가중치와 CUDA 의존 대체가 필요합니다.
- **재계산과 구조 토이는 완료했습니다.** `analysis/recompute.py`, `analysis/toy_key_order_invariance.py` (각 수 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 판본 차이, 방법, 실험, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 불일치, 토이, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 구현 가능성, 이슈 보고 |
| `analysis/recompute.py` | 표 재계산과 검정 (표준 라이브러리) |
| `analysis/toy_key_order_invariance.py` | 구조 토이 (numpy) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/recompute.py   # 표 5 파싱에는 arXiv 원문의 pdftotext 결과(paper/paper.txt)가 필요합니다
python3 analysis/toy_key_order_invariance.py
python3 analysis/plot_figures.py   # 같은 paper.txt 필요
```

## 출처와 라이선스

- 논문은 arXiv에 비독점 배포 라이선스로 공개되어 있어 그림을 옮기지 않았습니다. 표 값은 인용과 재계산을 위해 옮겼습니다.
- 공식 코드는 `github.com/RLinf/LaWAM`에 있습니다 (LICENSE 파일 없음, pyproject는 MIT). 이 저장소에는 넣지 않았고 실행하거나 설치하지 않았습니다.
- 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

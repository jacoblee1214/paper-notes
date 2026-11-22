# ImageWAM (arXiv:2606.19531, CoRL 2026 포스터) 분석 노트

*ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?* (arXiv v1, 2026-06-17)를 표 재계산, 쌍체 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2606.19531에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **논문 수치 재현은 이 맥(Apple M4 Pro, MPS, CUDA 없음)에서 불가입니다.** 공식 학습은 8 GPU 기준이고 RoboTwin 평가와 실로봇은 GPU 시뮬레이터와 비공개 코드가 필요합니다.
- **공개 체크포인트로 LIBERO 추론은 조건부입니다.** 약 25 GB 다운로드, 새 설치, MPS 포팅이 필요합니다.
- **표 재계산과 어텐션 마스크 규칙 확인은 가능하고 끝냈습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계 |
| `analysis/02_prior_work.md` | 학회·라이선스 확인, 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/table12_data.json`, `fastwam_table3_data.json` | 논문 표 12와 Fast-WAM 논문 표 3의 과제별 값(옮긴 것) |
| `analysis/mask_check.py` | 공식 코드의 어텐션 마스크 규칙 확인 (numpy) |
| `analysis/plot_figures.py` | 그래프 다섯 장 생성 |
| `report/figures/` | 우리가 그린 그래프 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/mask_check.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문의 arXiv 라이선스는 비독점 배포 라이선스(CC 아님)입니다. 그래서 논문 그림은 넣지 않고 우리가 그린 그래프만 넣었습니다.
- 공식 코드는 `github.com/yuyangalin/ImageWAM`(MIT)에 있습니다. 이 자료에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 자료 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

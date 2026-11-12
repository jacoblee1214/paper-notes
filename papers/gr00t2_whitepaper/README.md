# GR00T 2 백서 (NVIDIA, 2026-11) 분석 노트

*GR00T 2: Scaling Robot Foundation Models for Generalization, Dexterity, and Adaptation*
(NVIDIA 기술 백서, arXiv 없음, 동료 심사 없음)를 표 재계산, 통계 검정, 공개 메타데이터 대조, toy 계산으로 정리한 자료입니다.
원문 확인일은 2026-11-12입니다.

- 백서: https://research.nvidia.com/labs/gear/gr00t-2/assets/gr00t2_whitepaper.pdf (저작권 표기 "© 2026 NVIDIA. All rights reserved.")
- 프로젝트 페이지: https://research.nvidia.com/labs/gear/gr00t-2/ (코드 공개 "December" 예정)
- 공개 메타데이터: GitHub NVIDIA/Isaac-GR00T, Hugging Face `nvidia/GR00T-N1.7-3B`, `nvidia/PhysicalAI-Human-1M-100K`

백서의 그림과 원문 PDF는 저작권 때문에 이 저장소에 넣지 않았습니다. 그래프는 백서 표의 수치로 직접 그렸습니다.
모델 가중치는 받지 않았고, 어떤 코드도 실행하지 않았습니다(공개 메타데이터와 N1.7 설정 파일만 읽음).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 초록에서 실험까지, 방법과 데이터, 한계 |
| `analysis/02_prior_work.md` | GR00T 계보, 구성 요소의 출처, GitHub·Hugging Face 표, N1.7 설정 대조 |
| `analysis/03_analysis.md` | 표 재계산, 통계 검정, 산술 점검, toy 결과, 맥 지연 추정 |
| `analysis/04_claim_verification.md` | 주장 검증(C1–C22), 구현 가능성 판정 |
| `analysis/paper_checks.py` | 표와 본문 수치의 재계산과 검정 → `paper_checks_results.json` (수 초) |
| `analysis/toy_power_fit.py` | 시도 수별 검출력 몬테카를로, 2B 가중치 메모리 → `toy_power_fit_results.json` (1초 미만) |
| `analysis/fetch_meta.py` | GitHub·Hugging Face·arXiv 메타데이터 수집 → `analysis/meta/` (가중치 없음) |
| `analysis/plot_figures.py` | 직접 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/paper_checks.py
OMP_NUM_THREADS=1 python3 analysis/toy_power_fit.py
(cd analysis && python3 plot_figures.py)
python3 analysis/fetch_meta.py   # 선택: 메타데이터 갱신 (네트워크 필요)
```

## 출처와 라이선스

- 백서는 "© 2026 NVIDIA. All rights reserved."이며 CC 라이선스가 없습니다. 그래서 백서 그림과 원문 PDF를 넣지 않았습니다.
- 공개 config는 `nvidia/GR00T-N1.7-3B`의 `config.json`(라이선스: nvidia-open-model-license)입니다. 다른 모델의 구조를 비교하는 데만 씁니다.
- 분석 문서와 스크립트, 그래프는 이 저장소 작성자가 만들었습니다.

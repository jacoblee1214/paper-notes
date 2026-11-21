# BBF (arXiv:2305.19452, ICML 2023) 분석 노트

*Bigger, Better, Faster: Human-level Atari with human-level efficiency* (arXiv v3, 2023-11-13, ICML 2023 camera-ready)을 공개 실행별 점수로 다시 계산하고, 공식 코드를 정적으로 대조한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2305.19452에서 받을 수 있습니다). 그래프는 우리가 공개 점수로 그렸습니다.

## 구현 가능성 (요약)

- **논문 주 결과 재현(리플레이 비율 8, 26게임 × 50시드)은 CUDA 없는 맥에서 불가능합니다.** 한 실행이 A100 절반으로 10시간이라고 논문이 말합니다.
- **공식 JAX 코드는 2022년 무렵의 고정 버전을 요구합니다.** 한 게임 한 시드 실행은 환경 설치 후 조건부로 가능할 수 있습니다. 실행하지 않았습니다.
- **재계산과 정적 대조는 완료했습니다.** `analysis/recompute.py`는 수 초, numpy와 scipy만 씁니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 결과, 버전 차이, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 설정 차이, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json`, `table_a1.csv` | 표와 점수 파일 재계산, 부트스트랩 CI, 절제 검정 |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

공식 저장소의 실행별 점수 파일(`scores/`)이 필요합니다. 이 저장소에는 넣지 않았습니다.

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-research/google-research.git repo/gr
cd repo/gr && git sparse-checkout set bigger_better_faster && cd ../..
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문 라이선스는 arXiv 비독점 배포 라이선스 1.0입니다(abs 페이지의 "view license" 링크). CC BY가 아닙니다. 그래서 논문 그림은 넣지 않았습니다.
- PDF에 회사 저작권 표시는 없습니다. "Copyright 2023 by the author(s)"(PMLR 표준 문구)만 있습니다.
- 학회 표기는 PMLR 202 공식 프로시딩(`proceedings.mlr.press/v202/schwarzer23a.html`)에서 확인했습니다.
- 공식 코드는 google-research 모노레포(Apache-2.0)에 있습니다. 하위 폴더에는 LICENSE 파일이 없습니다. 이 저장소에는 코드와 점수 파일을 복사하지 않았습니다.
- 표 A.1 수치는 논문에서 옮겨 적었습니다. 그래프, 재계산 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.

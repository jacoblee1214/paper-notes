# OpenWAM (arXiv:2609.07398) 분석 노트

*OpenWAM: An Open, Modular Exploration Towards Systematic World–Action Model Pretraining*
(Wang, Huang, Li, Zhang et al., NUS · 칭화대 · 베이징대 · 홍콩대 외, 2026)을 논문 표 재계산, noise 평면 toy,
[공식 코드](https://github.com/OpenWAM-Official/OpenWAM)(`a00c4ba`)와 Hugging Face 메타데이터 대조로 분석한 자료입니다.
학습·평가 재현(GPU)은 하지 않았고, 가중치와 데이터셋은 받지 않았습니다.

**구현 가능성:** 평가와 SFT는 공개 코드·체크포인트로 CUDA GPU에서 재현할 수 있습니다. 사전학습은 사람 영상(30%)과 그 로더가
공개되지 않아 재현할 수 없습니다. 24 GiB 통합 메모리 맥에서는 학습이 불가능하고, 추론도 수정 없이는 안 됩니다(`analysis/04_code_verification.md` §4).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, 공개 저장소·체크포인트·데이터셋 표 |
| `analysis/03_analysis.md` | noise 평면 유도, Fig. 9 toy, 효과 크기와 잡음, 데이터·연산 산술, 벤치마크 표 읽기 |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조(D1–D14), 수치 재계산(C1–C10, V1–V2), 구현 가능성 |
| `analysis/table_check.py` | 표·본문 수치 재계산 |
| `analysis/noise_toy.py` | 학습 분포와 Fig. 9 경로 17개의 Gaussian 혼합 toy |
| `analysis/plot_figures.py` | 리포트 그래프 네 장 |
| `analysis/data/robodojo_per_seed.json` | 공식 저장소 `assets/robodojo_verification/`의 RoboDojo 시드별 결과(Apache-2.0) |
| `analysis/data/param_counts.json` | 공개 사전학습 체크포인트 safetensors 헤더로 센 파라미터 수 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/noise_toy.py
python3 analysis/plot_figures.py
```

모두 CPU 단일 스레드로 약 1분 안에 끝납니다. 논문 PDF, 공식 코드, 가중치는 필요하지 않습니다(수치는 스크립트에 출처와 함께 적었습니다).

## 출처와 라이선스

- 논문은 **CC BY 4.0**입니다(arXiv abs 페이지 확인, PDF 안에 다른 권리 고지 없음). `report/figures/paper_fig*.png`는 논문에서 잘라낸 그림이며 © 저자, CC BY 4.0, 출처 arXiv:2609.07398입니다.
- `analysis/data/robodojo_per_seed.json`은 공식 저장소(Apache-2.0)의 파일을 그대로 옮겼습니다.
- 나머지 그래프는 논문 수치, 공식 코드의 식, 위 결과 파일로 직접 그렸습니다.
- 공식 코드와 원문 PDF는 이 저장소에 넣지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

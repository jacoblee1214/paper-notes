# EagleVLA (arXiv:2607.12659 v7, CoRL 2026 스포트라이트) 분석 노트

*EagleVLA: Towards Onboard Real-Time Robot Control via Foresight-Aligned Asynchronous Inference*를 표 재계산, 공식 저장소의 정적 읽기, arXiv 버전 비교로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다.

## 판정 (요약)

- **구조는 구현할 수 있습니다.** 보정 모듈과 스케줄러는 PyTorch로 옮길 만합니다(추정 1–2주, 학습 제외).
- **학습, 평가, 지연 측정은 CUDA 환경이 필요합니다.** 실로봇 재현은 장비와 데이터가 필요합니다.
- **헤드라인 배율(Orin 14.71×)은 세 번 바뀌었습니다.** 최종 행은 엔진 저장소의 최적화 구성 전체의 측정값이고, Thor 구성은 정확도 게이트를 통과하지 못했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림과 저장소 내부 참조 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계, 버전과 내부 특이점 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 표 재계산, 격자 검사, 근사 검정, 지연 산술, 버전 차이 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조(D1–D15), 저자에게 물을 질문, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy). 공식 README가 있으면 표 복사본도 대조 |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py      # 공식 저장소 README를 repo/EagleVLA/에 클론했을 때만 표 복사본 대조가 돌아감
python3 analysis/plot_figures.py
```

공식 저장소는 `git clone --depth 1 https://github.com/PKU-SEC-Lab/EagleVLA.git repo/EagleVLA` 로 받습니다(실행하지 않음).

## 출처와 라이선스

- 논문은 CC BY 4.0입니다(arXiv 초록 페이지의 라이선스 링크 확인, 2026-11-22). 이 자료에는 논문 그림과 원문 PDF를 넣지 않았습니다. 원문은 arXiv:2607.12659에서 받을 수 있습니다.
- 공식 저장소 EagleVLA는 Apache-2.0, EagleVLA-Edge는 MIT입니다. 두 저장소 모두 정적으로만 읽었고 실행하지 않았습니다. 체크포인트와 데이터는 받지 않았습니다.
- 논문 표와 본문의 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

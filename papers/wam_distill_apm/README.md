# APM (arXiv:2611.22592) 분석 노트

*Distilling World Action Models for Efficient Bimanual Manipulation via Anchored Path Matching* (arXiv v1, 2026-11-17, 학회 표기 없음)을 원문 확인, 공식 코드 정적 대조, 표 재계산과 통계 검정으로 분석한 자료입니다. 재현 실험과 학습은 하지 않았습니다.

## 구현 가능성 (요약)

- **논문 결과 재현은 이 맥에서 불가능합니다.** 학습은 GPU, 백본, 데이터가 필요하고, 학습 설정(옵티마이저, 학습률, λ_a)이 공개되지 않았습니다. 실로봇과 시뮬레이터 평가도 불가능합니다.
- **CPU 수준의 확인은 가능합니다.** 투영 수식 검사와 목표 형태 토이 비교가 후보입니다.
- **중요한 발견:** 논문 식 (7)–(8)의 짧은 전이 목표와 공식 코드의 순간 방향장 목표가 다릅니다. 저장소 README도 이를 밝힙니다(`analysis/04_code_verification.md` D1).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전과 공개 정보 차이, 비판적 메모 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안 표 |
| `analysis/03_analysis.md` | 표 재계산, 독립 근사 검정, 검정력, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조, 차이 D1–D8, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/figures/` | 그래프 세 장(우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문 원문은 이 저장소에 넣지 않았습니다. `curl -L -o paper/paper.pdf https://arxiv.org/pdf/2611.22592v1`로 받을 수 있습니다.
- arXiv 라이선스 페이지는 비독점 배포(`arxiv.org/licenses/nonexclusive-distrib/1.0/`)입니다. 그래서 논문 그림은 싣지 않았습니다. 그래프는 이 저장소 작성자가 표 값을 옮겨 그렸습니다.
- 공식 코드는 `github.com/JIAlonglong/Anchored-Path-Matching`(Apache-2.0)입니다. 이 저장소에 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다.

# UniIntervene (arXiv:2606.12372, CoRL 2026 포스터) 분석 노트

*UniIntervene: Agentic Intervention for Efficient Real-World Reinforcement Learning* (arXiv v1, 2026-06-10)을 표 재계산, 통계 검정, 공식 코드 정적 대조, numpy 토이로 분석한 자료입니다. 논문 원문 PDF와 공식 코드는 넣지 않았습니다(arXiv:2606.12372, github.com/Denghaoyuan123/UniIntervene에서 받을 수 있습니다). 논문은 arXiv 비독점 배포 라이선스라서 논문 그림은 넣지 않았고, 우리가 그린 그래프만 넣었습니다.

## 구현 가능성 (요약)

- **실로봇 평가(UR7e, 5과제)는 이 맥에서 불가입니다.** 하드웨어와 HIL-SERL 통합 코드가 공식 저장소에 없습니다.
- **공식 학습 파이프라인은 사실상 불가입니다.** 입력 데이터와 체크포인트가 없고 코드가 CUDA를 전제합니다.
- **라벨 규칙과 트리거 로직의 소형 재구현은 가능합니다.** `analysis/toy_tvr_labels.py`(numpy, 1초 안팎)로 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V17, 토이, 주장 판정 C1–C11 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D18, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_tvr_labels.py` | 토이: TVR 트리거와 오프라인 라벨 규칙 (numpy, 1초 안팎) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (먼저 토이를 실행) |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_tvr_labels.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv:2606.12372. arXiv 라이선스 페이지가 비독점 배포 라이선스(nonexclusive-distrib 1.0)를 가리킵니다. CC BY가 아닙니다. 논문 그림은 싣지 않았습니다.
- 학회: CoRL 2026 포스터(corl.org 채택 목록 확인).
- 공식 코드는 `github.com/Denghaoyuan123/UniIntervene`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 정적으로 읽기만 했으며 실행이나 설치를 하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

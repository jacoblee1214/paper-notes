# UniVLA: Learning to Act Anywhere with Task-centric Latent Actions (arXiv:2505.06111, RSS 2025) 분석 노트

*UniVLA* (arXiv v3, 2025-11-03)의 표를 다시 계산하고, 본문과 표를 맞춰 보고, 공식 코드를 정적으로 대조한 자료입니다. 재현 실험은 하지 않았습니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 그래프 두 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 사전학습은 32 A100에 960 A100-hours, 미세조정은 8 GPU가 필요하고, 공식 평가 코드는 CUDA를 직접 호출합니다.
- 공개 체크포인트의 LIBERO 추론 확인은 조건부로 가능합니다(모델 다운로드와 MPS 포팅 필요, 측정 전).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 결과, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, 학회와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산, 본문-표 일관성 검사, 통계 검정 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 불일치, 구현 가능성, 저자에게 물을 질문 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-21). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 학회: RSS 2025 (roboticsconference.org 2025 채택 목록과 온라인 proceedings로 확인).
- 공식 코드(https://github.com/OpenDriveLab/UniVLA)는 Apache-2.0 라이선스입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

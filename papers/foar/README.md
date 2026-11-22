# FoAR (arXiv:2411.15753, IEEE RA-L 2025) 분석 노트

*FoAR: Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation* (arXiv v2, 2025-05-02)을 표 재계산, 통계 검정, 공식 코드 정적 대조, 구현 가능성 판정으로 읽은 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2411.15753에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다(CC BY-NC-SA 4.0). 그래프는 모두 우리가 만들었습니다. 공식 코드는 넣지 않았습니다(LICENSE 파일 없음).

## 구현 가능성 (요약)

- **논문 재현은 이 맥에서 불가능합니다.** 실로봇 평가는 저장소 밖 장비가 필요하고, 공식 학습은 CUDA와 NCCL을 전제합니다. 체크포인트도 공개되어 있지 않습니다.
- **구조의 PyTorch 재구현과 반응 제어 로직 검사는 가능합니다.** 점군 인코더 대체가 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (공개용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, v1과 v2의 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소의 출처, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 보정 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드와의 대조, 불일치 목록, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/figures/` | 우리가 만든 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY-NC-SA 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/Alan-Heoooh/FoAR`에 있습니다. LICENSE 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트와 분석 문서는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터는 받지 않았습니다.

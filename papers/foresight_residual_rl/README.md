# Foresight Residual RL (arXiv:2607.16506) 분석 노트

*Foresight Residual RL for Long-Horizon Robot Manipulation with Vision-Language-Action Models*
(Liu, Zhang, Liu, Boularias, IROS 2026 채택)을 표 재계산, 곱 검사, 3단계 체인 toy로 분석한 자료입니다.
공식 코드는 없습니다(2026-11-10 확인). 저장소는 프로젝트 페이지 템플릿이고, 프로젝트 페이지는 404입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림·표 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 식, 실험 설정, 빈칸 |
| `analysis/02_prior_work.md` | 선행연구, 가까운 대안, 관련 GitHub 저장소 |
| `analysis/03_analysis.md` | 표 재계산과 유의성, 곱셈 보상 유도, toy 결과 |
| `analysis/04_claim_verification.md` | 주장별 판정, 빠진 정보, 권리 표기, 구현 가능성 |
| `analysis/table_check.py` | 표 재계산 (수 초) |
| `analysis/shaping_toy.py` | 3단계 체인 MDP toy: 곱셈 보상, PBRS, 희소 보상, 추정기 오차 (약 70초, 1스레드) |
| `analysis/extra_chain_sweep.py` | 3–5단계 체인에서 핸드오프 격차와 예측기 표본 수를 바꾼 foresight 이득 스캔 (numpy, 수 분, 1스레드) |
| `analysis/plot_figures.py` | 직접 만든 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/shaping_toy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림, 논문 표 이미지, 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 표의 수치와 직접 만든 toy로 그렸습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

# Geometric Entropy (arXiv:2606.20871) 분석 노트

*Geometric Entropy: When Trajectory Diversity Helps and Hurts in Imitation Learning*
(Luo, Liu, Zhou, Zhou, Yang; IROS 2026 채택)을 재계산, 그림 판독 검정, 합성 궤적 toy로 분석한 자료입니다.
공식 코드는 찾지 못했습니다(2026-11-10 확인).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림·표 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Results → Limitations, 프로젝트 페이지와의 차이 |
| `analysis/02_prior_work.md` | 선행연구 계보, 기준선, 관련 GitHub 저장소 |
| `analysis/03_analysis.md` | 표 재계산, 그림 판독 검정, H_G 유도, toy 결과 |
| `analysis/04_claim_verification.md` | 주장별 판정, 빠진 정보, 재현 가능성, 구현 가능성 판정 |
| `analysis/entropy_toy.py` | 합성 전이 궤적에 H_G 적용: r·k 의존, 거리, N 편향, 노이즈, T 민감도 |
| `analysis/extra_hg_replicates.py` | 순서·N 편향·T 민감도를 시드 10개로 반복 (numpy 1스레드, 약 20초; `entropy_toy.py` 함수를 가져다 씀) |
| `analysis/bc_toy.py` | 2차원 BC toy: 커버리지와 모드 평균화 |
| `analysis/stats_check.py` | 판독 값의 검정(2차항, 피크, Fisher, Holm, CMH) |
| `analysis/plot_figures.py` | 직접 만든 그래프 |

## 재실행 (CPU, 1스레드)

```bash
pip install numpy scipy matplotlib
python3 analysis/entropy_toy.py && python3 analysis/extra_hg_replicates.py && python3 analysis/bc_toy.py && python3 analysis/stats_check.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림, 논문 표 이미지, 원문 PDF를 넣지 않았습니다.**
- 그래프는 직접 만든 toy 결과로 그렸습니다. 논문 수치는 본문 인용과 그림에서 판독한 근사 값(±0.01–0.02)으로만 씁니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

# Motion-Focused Latent Action (arXiv:2606.18955) 분석 노트

*Motion-Focused Latent Action Enables Cross-Embodiment VLA Training from Human EgoVideos*
(Xu, Zhang, Wang, Wang, Yu; IROS 2026 채택)을 표 재계산, 통계 검정, 공개 코드 대조, 잠재 행동 토이로 분석한 자료입니다.
공식 코드는 GitHub `shuiyigong/Motion-Focused-Latent-Action`(Apache-2.0, 커밋 `fd614f7`)에 있습니다. 이 저장소에는 코드 사본을 넣지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 분석: 방법, 실험, 비판적 해석, v1 → v2 차이 |
| `analysis/02_prior_work.md` | 선행연구 계보, 가까운 대안, GitHub 표 |
| `analysis/03_analysis.md` | 표 재계산, 유의성(Fisher, CMH, Holm, Wilson), 검출력, 토이 결과 |
| `analysis/04_code_verification.md` | 논문과 공개 코드 대조(D1–D27), 주장 검증(C1–C14), 구현 가능성 |
| `analysis/table_check.py` | 표 재계산과 검정 (1초 미만) |
| `analysis/motion_toy.py` | 잠재 행동 토이: 마스크 누수, 자르기 불일치, 혼잡도 대 판독 (수 초) |
| `analysis/plot_figures.py` | 직접 만든 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/motion_toy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 표의 수치(`libero_wilson.png`, `robotwin_tasks.png`)와 직접 만든 토이(`toy_leakage.png`, `toy_k_sweep.png`)로 그렸습니다.
- 공식 코드는 Apache-2.0이며 이 저장소에 복사하지 않았습니다. 코드 대조는 공개 커밋 `fd614f7` 기준입니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

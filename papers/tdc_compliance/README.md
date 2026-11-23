# TDC (arXiv:2602.14174) 분석 노트

*TDC: Sim-to-Real Transferable Directional Compliance for Contact-Rich Manipulation* (Yang et al., CoRL 2026)을
논문 표·그림 값의 재계산과 어드미턴스 제어기의 1축 numpy toy로 분석한 자료입니다. 공식 코드는 찾지 못했습니다(2026-11-24 확인).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계, v1과 v2 차이 |
| `analysis/02_prior_work.md` | 선행연구 계보, 관련 저장소 표 |
| `analysis/03_analysis.md` | 표·그림 재계산(V), 내부 일관성(K), toy(T) |
| `analysis/04_claim_verification.md` | 주장별 판정(C), 공개 E2VLA 저장소 부분 대조(D), 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (수 초) |
| `analysis/toy_admittance.py` | 어드미턴스 제어기 1축 toy (1초 미만) |
| `analysis/plot_figures.py` | 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_admittance.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 폴더에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 표의 수치와 직접 만든 toy로 그렸고, Fig. 5와 Fig. 6의 값은 그림에서 읽은 값입니다.
- 분석 문서와 스크립트는 이 폴더의 작성자가 만들었습니다.

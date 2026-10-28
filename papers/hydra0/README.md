# Hydra-0 (arXiv:2608.18077) 분석 노트

*Hydra-0: Action Flow for Generalist World Modeling and Control* (Li et al., NVIDIA 외, 2026)을
논문과 [프로젝트 페이지](https://nvidia-isaac.github.io/video_to_data/hydra-0/)의 공개 수치를 대조해 분석한 자료입니다. 모델 코드는 아직 공개되지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보 |
| `analysis/03_analysis.md` | 헤드라인 분해, 4-step 결과, Gaussian 전파 폭, RoboLab 수치 정합성 |
| `analysis/04_claim_verification.md` | 수치 정합성과 주장별 근거 검증 (C1–C8) |
| `analysis/table_check.py` | 재계산 스크립트 (논문 PDF와 프로젝트 페이지 HTML이 `paper/`에 있어야 함) |

## 재실행

```bash
pip install matplotlib
mkdir -p paper && curl -L -o paper/paper.pdf https://arxiv.org/pdf/2608.18077 && pdftotext -layout paper/paper.pdf paper/paper.txt
curl -L -o paper/project_page.html https://nvidia-isaac.github.io/video_to_data/hydra-0/
python3 analysis/table_check.py && python3 analysis/plot_headline.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스, © 2026 NVIDIA). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- `report/figures/headline_epe.png`는 논문 Table 2의 수치로 직접 그린 그래프입니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

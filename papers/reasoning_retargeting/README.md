# Retargeting Through Reasoning (블로그, 2026) 분석 노트

프로젝트 블로그 *Retargeting Through Reasoning for Humanoid–Object Interaction* (2026-11-06 게시, 2026-11-10 접근)를
블로그 원자료의 수치로 재계산하고 통계 검정과 numpy toy로 읽은 자료입니다. 심사를 거친 논문이 아닙니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 공개용 리포트 (블로그 그림은 넣지 않음) |
| `analysis/01_paper_analysis.md` | 블로그의 주장, 방법, 평가, 결과 |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 라이선스, GitHub 저장소 표 |
| `analysis/03_analysis.md` | 수치 재계산, McNemar 검정, 선택 편향 유도, toy 결과 |
| `analysis/04_code_verification.md` | 저장소 내용, 블로그와 원자료의 차이, 주장 검증, 구현 가능성 |
| `analysis/recompute_blog.py` | 원자료 재계산 (결과 `recompute_results.json`) |
| `analysis/refinement_toy.py` | 피드백 제안기 toy와 선택 편향 toy (결과 `refinement_toy_results.json`) |
| `analysis/plot_figures.py` | 그림 생성 |
| `report/reasoning_retargeting.pptx` | 발표 자료 12장 |

## 재실행

```bash
pip install numpy matplotlib
git clone --depth 1 https://github.com/narayananpp/semantic-humanoid-retargeting-blog repo   # 커밋 84cbe2253
python3 analysis/recompute_blog.py && python3 analysis/refinement_toy.py && python3 analysis/plot_figures.py
```

원자료 저장소에는 라이선스 파일이 없습니다. 원자료는 이 패키지에 넣지 않았고, 수치만 요약했습니다.
상위 모션 데이터는 CC BY-NC-SA 4.0 조건이 붙는다고 원자료 공지에 적혀 있습니다.

## 출처와 라이선스

- 블로그 그림과 영상은 라이선스가 없어 싣지 않았습니다. 그림은 모두 재계산 값으로 직접 그렸습니다.
- HOI-Retarget(arXiv:2609.34674, CC BY 4.0), SPIDER(코드 CC BY-NC 4.0), SONIC(arXiv CC BY-SA 4.0)은 그 저장소의 조건을 따릅니다.
- 이 패키지의 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

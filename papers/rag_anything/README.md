# RAG-Anything (arXiv:2510.12323) 분석 노트

*RAG-Anything: All-in-One RAG Framework* (Guo, Ren, Xu, Zhang, Huang, The University of Hong Kong, 2025)를
논문 표 재계산, 그림 픽셀 판독, [공식 코드](https://github.com/HKUDS/RAG-Anything)(HEAD `2cd2b33`, 논문 시점 `806ac9b`) 대조로 분석한 자료입니다.
외부 API 호출이나 재현 실험은 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, 관련 GitHub 표 |
| `analysis/03_analysis.md` | Overall 가중 평균 검산, 유형 분해, Fig. 2 판독, 표본 오차, 그래프 연결 구조, 판정 fallback, 비용 |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조(D1–D10), 수치 재계산(C1–C9), 구현 가능성 |
| `analysis/table_check.py` | 표·그림 수치 재계산 |
| `analysis/code_logic_toy.py` | 판정 fallback 규칙과 그래프 병합 규칙의 CPU 확인 |
| `analysis/plot_figures.py` | 리포트 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/code_logic_toy.py
cd analysis && python3 plot_figures.py
```

모두 CPU에서 몇 초 안에 끝나고, API 키나 원문 PDF가 필요하지 않습니다(수치는 스크립트에 출처와 함께 적었습니다).

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- `report/figures/*.png`는 논문 표의 수치와 Fig. 2 판독값으로 직접 그린 그래프입니다.
- 공식 코드(MIT)는 이 저장소에 넣지 않았습니다. 인용한 행 번호는 위 커밋 기준입니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

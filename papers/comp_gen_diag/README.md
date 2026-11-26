# Diagnosing Compositional Generalization (arXiv:2607.29687, CoRL 2026 스포트라이트) 분석 노트

*Diagnosing Compositional Generalization in Sequential Robot Tasks* (arXiv v2, 2026-10-15)를 저자가 공개한 집계 파일로 다시 계산하고 공식 코드를 정적으로 대조한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(원문은 arXiv:2607.29687에서 받을 수 있고, arXiv 비독점 배포 라이선스라서 그림을 옮기지 않았습니다).

## 구현 가능성 (요약)

- **논문 전체 재현은 이 맥(Apple M4 Pro, CUDA 없음)에서 불가입니다.** 데이터 약 42.6 GB, 참조 환경이 CUDA GPU와 Linux, 사전학습 가중치 없음.
- **집계 파일 기반 재계산과 공식 코드 정적 대조는 끝냈습니다.** `analysis/recompute.py`(수 초).
- **공식 파이프라인 스모크 테스트는 조건부**(새 설치, 4.7 GB 다운로드, 공식 코드 실행)이고, 구조 토이는 새 설치 없이 구현할 수 있습니다. 둘 다 이 노트에서는 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 이론, 실험, 버전 차이(v1/v2) |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V14, 설계 검사 K1–K4, 주장 판정 C1–C16 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조 D1–D24, 라이선스, 구현 가능성 |
| `analysis/fetch_page_data.py` | 저자 프로젝트 페이지 저장소의 집계 파일 받기 (이 노트에는 넣지 않음) |
| `analysis/recompute.py` | 재계산, 검정, 설계 검사 (numpy, scipy, 수 초) |
| `analysis/repo_static_stats.py` | 공식 저장소 프리셋 설정값 표 (정적 읽기) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/fetch_page_data.py   # data/page/ 에 집계 파일을 받음 (네트워크)
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

`analysis/repo_static_stats.py`는 공식 저장소를 `repo/`에 얕게 복제해야 합니다(`git clone --depth 1 https://github.com/YixiaoWang7/Diagnosing_Compositional_Generalization_Robot repo` 후 커밋 `e29c44474495b341d19636bbf90f3cf0fcb1d2d2`로 체크아웃). 저장소 코드는 실행하거나 import하지 않고 설정 JSON만 읽습니다.

## 출처와 라이선스

- 논문 라이선스: arXiv abs 페이지의 라이선스 링크가 arXiv 비독점 배포 라이선스(`nonexclusive-distrib/1.0`)를 가리킵니다. CC BY가 아니라서 논문 그림은 옮기지 않았습니다.
- 공식 코드(`github.com/YixiaoWang7/Diagnosing_Compositional_Generalization_Robot`)에는 라이선스 파일이 없습니다. 이 노트에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 수치 재계산에 쓴 집계 파일은 저자의 프로젝트 페이지 저장소(`github.com/YixiaoWang7/Diagnosing_Compositional_Generalization_Robot_Page`)에서 내려받아 쓰며, 이 노트에 다시 배포하지 않습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

# AutoDex (arXiv:2606.23689, CoRL 2026) 분석 노트

*AutoDex: An Automated Real-World System for Dexterous Grasping Data Collection* (Choi et al., Seoul National University, CoRL 2026 채택)를 표 재계산, 통계 검정, 공식 코드 정적 대조, numpy 토이 두 개로 분석한 자료입니다. 실물 재현은 하지 않았고, 아래 판정 근거는 `analysis/04_code_verification.md` §4에 있습니다.

## 구현 가능성 (판정)

- **실물 재현: 불가능.** 20대 카메라 워크셀, 6축 로봇 팔, 다지 손, CUDA GPU, 비공개 `paradex` 라이브러리가 필요합니다.
- **공식 코드 그대로: 불가능.** 저장소에 `paradex`가 없고, 추적 서브모듈이 비어 있으며, 작성자 환경의 절대 경로가 박혀 있습니다. 라이선스 파일도 없습니다.
- **아이디어 재구현: 중간 난이도.** 수집 루프와 검색은 하드웨어 없이 2–4주(추정), 실물 루프는 수 개월입니다.

## 파일

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 결과 요약 |
| `analysis/02_prior_work.md` | 선행연구, 관련 GitHub 저장소 표, v1 → v2 차이 |
| `analysis/03_analysis.md` | 표 재계산, 손별·검증 효과 검정, 시간 예산, 토이 두 개 |
| `analysis/04_code_verification.md` | 공식 저장소 대조(D1–D20), 구현 가능성 판정, 권리 표기 |
| `analysis/table_check.py` | 표 재계산과 통계 (1초 미만) |
| `analysis/label_noise_toy.py` | 토이 A: 자동 라벨 오차와 성공 분류기 (numpy 1스레드, 1초 미만) |
| `analysis/collection_toy.py` | 토이 B: 리셋 실패와 처리량, 선택 편향 (numpy 1스레드, 1초 미만) |
| `analysis/plot_figures.py` | 리포트 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/label_noise_toy.py
python3 analysis/collection_toy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.** 그래프는 논문 본문과 표의 수치, 그리고 직접 만든 토이로 그렸습니다.
- 공식 코드 저장소(snuvclab/autodex-code)에는 라이선스 파일이 없습니다. 클론하지 않았고, 정적 대조 결과만 `04_code_verification.md`에 있습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

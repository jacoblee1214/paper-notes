# TactSpace (arXiv:2606.18959v1, IROS 2026 채택) 분석 노트

TactSpace 논문(시뮬 촉각과 실측 정전용량의 공유 잠재 정렬)을 표 재계산, 공개 코드 대조, numpy 토이로 분석한 자료입니다.
논문 라이선스는 CC BY-NC-ND 4.0이라 논문 그림과 표 이미지는 넣지 않았습니다. 그래프는 직접 그렸습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 분석 (방법, 데이터 짝짓기, 표 대조) |
| `analysis/02_prior_work.md` | 선행연구, 관련 GitHub 저장소 표 |
| `analysis/03_analysis.md` | 헤드라인 재계산, 유효 표본 검정, 주장 대조 C1–C8, 토이 결과, 유도 |
| `analysis/04_code_verification.md` | 공개 코드 대조 D1–D18, 구현 가능성 |
| `analysis/table_check.py` | 표 재계산 (1초 미만) |
| `analysis/align_toy.py` | 공유 잠재 토이 (numpy 1스레드, 약 14초) |
| `analysis/extra_penalty_depth.py` | 추가 검증: penalty 방식 침투 깊이 분포의 numpy 재구현 (numpy, 1스레드, 약 2초) |
| `analysis/extra_pair_count.py` | 추가 검증: 자체 시뮬 쌍 수 곡선, 소형 ViT 정렬과 무정렬 인코더 (torch CPU, 약 5분) |
| `analysis/plot_figures.py` | 직접 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/table_check.py
python3 analysis/align_toy.py
python3 analysis/extra_pair_count.py
python3 analysis/plot_figures.py
```

## 저장소 상태 (2026-11-10 확인)

- `leggedrobotics/TactSpace` (메타 저장소, 공개, 라이선스 없음)만 접근했습니다. 시뮬레이터와 정렬 저장소는 GitHub API에서 404였습니다.
- 공개된 FEM 스크립트는 입력 생성과 출력 추출을 담고 있습니다. 실행 줄은 주석 처리되어 있습니다.

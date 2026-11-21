# π0.5: a Vision-Language-Action Model with Open-World Generalization (arXiv:2504.16054, CoRL 2025) 분석 노트

π0.5 (Physical Intelligence, arXiv v1, CoRL 2025 PMLR v305)를 논문 구조 정리, 수치 재계산, 공식 코드(openpi) 정적 대조로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 두 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 학습·재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 혼합 데이터(모바일 약 400시간 등)와 학습 스텝 규모, 실로봇 평가가 필요합니다.
- **공개 코드는 flow-only입니다.** 논문의 FAST 토큰 혼합 손실과 고수준 텍스트 추론 경로는 공개 저장소에서 확인되지 않았습니다.
- 가능한 것: 정적 대조(완료), CPU toy 실험, 공개 체크포인트의 추론 지연 측정(다운로드와 설치 필요).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 학습 레시피, 실험, 한계, 라이선스 |
| `analysis/02_prior_work.md` | 학회 판정, 공식 코드 목록, 선행연구 계보 |
| `analysis/03_analysis.md` | 수치 재계산과 일관성 검사 (그림 판독 기반) |
| `analysis/04_code_verification.md` | openpi와 논문의 대조, 불일치와 미공개 항목 |
| `analysis/recompute.py`, `recompute_results.json` | 재계산 스크립트와 결과 (CPU, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-21). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드 openpi(https://github.com/Physical-Intelligence/openpi, Apache-2.0)는 얕게 받아 정적으로만 읽었습니다. 실행, 설치, 가중치 다운로드는 하지 않았습니다.
- 그래프 두 장과 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다. 그래프 값은 논문 그림을 눈으로 읽은 근사치입니다.

# Action Influence Maximization (arXiv:2610.27387, CoRL 2026 포스터) 분석 노트

*Training Controllable World Models via Action Influence Maximization* (arXiv v1, 2026-10-19)을 표 재계산, 정리 유도 검토, 작은 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2610.27387에서 받을 수 있습니다). 논문 그림은 세 장(그림 3, 5, 6)만 출처와 함께 넣었습니다.

## 구현 가능성 (요약)

- **논문 규모 재현은 불가입니다.** GPU가 필요한 세계모델 규모, 전문가 300k 에피소드, 사람 평가가 필요하고 학습 코드가 공개되지 않았습니다(2026-11-26 기준).
- **재계산과 정리 토이는 완료했습니다.** `analysis/toy_theory.py`(약 70초, numpy)로 정리 1의 적용 범위와 정리 2의 표 형태 확인을 했습니다.
- **Four Rooms 규모의 직접 구현은 가능**(노력 추정 2–4주, 측정 아님)하고, Push-T 상태 기반 축소는 조건부입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산(V), 일관성(K), 토이(T) |
| `analysis/04_claim_verification.md` | 주장 판정 C1–C16, 구현 가능성 |
| `analysis/recompute.py` | 표 1·그림 3·그림 5 재계산 (numpy, scipy) |
| `analysis/toy_theory.py` | 토이: 정리 1의 범위, 정리 2 표 형태 |
| `analysis/plot_figures.py` | 우리가 그린 그래프 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 세 장(© 저자, CC BY 4.0) |

## 재실행

`recompute.py`와 `plot_figures.py`는 두 가지 입력이 필요합니다. 논문 텍스트는 `curl -L -o paper/paper.pdf https://arxiv.org/pdf/2610.27387v1`로 받은 뒤 `pdftotext -layout paper/paper.pdf paper/paper.txt`로 만들고, 공식 저장소는 `git clone --depth 1 https://github.com/princeton-prism/action-influence-maximization repo`로 받은 뒤 `git -C repo checkout 11e0da0a0b43ddb4c02f0140d4804cdc65c69101`을 실행합니다. 저장소의 코드는 실행하지 않고 `docs/static/data/`의 JSON만 읽습니다.

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_theory.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시는 찾지 못했습니다.
- 논문 그림 세 장은 저자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2610.27387"). 시뮬레이션 화면이 들어간 그림은 넣지 않았습니다.
- 공식 저장소(`github.com/princeton-prism/action-influence-maximization`)에는 라이선스 파일이 없고, 이 폴더에는 코드를 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 노트 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

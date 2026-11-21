# TD-MPC2 (arXiv:2310.16828, ICLR 2024 Spotlight) 분석 노트

*TD-MPC2: Scalable, Robust World Models for Continuous Control* (arXiv v2, 2024-03-21)을 공식 저장소의 학습 곡선 파일로 다시 계산하고, 공식 코드와 대조해 읽은 자료입니다. 논문 원문 PDF는 넣지 않았습니다. 아래 명령으로 받을 수 있습니다.

## 구현 가능성 (요약)

- **전체 재현은 이 맥(Apple M4 Pro, CUDA 없음)에서 불가능합니다.** 공식 코드가 CUDA를 전제하고, 80과제 317M 학습은 저자 기준 33 GPU일이 필요합니다.
- **단일 과제 축소 실행은 조건부입니다.** 포팅과 패키지 설치, 체크포인트 다운로드가 필요합니다.
- **SimNorm, two-hot, 잠재 계획의 개념 재구현은 가능합니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림은 넣지 않음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이, 학회 표기 |
| `analysis/02_prior_work.md` | 선행 계보와 공식 저장소 표 |
| `analysis/03_analysis.md` | 공개 학습 곡선 재계산, 짝 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조 (D 항목), 라이선스, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
git clone --depth 1 https://github.com/nicklashansen/tdmpc2 repo     # 공식 결과 파일 (정적 읽기만 함)
curl -L -o paper/paper.pdf https://arxiv.org/pdf/2310.16828v2       # 과제 목록 확인용 (선택)
python3 -I analysis/recompute.py
python3 -I analysis/plot_figures.py
```

`paper/paper.pdf`가 없으면 `recompute.py`는 과제 목록 확인 단계에서 멈춥니다. 나머지 계산은 `repo/results/`만 있으면 됩니다.

## 출처와 라이선스

- 논문은 arXiv abs 페이지의 license 링크가 `http://creativecommons.org/licenses/by/4.0/`(CC BY 4.0)을 가리킵니다. PDF에서 저작권 표시는 찾지 못했습니다.
- 공식 코드는 `github.com/nicklashansen/tdmpc2`에 있고 MIT입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 작성자가 만들었습니다.
- MuJoCo 라이선스 키(`tdmpc2.com/files/mjkey.txt`)는 받지 않았습니다.
- 체크포인트와 데이터셋은 받지 않았습니다.

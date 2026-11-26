# RoboFollow: Unveiling the Instruction Following Mirage in Embodied Agents (arXiv:2609.25636, CoRL 2026) 분석 노트

*RoboFollow* (arXiv v1, 2026-09-22)를 표 재계산과 공식 코드의 정적 읽기로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문은 CC BY 4.0이지만 그림이 모두 시뮬레이터 화면이나 실로봇 사진입니다). 이 폴더의 그래프 다섯 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 공식 저장소의 전제는 Linux, CUDA GPU, SAPIEN, cuRobo이고, 모델 미세조정과 5,540회 평가가 GPU를 요구합니다. 데이터셋은 약 40.4 GB입니다(받지 않음).
- **표 재계산과 점수 규칙의 정적 대조는 새 설치 없이 끝냈습니다.** 자작 소형 환경에서 같은 현상을 보는 것은 조건부로 가능합니다(노력 1–2주 추정).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 벤치마크 설계, 표 1–3, 주장과 증거의 틈, 판본 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(Avg 블록, CR 정수 격자, 언어 없는 상한, 빈 언어 대조, 표 2·3, 실로봇, QA), 주장 판정 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조(D1–D17), 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 다섯 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

`recompute.py`는 공식 저장소(`git clone https://github.com/AutoLab-SAI-SJTU/RoboFollow repo && git -C repo checkout f642e3d19ddc0a3b0fb5795eef2cd17a12f286d8`)가 `repo/`에 있으면 프로젝트 사이트용 결과 CSV와 옮겨 적은 표 1을 대조하고, 없으면 그 단계만 건너뜁니다.

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0(`http://creativecommons.org/licenses/by/4.0/`)으로 공개되어 있습니다(abs 페이지 확인, 2026-11-26). 이 폴더에는 원문과 논문 그림을 넣지 않았습니다.
- 공식 코드(https://github.com/AutoLab-SAI-SJTU/RoboFollow)는 GitHub가 MIT로 표시합니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

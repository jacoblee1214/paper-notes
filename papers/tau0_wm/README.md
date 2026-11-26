# τ0-WM: A Unified Video-Action World Model for Robotic Manipulation (arXiv:2606.01027, CoRL 2026) 분석 노트

*τ0-WM* (arXiv v2, 2026-08-23)을 표 재계산, 공식 코드 정적 대조, 후보 선택 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 CC BY-NC-SA 4.0). 이 폴더의 그래프 여섯 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 사전학습 데이터(로봇 17.8K 시간)와 사전학습 코드가 없고, 사후학습은 CUDA 다중 GPU, 평가는 로봇 세 종이 필요합니다.
- 공식 가중치 추론은 다운로드, 설치, 포팅이 필요한 조건부이고, 재계산, 파라미터 계산, 후보 선택 토이는 CPU로 가능합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 판본·권리·학회, 방법, 실험 수치, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표·그림 재계산 V1–V12, 검정, 파라미터, 토이 T1 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조 D1–D20, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/param_count.py` | 공개 코드 구조로 파라미터 수를 해석적으로 계산 (`hf_file_sizes.json` 입력) |
| `analysis/toy_rcs_selection.py` | 토이: RCS 후보 선택 (torch, 약 40초) |
| `analysis/plot_figures.py` | 그래프 여섯 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/param_count.py
python3 analysis/toy_rcs_selection.py
python3 analysis/plot_figures.py
```

`param_count.py`는 코드를 읽지 않고 설정 값을 옮겨 계산하므로 공식 저장소가 없어도 돕니다. 공식 저장소(커밋 `e03902a4a8e0cf89b681e89920784ae8ca7b4b3c`)는 `git clone --depth 1 https://github.com/sii-research/tau-0-wm repo`로 받아 대조할 수 있습니다.

## 출처와 라이선스

- 논문은 arXiv에 CC BY-NC-SA 4.0으로 공개되어 있습니다(abs 페이지 확인, 2026-11-26). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/sii-research/tau-0-wm)와 가중치(https://huggingface.co/sii-research/tau-0-wm)는 실행하거나 복사하지 않고 정적으로 읽거나 파일 크기만 조회했습니다. 프로젝트 페이지는 https://tau0-wm.github.io/ 입니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

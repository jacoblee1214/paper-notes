# FingerEye: Learning Dexterous Manipulation with Continuous Vision-Tactile Sensing (arXiv:2604.20689, CoRL 2026) 분석 노트

*FingerEye* (arXiv v4, 2026-10-06)를 표 재계산, 공식 코드 정적 대조, 작은 numpy·torch 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 데모 데이터가 공개되지 않았고, 시뮬은 Isaac Sim과 NVIDIA GPU, 실세계는 xArm7, LEAP Hand, FingerEye 센서가 필요합니다.
- **정책 학습 자체는 가볍습니다.** 캐시된 RADIO 요약 위의 52.7 M 파라미터 정책은 한 스텝(배치 256)이 MPS에서 0.18초(무작위 입력 측정)입니다.
- **새 설치 없이 가능한 것:** 표 재계산, 그룹 디코더 구조 토이, 태그 배열 정밀도 계산.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈(C1–C7), 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산과 검정(V1–V12), 일관성 검사(K1–K7) |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조(D1–D14), 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산, 실로봇 Fisher·CMH·Holm, 시뮬 격자·Welch, 감도 산술 (numpy, scipy, 1초 미만) |
| `analysis/toy_group_decoder.py` | 단일 토큰 그룹의 교차 어텐션 상수성과 변형별 파라미터 수 (torch, CPU, 몇 초) |
| `analysis/toy_tag_crlb.py` | 35태그 배열의 PnP 정밀도 하한 (numpy, 1초 미만). 입력 `analysis/tag_layout_mm.json` |
| `analysis/bench_policy_step.py` | 정책 한 스텝 시간 측정 (torch, 무작위 입력, MPS가 있으면 약 20초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_tag_crlb.py
python3 analysis/plot_figures.py
pip install torch                       # 아래 두 스크립트에만 필요
python3 analysis/toy_group_decoder.py
python3 analysis/bench_policy_step.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-23). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/ariszxxu/FingerEye)에는 LICENSE 파일이 없습니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다. `analysis/tag_layout_mm.json`은 공식 설정 파일에 적힌 태그 중심 좌표 35개(물리 배치의 사실 정보)를 옮긴 것입니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

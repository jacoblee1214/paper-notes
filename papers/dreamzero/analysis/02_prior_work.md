# 02. 선행연구, 가까운 대안, 공개 자료

## 1. 계보

DreamZero는 세 줄기가 만난 자리입니다. 줄기마다 논문이 인용한 대표작을 적었습니다(참고문헌 번호는 원문 기준).

| 줄기 | 대표작 (원문 인용) | DreamZero와의 차이 |
|---|---|---|
| VLA: 사전학습 VLM에 행동 헤드 | RT-2 계열(Brohan 2023), π0·π0.5(Black 2024; Physical Intelligence 2025 [72]), GR00T N1.5·N1.6(Bjorck 2025 [9]), OpenVLA(Kim 2024) | 의미 일반화는 같지만, 행동을 영상 예측과 함께 학습하지 않음. 논문은 VLA가 "새 동작"에서 약하다고 주장(서론) |
| 영상 생성 → 로봇 정책 (분리형) | UniPi(Du 2023 [24]), Learning universal policies via text-guided video generation, Ko 2024 [55] (광학 흐름), Dreamitate(Liang 2024 [62]) | 영상 생성 후 역동역학 모델로 행동을 꺼냄. 논문은 두 단계를 한 모델로 합침(§3.1) |
| 영상·행동 결합 (WAM) | GR-1·GR-2(Cheang 2024 [17]), UVA(Li 2025 [60]), Video Prediction Policy(Hu 2024 [39]), Video generators are robot policies(Liang 2025 [63]), mimicvideo(Pai 2025 [71]), Genie Envisioner(Liao 2025 [64]), Causal World Modeling(Li 2026 [59]), Cosmos Policy(Kim 2026 [54]) | 대부분 영상 사전학습 백본 위에서 짧은 과제 데이터로 학습. DreamZero는 14B, 반복 없는 500시간, 자기회귀 구조, 추론 최적화를 함께 다룸 |
| 자기회귀 영상 diffusion | CA2-VDM(Gao 2024 [28]), Self-Forcing(Huang 2025 [43]), MAGI-1(Teng 2025 [77]), Yin 2025(본문 인용, 서지 확인 안 함), 흐름 정합(Lipman 2022), Wan2.1 I2V-14B(Team Wan 2025 [76]) | DreamZero는 이 구조를 로봇 행동에 맞춤(청크, 관측을 캐시에 넣기) |
| 확산 가속 | TeaCache(Liu 2024 [66]), TaylorSeer(Liu 2025), Flow UniPC(Zhao 2023), CFG 병렬, 단계 수 축소 | 논문은 DiT 캐시 규칙을 자체 설계로 씀(부록 D.1). 공식 코드의 규칙과 다름(`04` D3) |

**중심 질문.** 이 계보에서 DreamZero가 답하려는 질문은 "영상 생성 능력을 행동에 옮기는 데 필요한 것은 데이터 다양성인가, 모델 크기인가, 구조인가?"입니다. 논문의 답은 다양한 데이터와 14B, 자기회귀 구조입니다(표 4).

## 2. 가까운 대안 (같은 문제, 다른 선택)

| 대안 | 구조 | 행동 추출 | 공개 여부 (본문 기준) | 비교 포인트 |
|---|---|---|---|---|
| Cosmos Policy (Kim 2026 [54]) | 영상 백본 위 정책 미세 조정 | 같은 모델 안에서 행동 생성 (본문은 정밀 조작에서 WAM 이점이 있다고 인용, §6) | 본문 인용만 확인 | 정밀도 주장과 비교 대상 |
| Video generators are robot policies (Liang 2025 [63]) | 영상 생성기를 정책으로 쓰는 방식 | 역동역학 또는 공동 생성 | 본문 인용만 확인 | 사전학습 규모와 데이터 다양성 비교 |
| mimicvideo (Pai 2025 [71]) | 영상 모델 + 역동역학(분리형) | 분리 | 본문 인용만 확인 | 논문 §3.1이 "두 모델을 따로 두는" 방식으로 인용 |
| UVA / Causal World Modeling (Li 2025, 2026 [59–60]) | 영상과 행동 동시 예측 | 공동 | 본문 인용만 확인 | 자기회귀 대 양방향 비교 (논문 부록 B) |
| Video Prediction Policy (Hu 2024 [39]) | 영상 예측 표현 위 정책 | 중간 표현 | 본문 인용만 확인 | 영상 표현 재사용 방식 |
| VLA (GR00T N1.6, π0.5) | 사전학습 VLM + 행동 확산 헤드 | 직접 | GR00T N1.6과 π0.5 공식 체크포인트 공개 (본문) | 논문 기준 비교 대상. DROID에서만 공식 체크포인트 사용 |



## 4. 코드·가중치·데이터 표

2026-11-15 확인. "정적 읽기" 는 얕은 복제 후 파일만 읽었다는 뜻이고, 실행과 설치와 가중치 다운로드는 하지 않았습니다. 용량은 HuggingFace API의 파일 목록 합계입니다.

| 항목 | 위치 | 라이선스 (확인값) | 상태 |
|---|---|---|---|
| 논문 | arXiv:2602.15922 (v1, 2026-02-17) | CC BY 4.0 (abs 페이지 링크 확인). PDF에 "© 2026 NVIDIA. All rights reserved." 표기 | 원문 `paper/paper.pdf` (9.1 MB, 25 MB 미만이라 gitignore하지 않음) |
| 공식 코드 | github.com/dreamzero0/dreamzero, HEAD `ab790c1` (2026-04-19) | Apache-2.0 (LICENSE), 저작권 NVIDIA (COPYRIGHT) | `repo/`에 얕은 복제. 정적 읽기만 (`04`) |
| DreamZero-DROID 가중치 (14B) | huggingface.co/GEAR-Dreams/DreamZero-DROID | cc-by-nc-4.0 (모델 카드). 비상업 | 다운로드 안 함. 파일 합계 약 64.8 GB |
| DreamZero-AgiBot 가중치 (사전학습, 후학습 출발점) | huggingface.co/GEAR-Dreams/DreamZero-AgiBot | apache-2.0 | 다운로드 안 함. 파일 합계 약 45.9 GB. `config.json`의 action_horizon = 48 |
| DROID 전처리 데이터 | huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data | 확인 안 함 | 다운로드 안 함 (README에 있음) |
| 기반 모델 Wan2.1-I2V-14B-480P | huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P | apache-2.0 (API 확인) | README 기준 약 28 GB, 다운로드 안 함 |
| 토크나이저 umt5-xxl | huggingface.co/google/umt5-xxl | 확인 안 함 | 다운로드 안 함 |
| AgiBot 사전학습 데이터 (약 500시간) | 공개 안 됨 (논문 "plan to open-source") | - | 없음. 변환 스크립트 `scripts/data/convert_agibot.py`만 있음 |
| YAM 데이터, 사람 영상 데이터 | 공개 안 됨 | - | 없음 |
| 프로젝트 페이지 | dreamzero0.github.io | - | 본문과 링크만 확인, 내용 미검증 |
| 평가 | DROID 시뮬레이션은 호스팅 API 접근 신청 필요 (`sim_evals`). PolaRiS, Genie Sim 3.0, RoboArena 연동은 README에 언급 | - | 실행하지 않음 |

## 5. 출처와 이 폴더의 한계

- 선행연구 주장은 논문 인용 번호를 따랐습니다. 인용된 논문 대부분은 이 폴더에서 읽지 않았습니다(본문 인용만 확인). 특히 Cosmos Policy, mimicvideo, Genie Envisioner, Causal World Modeling, UVA는 검토하지 않았습니다. 다음 단계에서 필요할 때 PDF를 받아 대조해야 합니다.
- 공개 자료 목록은 2026-11-15 기준입니다. 공개 웹 페이지는 바뀔 수 있습니다.

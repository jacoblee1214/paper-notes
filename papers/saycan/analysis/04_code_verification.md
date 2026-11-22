# 04. 공식 코드 대조 (정적 읽기)

*SayCan* (arXiv:2204.01691 v2)의 공식 코드는 **google-research 저장소의 `saycan/` 폴더, 노트북 한 개** (`SayCan-Robot-Pick-Place.ipynb`, 41개 코드 셀, 저장된 출력 0개)입니다. 논문의 PaLM 540B·모바일 매니퓰레이터 실험 코드는 공개되지 않았습니다. 이 문서는 **노트북과 논문의 차이**를 기록합니다(D1–D17). 읽은 대상은 `saycan/repo/saycan/`(얕은 희소 복제, 커밋 a0080d3, 2026-04-28 최근 변경)입니다. **노트북을 실행하거나, 패키지를 설치하거나, 외부 파일을 받지 않았습니다.** 분석일 2026-11-22.

## 1. 노트북 구조 (셀 번호 0부터)

| 셀 | 내용 |
|---|---|
| 0–1 | Apache-2.0 헤더 (SPDX) |
| 2 | 설명: GPT-3 (InstructGPT), CLIP ViT-B/32, ViLD, CLIPort 변형 (Transporter Nets) |
| 3 | `openai_api_key = ""`, `ENGINE = "text-davinci-002"` |
| 5 | pip 설치 (ftfy, CLIP, flax 0.5.3, pybullet, moviepy, openai, …), gdown으로 UR5e·Robotiq 자산과 ViLD 체크포인트(`gs://cloud-tpu-checkpoints/detect`) |
| 7–11 | PyBullet 환경: UR5e + Robotiq 2F-85. 피킹 대상 블록 4개, 배치 대상 13개 (블록 4, 그릇 4, 고정 위치 5). 픽셀 크기와 작업 공간 |
| 19–27 | 스크립트 전문가 (`ScriptedPolicy`), 사전 생성 데이터 `dataset-9999.pkl`(약 10K 샘플, gdown) |
| 28–34 | CLIPort 변형 모델 (JAX/Flax, InfoNCE 학습), 체크포인트 `ckpt_40000`(gdown) 로드 |
| 37 | `run_cliport`: 텍스트 → 피킹·배치 히트맵 |
| 39 | LLM 캐시 `overwrite_cache = True` |
| 41–42 | `gpt3_call`, `gpt3_scoring`: 옵션 문자열의 토큰 로그확률 합 |
| 43–44 | `normalize_scores`, `affordance_scoring` (ViLD 탐지 기반 0/1) |
| 45 | 한 번의 선택 테스트 |
| 48 | 프롬프트 `gpt3_context` (예시 6개) |
| 49–51 | 실행 루프 (`max_tasks = 5`), `only_plan` 옵션 |
| 53–55 | Socratic Models 스타일 변형 (ViLD 장면 설명을 프롬프트에 넣는 실행, 셀 55의 `env_description = ''`) |

## 2. 논문과 노트북의 차이

| 번호 | 항목 | 논문 (v2) | 공식 노트북 | 영향 |
|---|---|---|---|---|
| D1 | 로봇·장면 | Everyday Robots 모바일 매니퓰레이터, 7자유도 팔, 551 기술, 객체 15개, 위치 5개, 사무실 주방 | PyBullet UR5e + Robotiq 2F-85, 블록과 그릇 수는 실행 설정마다 다름 (셀 11은 피킹 3·배치 3, 셀 49는 피킹 4·배치 1), 고정 위치 5 | 실험 환경이 전혀 다름 (노트북은 탁상 시뮬레이션) |
| D2 | LLM | PaLM 540B (v2), FLAN 137B (v1), 8B·62B 생성 ablation | 기본 `text-davinci-002` (GPT-3 InstructGPT, OpenAI API 유료) | 헤드라인 LLM이 아님. API 키 필요 |
| D3 | 가용성(affordance) 형식 | RL 가치함수 Q(s, ℓ) → v_min·v_max로 보정한 연속값 (피킹), 거리 기반 (이동) | 객체 탐지(ViLD) 결과에 **0 또는 1** (셀 44). 탐지된 블록과 그릇이 둘 다 있으면 1 | 연속 확률이 아니라 **0/1 게이트**. 탐지가 빠지면 옵션이 0이 됨 (D16) |
| D4 | 종료 가용성 | 0.1 (D.2) | 0.2 (셀 44) | 종료가 더 자주 선택될 수 있음 |
| D5 | 배치 가용성 | 1.0 고정 (계획기가 배치를 수행) | 목적지가 탐지되거나 고정 위치이면 1 (셀 44의 `list(PLACE_TARGETS.keys())[-5:]` 추가) | 배치 기술의 의미가 다름 |
| D6 | 가용성 갱신 | 매 스텝 현재 상태 s_n에서 계산 (Algorithm 1), 완료 기술은 상한 (D.2) | **루프 밖에서 초기 장면으로 한 번 계산** (셀 51, `affordance_scores = affordance_scoring(...)` 가 while 루프 앞). 완료 기술 마스킹 없음 | 같은 객체를 두 번 집는 계획이 나올 수 있음. 정적 읽기로 본 것이고 실행하지 않음 |
| D7 | 점수 단위 | 기술 설명 문장 "1. ℓ_π"의 로그확률 (스텝마다 이전 기술을 넣고 재평가) | 완전한 옵션 문자열 `robot.pick_and_place(a, b)` 또는 `done()`의 **토큰 로그확률 합** (셀 41). 길이 보정 없음. `exp`한 뒤 가용성과 곱하고 최댓값으로 정규화 (셀 51) | 순위는 논문의 곱 규칙과 같은 형태 (일치). 길이 편향은 논문에도 기록 없음 |
| D8 | 계획 길이 | "done"이 나올 때까지 제한 없음 | `max_tasks = 5` (셀 51) | 5스텝을 넘는 장기 계획은 잘림 |
| D9 | 프롬프트 | 17개 예시, 대화형 "How would you … / I would: 1. find …, 2. pick up …" | 6개 예시, `robot.pick_and_place` API 형식 (셀 48). `objects = …` 줄은 `use_environment_description = False`일 때 제거 (셀 48 및 51) | 기본 설정에서 LLM은 장면 목록을 보지 못함. 논문 Listing 1도 장면 목록이 없으므로 이 점은 같음 |
| D10 | 저수준 정책 | 피킹: BC-Z 방식 다중 과제 정책 + MT-Opt 방식 RL 가치함수 (TPU 16개, CPU 6,000개, 68,000 시연) | CLIPort 변형 (Transporter Nets + CLIP ViT-B/32), 사전 생성 ~10K 샘플, 체크포인트 40,000 스텝, **RL 없음** (셀 19–37) | 노트북 정책은 논문의 가치함수를 대신하지 않음 |
| D11 | 평가 프로토콜 | 3명 평가자 2 of 3 판정 (계획·실행 분리), 101개 지시, 2개 장면 | 평가 코드, 지시 목록, 평가 결과 없음 | 논문 수치를 노트북으로 재현할 방법이 없음 |
| D12 | 출력 형식 | 본문 §6 "pick up the object and place it in location" (자연어) | `robot.pick_and_place(obj, place)` (코드 형식) | 같은 점수화 방식이라도 입력 분포가 다름 |
| D13 | 실행 요구 | 실로봇 + 학습 인프라 | `.cuda()` 호출 (셀 14, 31, 37 등), TensorFlow 1.x 세션 (셀 17), JAX/Flax (셀 29–34) | GPU 전제. 이 맥(MPS)에서는 수정 없이 실행되지 않음 |
| D14 | 외부 자원 | 논문 본문과 부록에 목록 | `gdown`(Google Drive 파일 여러 개, 셀 5·26·34), `gcloud storage cp`(GCS, 셀 5), OpenAI API 키 (셀 3) | 재현에 외부 계정과 유료 API가 필요 |
| D15 | 라이선스 | arXiv non-exclusive (abs 페이지 "view license" 링크 확인) | 노트북 Apache-2.0 (SPDX 헤더). 저장소 README에는 라이선스 문구 없음 (상위 monorepo Apache-2.0). 프로젝트 저장소 say-can.github.io는 **라이선스 없음** | 노트북 재사용은 Apache-2.0 조건. 웹사이트 자산은 조건 불명 |
| D16 | 탐지 실패 시 동작 (정적 추론) | 가치함수가 연속적이라 한 기술이 낮게 평가되어도 다른 기술이 남음 | 탐지된 객체가 하나도 없으면 피킹 옵션은 전부 0이고, `done()`만 0.2가 남음. 정규화 후 `done()`이 선택되어 루프가 바로 끝남 (셀 51 `while not selected_task == termination_string`) | **계획이 빈 상태로 끝날 수 있음**. 실행하지 않고 코드에서 추론한 것 |
| D17 | 종료 토큰 표기 | "done" | `done()` (셀 48, 51) | 사소함 |

## 3. 일치하는 부분

- **점수 결합의 형태:** 노트북의 `exp(llm) × aff`를 최댓값으로 나누고 argmax를 취하는 것은 논문 식 argmax p_LLM · p_aff와 **순위가 같습니다** (D7). 확률 해석은 논문과 같은 가정(독립, 보정됨)을 씁니다. 노트북은 보정을 하지 않습니다.
- **피킹·배치 실행:** 노트북은 CLIPort 변형으로 "Pick the X and place it on the Y." 문장을 실행합니다(셀 37, 51). 논문 §6의 설명과 같은 구조입니다.
- **장면 정보 생략:** 기본 설정에서 LLM에 장면 목록을 주지 않는 점은 논문 Listing 1과 같습니다(D9).

## 4. 구현 가능성 판정

| 항목 | 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 | 근거 |
|---|---|---|
| 논문 Table 2 수치 재현 (PaLM 540B, 101 지시, 실제 로봇 평가) | **불가** | PaLM 540B 가중치·API 없음, 실로봇·평가 원자료 없음 |
| 공식 노트북 그대로 실행 | **불가** (GPU·유료 API 필요) | `.cuda()` 호출, OpenAI 키, Drive·GCS 자산 (D13, D14). 외부 API 호출은 이 분석 범위 밖 |
| 노트북을 MPS로 옮긴 실행 | **조건부, 노력 큼** | JAX·TF1·CLIP 코드를 PyTorch로 옮기고 CLIPort 체크포인트를 재학습해야 함. 학습은 GPU 전제 |
| SayCan 선택 규칙(Algorithm 1)의 CPU toy | **가능** | numpy만으로 가능. 본 분석은 재현 실험을 하지 않음 (`03` §0) |
| 로컬 소형 LM(1B 이하)으로 옵션 점수화 | **조건부** | 모델 가중치 다운로드와 패키지 설치 승인이 필요. 장면은 탐지 없이 합성 |
| PyBullet 탁상 환경(스크립트 전문가 + 0/1 가용성) | **조건부** | `pybullet` 설치 승인 필요. CPU로 가능하나 학습 없음 |

**노력 추정 (측정하지 않음):** 선택 규칙 toy 1–2일. 로컬 소형 LM + PyBullet 탁상 환경으로 텍스트 계획의 가용성 게이트를 검사하는 것 1–2주. 노트북을 PyTorch·MPS로 옮기는 것 2–4주 (CLIPort 학습에 GPU 필요). 실로봇 재현은 불가.

## 5. 저자에게 물을 것

- 표 2 실제 주방 열의 총계(81%, 60%)와 행 합(79%, 58%)이 맞지 않는 이유 (K1).
- 표 5의 "0예시 무종료 54%"(본문)와 52%(표) 중 어느 쪽이 맞는지 (K4).
- 표 7 드로어의 "21 queries"가 몇 회인지 (K5).
- 노트북의 가용성 갱신 방식(D6)과 완료 기술 상한(D.2)이 논문 PaLM 실험에도 쓰였는지.
- 가치함수 보정 상수 v_min, v_max의 값이 기술마다 다른지.

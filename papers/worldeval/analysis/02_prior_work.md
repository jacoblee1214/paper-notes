# WorldEval 선행연구와 관련 자료 (02)

## 1. 자료 목록: 코드·프로젝트 페이지·관련 저장소

| 구분 | 항목 | 라이선스 | 스타 | 포크 | 마지막 push | 커밋 | 확인 방법 |
|---|---|---|---|---|---|---|---|
| 공식 코드 | [liyaxuanliyaxuan/Worldeval](https://github.com/liyaxuanliyaxuan/Worldeval) | MIT (저자 수정부분) + Apache-2.0 (DiffSynth·Wan 원본) | 31 | 3 | 2025-06-11 | `46084268` (clone, `repo/`) | GitHub API, 얕은 clone, 정적 읽기 |
| 프로젝트 페이지 | [worldeval.github.io](https://worldeval.github.io) | — | — | — | — | — | HTML 확인. 영상 예시 중심이며 수치 표는 없음 |
| 논문 | arXiv:2505.19017 (v1 only) | CC BY 4.0 | — | — | 2025-05-25 제출 | — | arXiv abs·API 확인 |
| 기반 모델 | Wan 2.1 14B I2V (Wan-AI) | Apache-2.0 | — | — | — | — | 논문·README 언급. 가중치 미다운로드 |
| 기반 프레임워크 | DiffSynth-Studio (modelscope) | Apache-2.0 | — | — | — | — | README 명시. 미조회 |
| 정책 가중치 | π0, DexVLA, Diffusion Policy (HuggingFace `kuromivv/…`) | — | — | — | — | — | README 링크만 확인. 미다운로드 |
| 실물 시뮬레이터 비교 | RoboTwin 2.0 (arXiv:2504.13059) | — | — | — | — | — | 논문 언급. 미조회 |
| 시뮬레이터 기법 | SIMPLER (arXiv:2405.05941) | — | — | — | — | — | 논문 언급. 직접 비교는 하지 않음 (논문 §4.2) |

공식 저장소의 README는 "Our code is released!"(2025-05-19)라고 씁니다. 마지막 커밋은 2025-06-11이고, 그 뒤로 업데이트가 없습니다. 저장소에는 학습·추론 코드와 VQVAE·one-hot 인코더가 있고, 평가 파이프라인(Gemini 판정, Pearson·MMRV, 실물 시뮬레이터 실험)은 없습니다. 자세한 대조는 `04_code_verification.md`에 있습니다.

## 2. 선행연구 계보

**(a) 실물 정책 평가의 대안.**
- *SIMPLER* (Li 외, arXiv:2405.05941): 실물 정책을 시뮬레이터에서 평가하는 틀입니다. 실물 시각과 시뮬레이터 사이의 차이를 real-to-sim 기법으로 줄입니다. WorldEval의 주된 비교 대상입니다.
- *AutoEval* (Zhou 외, arXiv:2503.24278): 연구실 안에서 자동으로 실물 평가를 돌리는 체계입니다. WorldEval 논문은 이를 연구실 설정과 단일 로봇에 묶인 방식으로 봅니다.
- *RoboTwin* (Mu 외, arXiv:2504.13059): 쌍팔 로봇의 디지털 트윈 벤치마크입니다. 논문 §4.2에서 WorldEval이 쓰는 로봇과 같은 임베디먼트를 가진 시뮬레이터로 씁니다.

**(b) 행동 표현.** 논문 §2는 행동 표현을 네 갈래로 나눕니다. 이산화(RT-1), 함수 근사, 잠재 공간(VAE), 언어 토큰(RT-2, OpenVLA)입니다. Policy2Vec은 함수 근사 쪽에 속하며, 정책 네트워크 내부 표현을 그대로 씁니다. 같은 역할을 하는 비교군으로 VQVAE(Discrete Policy, arXiv:2409.18707)와 one-hot 이산화를 씁니다.

**(c) 세계모델을 로봇에 쓰는 흐름.**
- *1X-world*: 세계모델을 정책 평가기로 쓰자는 개념을 처음 제안했지만 방법을 보이지 않았다고 논문이 평가합니다(§2).
- *Ctrl-World* (제목 "A Controllable Generative World Model for Robot Manipulation"): 로봇 조작용 생성 세계모델로 보입니다. arXiv 페이지의 관련 목록에서 제목만 확인했고, arXiv ID는 검증하지 않았으며 본문은 읽지 않았습니다. 후속 분석에서 다룰 때 확인이 필요합니다.
- *Video Prediction Policy* (arXiv:2412.14803), *Genie*, *Navigation World Models* 등은 세계모델을 정책 학습에 쓰는 흐름입니다. WorldEval은 세계모델을 정책 **평가**에 쓴다는 점에서 다릅니다.

**(d) 세계모델 평가 지표.** EWMBench(arXiv:2505.09694)는 세계모델 영상의 장면, 움직임, 의미 품질을 평가합니다. WorldEval은 영상 품질이 아니라 **정책 순위의 일치**를 봅니다.

## 4. 이 논문의 위치 (요약)

WorldEval은 "세계모델을 정책 평가기로 쓰자"는 개념을 실물 로봇 데이터로 구체화한 논문입니다. 기여는 두 가지입니다. 정책 내부 잠재 벡터를 조건으로 쓰는 Policy2Vec, 그리고 생성 영상 판정과 순위 지표를 묶은 WorldEval 파이프라인입니다. 실물 시뮬레이터 대비 우위와 행동 인코더 대비 우위를 보고합니다. 판정기 정확도, 정책 수, 학습 데이터 누출 점검은 논문이 다루지 않았습니다.

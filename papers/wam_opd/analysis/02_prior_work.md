# 02. 선행연구와 계보

WAM-OPD의 부품은 새롭지 않습니다. Flash-WAM(Student), LingBot-VA(Teacher), RoboTwin 2.0(평가), LoRA와 pseudo-Huber 회귀(학습)를 가져다 쓰고, **"Student 이력 위에 Teacher의 영상·행동 쌍 목표를 붙여 한 번 학습한다"**는 조합이 기여입니다(§1 기여 문단). 아래 출처는 논문 §2 참고문헌 번호를 따릅니다.

## 1. 계보

| 줄기 | 대표 작업 | WAM-OPD와의 관계 |
|---|---|---|
| 온폴리시 증류 (언어) | GKD, Agarwal 외, ICLR 2024 [37] | Student가 만든 접두에 Teacher 분포를 맞춘다는 원리. WAM-OPD는 "Student가 만든 이력 + Teacher 목표"만 이어받고 분포 발산은 쓰지 않음 |
| 온폴리시 증류 (연속 생성기) | DiffusionOPD [38], Flow-OPD [39] | Student 샘플링 상태에서 Teacher 전이·벡터장을 맞춤. WAM-OPD는 샘플링 경로가 아니라 끝점(영상 토큰, 행동)과 초기 속도 한 점만 맞춤(§3.2) |
| 온폴리시 증류 (로봇) | VLA-OPD [41] (arXiv:2603.26666) | 행동 토큰 reverse-KL. 논문은 "WAM은 영상이 행동을 조건화하는 쌍 목표 문제"라고 구분(§2). VLA-OPD의 외부 반복(수집 → 갱신 → 재수집)과 달리 WAM-OPD는 1회 수집 후 오프라인 학습 |
| 상호작용 모방 | DAgger, Ross 외 [5] | 공분산 이동 동기. WAM-OPD도 같은 동기를 인용하나 반복하지 않음 |
| 온라인 RL | SimpleVLA-RL [6], πRL [7], WAM-RL [31] | 보상으로 갱신. WAM-RL은 "행동 RL + 재구성 보상 + 온라인 영상 SFT"로 WAM에서 영상과 행동을 함께 다룸. 같은 Student에서의 비교는 없음 |
| 세계모델 이용 사후학습 | WorldSample [32], WCM [33], Hi-WM [34], WISE [35] | 상상된 전이·비평가·사람 교정. 비교 실험 없음 |
| WAM | VPP [13], PAD [14], UVA [15], UWM [16], LingBot-VA [17], RoboDreamer [18] | 영상과 행동을 한 모델에서 다룸 |
| WAM 가속 | Flash-WAM [42] (arXiv:2606.05254) | 영상·행동 잡음 영역이 다른 모달리티 인지 일관성 증류. WAM-OPD의 Student 초기값 |

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| Student 1영상·1행동 스텝 | Flash-WAM [42] | 공개 체크포인트 `FlashWAM-RoboTwin`(저장소 README). 저장소 라이선스 표기 NOASSERTION |
| Teacher 25/50 스텝 | LingBot-VA [17] (arXiv:2601.21998, Apache-2.0 코드) | `lingbot-va-posttrain-robotwin` Transformer를 사용(저장소 README) |
| stop-gradient 영상 조건 | 논문 §3.1 | 배포 계산과 학습 계산을 맞추는 설계. 본문에 인과적 이득의 직접 실험은 없음 |
| 속도 항 (σ = 1) | 흐름 매칭 [12] 계열, Flash-WAM의 행동 흐름 앵커 | 저장소 설계 문서가 Flash-WAM 논문 부록의 0.2 가중치를 인용 |
| rank-8 JointLoRA 블록 0–29 | LoRA | 저장소 문서: 300개 선형층, 학습 파라미터 약 20.15M |
| pseudo-Huber c = 1e-3 | Flash-WAM의 손실 정의 | 코드 주석 "FlashWAM's exact pseudo-Huber objective on selected elements" |
| RoboTwin 2.0 demo_clean | Chen 외 [43] (arXiv:2506.18088) | 평가 환경. 코드는 학습·수집이 demo_clean(Easy)만 허용 |
| 정확 쌍체 평가 | 저장소 프로토콜 | 같은 지시·초기 스냅샷·장면 시드·잡음 은행을 공유 |

## 4. GitHub와 자료 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (UCL-ERL/WAM-OPD)** | https://github.com/UCL-ERL/WAM-OPD | Apache-2.0 | 별 43, 포크 5, 최신 `191b2e3` (2026-10-19, 커밋 약 20개) | `repo/`에 얕은 clone(7.7 MB), 정적 읽기만. README가 릴리스 상태를 "HOLD(전체 벤치 재현 미인증)"로 적음 |
| 프로젝트 페이지 | 없음 | - | - | 연구실 홈(ucl-erl.github.io)의 논문 목록에 "arXiv preprint"로만 나옴 |
| LingBot-VA | https://github.com/Robbyant/lingbot-va | Apache-2.0 | 별 1,989 / 2026-07-09 | Teacher 소스. WAM-OPD 패치 대상 커밋 `58c2ae5` |
| Flash-WAM | https://github.com/NU-World-Model-Embodied-AI/Flash-WAM | NOASSERTION | 별 55 / 2026-06-30 | Student 소스와 체크포인트. 라이선스 미표기 |
| RoboTwin | https://github.com/RoboTwin-Platform/RoboTwin | MIT | 별 3,074 / 2026-09-24 | 시뮬레이터. 저장소는 `2eeec32`를 고정 |
| 같은 이름 다른 논문 | https://github.com/pjliu3/WAM-OPD (arXiv:2609.34250 "Sharpening World Action Models via On-Policy Distillation") | MIT | 별 2 / 2026-10-20 | **이 논문과 다른 연구**입니다(저자·제목·날짜가 다름). 혼동 주의. 분석 대상 아님 |
| VLA-OPD | arXiv:2603.26666 | - | 공식 코드 미확인 | 저장소 문서가 "공식 코드 이용 불가"라고 적음 |
| 가중치·데이터 | Flash-WAM 체크포인트, LingBot-VA 체크포인트, RoboTwin 자산 | 미확인 | - | **받지 않음.** 크기는 확인하지 않음 |

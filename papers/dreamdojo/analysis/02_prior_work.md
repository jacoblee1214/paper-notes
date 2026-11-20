# 02. 선행 연구, 구성 요소의 출처, 관련 자료

인용 번호와 저자 표기는 논문 본문을 따릅니다(예: Gao et al., 2025 = AdaWorld). 논문 원문 대조는 `paper/paper.txt`에 있습니다.

## 1. 계보

DreamDojo는 세 줄기가 합쳐진 연구입니다.

1. **영상 기반 세계 모델.** 영상 생성 모델을 상호작용 세계 모델로 쓰는 흐름입니다. 게임과 주행에서 시작했고(Bruce et al., 2024; Ha & Schmidhuber, 2018 등), 로봇 조작에서는 IRASim, UniSim, WorldGym, Ctrl-World, Cosmos 계열이 있습니다. 대부분은 실험실 로봇 데이터의 분포 안에서 평가되고, 쓰지 않은 행동에 대한 반응(반사실)은 약하다는 것이 논문의 문제 제기입니다(§1, §B).
2. **잠재 행동 (latent action).** 행동 라벨이 없는 영상에서 프레임 쌍의 변화를 연속 벡터로 압축해 행동 대용으로 쓰는 방법입니다(Bruce et al., 2024; Gao et al., 2025 AdaWorld; Ye et al., 2025). DreamDojo는 이를 **사람 영상 전체를 위한 공통 행동 공간**으로 씁니다.
3. **실시간 자기회귀 증류.** Self Forcing(Huang et al., 2025)과 분포 일치 증류(DMD, Yin et al., 2024)로 양방향 확산 모델을 인과적 4 스텝 학생으로 바꿉니다. 논문은 이것을 로봇 세계 모델에 적용하고 실시간 원격조작과 계획에 씁니다.

새로운 점은 (a) 이 셋을 한 파이프라인으로 묶은 규모(44k h 사람 영상), (b) 잠재 행동을 로봇 사후학습의 행동 입력 공간과 맞추는 방식(§3.3.2–3.3.3), (c) 실시간 증류 후에도 사전학습 이득이 남는다는 주장(§4.6)입니다.

## 2. 구성 요소별 출처

| 구성 요소 | 논문 위치 | 출처 | 이 논문의 기여 |
|---|---|---|---|
| 기반 영상 확산 모델 (Cosmos-Predict2.5) | §2 | Ali et al., 2025 | 그대로 사용 (2B, 14B) |
| 잠재 공간 토크나이저 (WAN2.2) | §2, §3.3.1 | Wan et al., 2025 | 그대로 사용 (시간 압축 4) |
| 흐름 매칭 손실 | 식 (2) | Lipman et al., 2022 | 그대로 사용 |
| 잠재 행동 VAE | §3.3.2, 식 (3) | Gao et al., 2025 (AdaWorld), Bruce et al., 2024 | 700M 시공간 트랜스포머, 차원 32, β = 1e-6 |
| 행동 청크 주입 | §3.3.1 | Guo et al., 2025; Huang et al., 2025; Zhu et al., 2025 | 4 연속 행동을 잠재 프레임 하나에 묶음. 상대 행동과 함께 표 5에서 효과 확인 |
| 상대 행동 재기준 | §3.3.1 | 논문 자체 | 잠재 프레임마다 재기준 (재현 방식은 `04` D2) |
| 행동 임베더 0 초기화 | §3.3.2 | Zhang et al., 2023 (ControlNet 계열) | 그대로 사용 |
| 시간 일관성 손실 | §3.3.2, 식 (4)–(5) | Gao et al., 2024; Wang et al., 2024; Yang et al., 2025 | 속도 차분 손실 추가 (λ = 0.1). 구현 축 문제는 `04` D1 |
| 자기회귀 증류 | §3.3.4 | Huang et al., 2025 (Self Forcing), Yin et al., 2024 (DMD) | 사전학습 교사에 적용. 워밍업 단계 추가 |
| 사람 영상 (EgoDex) | §3.2 | Hoque et al., 2025 | 공개 데이터 (Apple Vision Pro 손 포즈) |
| 사람 영상 (DreamDojo-HV) | §3.2 | 논문 자체 (크라우드소싱) | 44k h 중 대부분. 공개되지 않음 (`04` D7) |
| 대안 행동 신호 (HaMeR, MANO) | §3.3.2, §4.2 | Pavlakos et al., 2024; Romero et al., 2022 | 비교 대상으로만 사용 |
| 정책 평가 지표 (Pearson, MMRV) | §4.7 | Li et al., 2025 (WorldEval), Li et al., 2024 (SIMPLER) | 그대로 사용 |
| 정책 (GR00T N1.5) | §4.7 | Bjorck et al., 2025 | 단일 시점 변형 학습 |
| 배경 편집 (novel 평가셋) | §4.1 | Comanici et al., 2025 (Gemini 2.5 Flash Image) | 평가셋 생성에만 사용 |
| 화질 지표 PSNR, SSIM, LPIPS | §4.1 | Hore & Ziou, 2010; Zhang et al., 2018 | 표준 지표 |
| 가장 가까운 대안 | §B | VAP (Wang et al., 2025), AdaWorld (Gao et al., 2025), DexWM (Goswami et al., 2025) | 논문은 이들이 실험실 데이터에 머문다고 봄 |

## 4. GitHub와 코드 자료 (조회 2026-11-20)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (DreamDojo)** | https://github.com/NVIDIA/DreamDojo | Apache-2.0 (LICENSE, 저장소 설명) | 별 1,133, 포크 82, 최종 push 2026-03-21, 생성 2026-02-09 (조회 시점 값) | `repo/` 얕은 clone, commit `02f119b`. 사전학습·사후학습·LAM·증류·평가 코드 공개 (README News 2026-02-18). 정적 읽기만 |
| 프로젝트 페이지 | https://dreamdojo-world.github.io/ | 명시 없음 | - | 논문, 코드, 스레드 링크, 원격조작 영상 |
| 모델 가중치 (2B, 14B, LAM) | https://huggingface.co/nvidia/DreamDojo | 확인 안 함 | - | README가 공개를 주장. 받지 않음 |
| 데이터 (GR-1 사후학습, 평가셋) | https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1 | 확인 안 함 | - | DreamDojo-HV, In-lab, EgoDex 사전학습 데이터는 포함되지 않음 (`04` D7). 받지 않음 |
| 기반 모델 코드 (Cosmos-Predict2) | https://github.com/nvidia-cosmos/cosmos-predict2 | 확인 안 함 | - | `pyproject.toml`의 연결 링크 |
| ICML 2026 공식 목록 | https://icml.cc/virtual/2026/poster/65193 | - | - | 제목이 arXiv와 다름 (`03`, `01` §6) |

## 5. 아직 모르는 것

- 이 논문과 ICML 2026 채택판의 관계 (제목과 저자 목록의 차이).
- 평가셋과 사람 평가 UI의 공개 여부.
- 사전학습 데이터 공개 계획.

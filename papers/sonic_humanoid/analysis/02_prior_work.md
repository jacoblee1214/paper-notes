# 02. 선행연구, 공식 저장소, 관련 분석

## 1. 선행연구 계보

논문이 자리 잡는 흐름은 네 갈래입니다. 번호는 논문 참고문헌 번호입니다.

| 갈래 | 대표 연구 | 논문에서의 역할 |
|---|---|---|
| 적대적 모방 (보상 설계 없이 모션 분포를 맞춤) | AMP [16], ASE [17], CALM [18], MaskedMimic [19] | 다양성이 커지면 판별기 신호가 약해져 모드 붕괴가 생긴다고 주장의 출발점 (§2.6) |
| 동작 추종 (사람 모션을 프레임마다 목표로 씀) | PHC [34], Universal humanoid motion representation [20], OmniH2O [37], GMT [33], Any2Track [30], BeyondMimic [29] | 가장 가까운 기준선 세 개(GMT, Any2Track, BeyondMimic). 데이터와 재타겟팅이 달라 "데이터 일치 벤치마크가 아님" (§2.1) |
| 특화 보행·조작 제어기 | OpenHomie(HOMIE) [13] | 속도 추종 특화 기준선. 계산량 확장 비교 대상 (§2.6, 부록 S2.5) |
| 데이터와 재타겟팅 | 사람 모션캡처 (BONES-SEED [56], PHUMA [43]), GMR [54], PyRoki [55] | 학습 데이터 파이프라인과 외부 평가 셋 |
| 생성형 사람 모션 | GENMO [50] (논문에서 GEM) | 영상·텍스트·음악 입력을 사람 모션으로 바꾸는 상위 모델 (§2.3) |
| VLA 기반 로봇 제어 | GR00T N1 [42], GR00T N1.5 [42, 53] | 토큰 인터페이스에 붙는 VLA (§2.5) |
| 이산 잠재 표현 | FSQ [61], VQ-VAE [62] | 토큰 양자화 설계 비교 (표 3 A) |
| 학습 방법 | PPO [58], 비대칭 액터-크리틱 [63], Isaac Lab [45], MuJoCo [46] | 학습과 평가 기반. 비대칭 액터-크리틱과 PPO 설정은 표 S2 |

**논문의 위치.** 같은 추종 계열 가운데 논문은 (1) 데이터와 계산량을 한 자릿수 이상 키운 점, (2) 사람 입력과 로봇 입력을 한 토큰 공간에 넣은 점, (3) 계획기와 VLA까지 같은 토큰으로 묶은 점으로 차별화합니다. 학습 목표 자체(프레임 단위 추종 + PPO)는 BeyondMimic과 GMT와 같은 틀입니다. 새로운 것은 규모와 인터페이스입니다.

**모드 붕괴 주장의 근거.** §2.6은 AMP 계열의 판별기가 다양성에 따라 약해진다는 주장을 MaskedMimic [19]과 [20]에 기댑니다. 이 분석에서 두 문헌은 확인하지 않았습니다.

## 2. 공식 코드와 자료

| 저장소 또는 자료 | 관계 | 라이선스 | 규모와 최근 활동 | 대조 기준 |
|---|---|---|---|---|
| [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) | **공식 코드.** 논문 본문과 데이터 절이 지목. `gear_sonic/`(학습), `gear_sonic_deploy/`(C++ 배포), `decoupled_wbc/`(이전 분리형 WBC), `motionbricks/`(계획기 공개판) | LICENSE 파일이 이중 고지: 소스 Apache-2.0, 모델 가중치 NVIDIA Open Model License. GitHub API는 `NOASSERTION` | 별 3,938, 포크 644, 마지막 push 2026-11-03 | 커밋 `1e1ec4f` (2026-11-02, PR #287 병합). 얕은 복제 `repo/`(정적 읽기 전용) |
| [프로젝트 페이지](https://nvlabs.github.io/GEAR-SONIC/) | 공식 페이지. 영상, 그림, 코드 링크 | - | - | 본문 대조 |
| [nvidia/GEAR-SONIC (Hugging Face)](https://huggingface.co/nvidia/GEAR-SONIC) | 체크포인트 세 개: 기본(`sonic_release`), 저지연 원격조작(`low_latency`), v1.1(`sonic_v1_1`). ONNX 배포 파일과 PyTorch 파일 포함 | 허깅페이스 태그 `other` (저장소 LICENSE의 가중치 조항과 같음) | 다운로드 2,852회, 마지막 수정 2026-08-26 | 받지 않음. 논문 모델과 대응은 `04` D16 |
| [bones-studio/seed (Hugging Face)](https://huggingface.co/datasets/bones-studio/seed) | BONES-SEED 공개 데이터. 142,220 시퀀스, 288시간, G1 MuJoCo 궤적 포함(저장소 README) | 허깅페이스 태그 `other` | 다운로드 17,271회, 마지막 수정 2026-05-03 | 받지 않음 |
| [Zenodo 10.5281/zenodo.21273312](https://doi.org/10.5281/zenodo.21273312) | 논문이 말한 코드 아카이브와 Data S1 | Zenodo 기록은 Apache-2.0 | 게시 2026-07-09 | 메타데이터만 확인, 파일은 받지 않음 |
| `motionbricks/` (저장소 하위 폴더) | 실시간 잠재 생성 모델의 공개판. README는 "GEAR-SONIC 파이프라인 안의 전체 학습은 약 한 달 뒤 예정"이라고 씀(2026-04-27 시점) | 저장소와 같음 | 같음 | 계획기 식 (8) 대조 (`04` D10) |

- 저장소 이력: 2025-11-05 생성. 2026-02-19 GEAR-SONIC 공개, 2026-04-10 SONIC 학습 코드와 체크포인트 공개, 2026-07-23 v1.1 체크포인트, 2026-08-31 배포 Kp/Kd 조정 (README 공지).
- 논문 본문은 "Code is available at ..."이라고 씁니다. 학습 설정 파일 대부분이 저장소에 있어 정적 대조가 가능했습니다(`04`).
- 프로젝트 페이지는 "All models shown in the videos will be released"라고 씁니다. 영상에 나온 모델이 모두 공개되었는지는 확인하지 못했습니다.
- 공식 코드 목록에 논문이 쓴 BONES-SEED 학습 데이터 경로 `data/motion_lib_bones_seed`가 있습니다. 데이터 자체는 받지 않았습니다.


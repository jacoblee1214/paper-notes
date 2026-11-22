# 02. 선행연구와 구성 요소 출처

번호는 EgoWAM v2 참고문헌 번호입니다. 논문과 공식 저장소(`egowam/`, `external/`)에서 확인한 관계만 적었습니다.

## 1. 계보

EgoWAM은 세 줄기가 만나는 자리에 있습니다.

1. **사람 일인칭 데이터로 로봇 정책을 키우는 줄기.** EgoMimic [4]가 사람 손 궤적을 로봇 행동 공간으로 리타게팅해 같은 정책에 공동 학습하는 틀을 만들었고(같은 Georgia Tech 그룹), Humanoid Policy ~ Human Policy [5], EgoScale [6], EgoVerse [7]가 규모와 다양성을 키웠습니다. 논문은 이 줄기를 "시점·속도·양식을 정렬해야만 규모가 선다"고 요약하고(§1), 그 정렬을 §3.1에서 **최대한 해 둔 BC를 강한 기준선**으로 세웁니다.
2. **World Action Model(WAM) 줄기.** 정책이 행동과 함께 미래를 예측하는 계열: DreamZero [8], Cosmos Policy [9], UVA [10], UWM [11], mimic-video [48], DiT4DiT [49]. 대부분 사전학습 영상 VAE 잠재 공간을 예측합니다(§1). 추론 때 미래를 상상하지 않고 학습 때만 표현을 다듬는다는 관점은 Fast-WAM [60]과 UVA [10]에서 가져옵니다(§4.3). ImageWAM [68]은 부록 A.3에서 "WAM이 영상 전체를 생성할 필요는 없다"는 근거로 인용됩니다.
3. **세계 표현 선택 줄기.** 픽셀 대신 사전학습 특징 공간에서 예측하는 DINO-WM [54], "복원인가 의미인가"를 묻는 Nilaksh 외 [55], 표현 오토인코더(RAE) [13], 3D 점 궤적·흐름을 중간 표현으로 쓰는 계열(ATM [33], 3PoinTr [39], MolmoMotion [59] 등)입니다. 3D 흐름 계열은 보통 **추론 때 흐름에서 행동을 복원**하는 데 쓰이는데(§2), EgoWAM은 흐름을 학습 때만 보조 감독으로 쓰고 추론에서 버립니다.

**이 논문의 새로움**은 부품이 아니라 설계입니다(§1, §4). 같은 트렁크·행동 헤드·데이터 혼합에서 세계 목표만 바꿔 비교하는 통제 연구, 그리고 3D 흐름을 **카메라 안정화**(머리 운동 제거)해 사람·로봇의 목표를 맞추는 점입니다. 단, 헤드는 목표마다 구조가 달라 완전한 통제는 아닙니다(`04` D6).

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 논문 위치 | 공식 저장소에서 확인 |
|---|---|---|---|
| 트렁크·스템 (이질적 사전학습 트랜스포머, HPT) | Wang 외 [16], EgoVerse [7] | §4.1, 부록 B.1 | `egowam/algo/hpt.py`, `hpt_wm.py`. 가중치는 `pretrained: false`로 처음부터 학습 (`04` D8) |
| 행동 헤드: 흐름 일치 CrossTransformer | [66], [7] | §4.1 | `models/denoising_nets.py`, `fm_policy.py` |
| 사람 데이터와 구현체 정렬 | EgoVerse [7] (Aria, MPS 손 포즈, VIO 머리 포즈) | §3.1, C.2 | `rldb/embodiment/human.py`, `eva.py` |
| Pixel 목표: 영상 VAE | Wan [61] | §4.2, B.2 | `models/cosmos_vae.py`가 `Wan2pt1VAEInterface` 호출(토크나이저는 Cosmos-Predict2 저장소에 있음) |
| Pixel-PT 헤드 | VACE-1.3B [67] (논문), Wan2.1 T2V-1.3B 본체 블록(코드) | §4.2, B.2 | `models/wan_dit.py`, `hpt_wm_cotrain_pixelpt.yaml` (`04` D13) |
| DINO 목표 | DINOv2-B [12], 표현 오토인코더 [13] | §4.2, B.2 | `models/rae_vae.py` (인코더만 사용) |
| DINO 헤드의 넓은 DDT | DDT [69] | B.2 | 공개 설정에 없음 (`04` D3) |
| 3D 점 추적 | SpatialTrackerV2 [14], Track4World [15] | §4.2 | `external/Track4World`(NOASSERTION 라이선스), `scripts/3dflow_process/` |
| 카메라 포즈 | Project Aria VIO [64] | §4.2, A.5 | 데이터에 포함된 head pose |
| 구현체 평가 | RoboTwin 2.0 [72], ACT [74], Diffusion Policy [75] | 부록 D | `external/RoboTwin/policy/EgoWAM/` |

## 4. 공식 코드와 저장소

| 저장소 | 관계 | 라이선스 | 별 / 마지막 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|
| [GaTech-RL2/EgoWAM](https://github.com/GaTech-RL2/EgoWAM) | 공식 구현 (README에 "CoRL 2026") | MIT (LICENSE: "Copyright (c) 2025 EgoMimic-Team") | 49 / 2026-09-07 | `c87617fe37a6ed6a951e6b176ad552200c425c93` (2026-09-07 14:48 -0400) | 정적 읽기만 (`repo/`, 137 MB, gitignore). 열린 이슈 4개 |
| [gatech-rl2.github.io/egowam.github.io](https://gatech-rl2.github.io/egowam.github.io/) | 프로젝트 페이지 ("CoRL 2026" 표기, 코드·데이터·체크포인트 링크) | 명시 없음 (저장소 `GaTech-RL2/egowam.github.io`, API license null) | 0 / 2026-09-07 | `app.js`의 `PERF_DATA`로 그림 3 값 확인 | HTML과 `app.js`만 받아 읽음 |
| [GaTech-RL2/EgoVerse](https://github.com/GaTech-RL2/EgoVerse) | 데이터·BC 학습 생태계 (EgoWAM의 토대) | MIT | 586 / 2026-11-22 | 확인 안 함 | 읽지 않음 |
| [boeyyyy/EgoWAM-checkpoints](https://huggingface.co/boeyyyy/EgoWAM-checkpoints) (Hugging Face) | 학습된 체크포인트와 RAE 1단계 디코더 | 확인 안 함 | – | – | 받지 않음 (크기 미확인) |
| [partners.mecka.ai/egoverse](https://partners.mecka.ai/egoverse?task=EgoWAM_fold_clothes#explorer) | 데이터 탐색기 (EgoWAM 과제 포함) | – | – | – | 열지 않음 |
| [TencentARC/Track4World](https://github.com/TencentARC/Track4World) | 3D 흐름을 만드는 점 추적기 (저장소에 복제본 vendored) | NOASSERTION (Tencent 고지, 변경본 포함) | 348 / 2026-11-06 | 업스트림 `5b59b0c` + 5개 파일 변경 | 읽기만 |
| [bytetriper/RAE](https://github.com/bytetriper/RAE) | DINO 목표의 인코더와 DiT^DH 업스트림 (vendored) | MIT | 2110 / 2026-02-25 | `a4d18c4` + 5커밋 | 읽기만 |
| [NVlabs/cosmos-policy](https://github.com/NVlabs/cosmos-policy) | Wan 2.1 VAE 인터페이스 (vendored) | Apache-2.0 | 900 / 2026-01-23 | `18a2acc` + 11커밋 | 읽기만 |
| [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | 시뮬레이션 벤치마크 (vendored, 정책 브리지 추가) | MIT | 3074 / 2026-09-24 | `c3ddfa8` + 파일 14개 | 읽기만 |
| [liruiw/HPT](https://github.com/liruiw/HPT) | 이질적 사전학습 트랜스포머의 원 구현 | MIT | 547 / 2024-12-06 | – | 읽지 않음 |

- 별, 라이선스, push 일시는 `gh api`로 2026-11-22에 확인했습니다. EgoWAM 저장소는 2026-09-06에 만들어졌고 커밋은 3개(`d13cc07`, `f1e4838`, `c87617f`)입니다.
- 저장소 이슈(저자 답변 포함, 다른 사람이 쓴 데이터로 취급): #1 DINO와 3D 흐름을 함께 쓰는 실험은 하지 않았다는 답변, #2 L40S 한 장으로 약 3일(논문 B.3은 약 이틀), #5 3D 흐름 추출이 가져오는 `track4world.pipeline.efep_export` 모듈이 릴리스에 없음(정적으로 확인됨, `04` D11), #6 검증 영상 화살표 위치 오류 수정 제안.
- 체크포인트와 데이터는 받지 않았습니다. 데이터는 EgoVerse SQL 표와 객체 저장소에서 첫 학습 때 동기화되는 구조이며 계정이 필요합니다(`04` §3).

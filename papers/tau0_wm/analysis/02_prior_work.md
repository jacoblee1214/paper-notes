# 02. 선행연구와 관련 자료: τ0-WM

## 1. 계보

τ0-WM은 "영상 확산 모델을 로봇 정책과 시뮬레이터 양쪽에 쓴다"는 두 흐름을 한 백본에 묶습니다(II절).

1. **영상 예측으로 행동을 배운다 (Video Action Model, VAM).** 미래 영상 잠재와 행동을 함께 생성하는 계열입니다. 논문이 드는 대표: UVA [27], UWM [46], Video Policy [28], Genie Envisioner [29], Cosmos Policy [24], LingBot-VA [26], Motus [5], DreamZero [44], GigaWorld-Policy [43], Fast-WAM [45]. 대부분 사전학습된 영상 확산 백본(Wan, Cosmos, LTX-Video, CogVideoX)에서 출발해 영상과 행동을 같이 노이즈 제거합니다. 행동 가지를 따로 두고 영상 특징에 붙이는 구성은 Genie Envisioner의 GE-Act, EnerVerse [20]를 따른다고 본문이 적습니다.
2. **행동 조건 영상 시뮬레이터.** Visual Foresight [10, 11]에서 시작해 Cosmos [1, 2], Ctrl-World [15], DreamDojo [12], EnerVerse-AC [21]로 이어지는, 행동을 넣어 미래 관측을 만드는 모델입니다. ACVS의 행동 조건 방식은 Cosmos [2]를 따른다고 적었습니다(식 4).
3. **테스트 시간 계산.** 후보를 여러 개 뽑아 고르는 방식입니다. 논문의 비교 대상은 CFG [18]와 ACG [32](행동 일관성 가이던스)입니다.

본문은 τ0-WM이 이 둘을 "한 백본, 한 데이터 혼합, 한 행동 인터페이스"로 묶고, 테스트 시간에 제안-평가-수정을 한다는 점이 다르다고 설명합니다(II-A, II-B 끝). 같은 저자들(Pengfei Zhou, Shengcong Chen, Jianlan Luo 등)이 Genie Envisioner [29], EnerVerse [20], EnerVerse-AC [21]의 저자와 겹칩니다. 공개 코드도 GE-Act 코드를 수정해 썼다고 README가 밝힙니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| 영상 백본 5B | Wan2.2-TI2V-5B [39] | 코드의 `WanModel`, Wan VAE(스트라이드 16), umT5-XXL 텍스트 인코더를 그대로 사용 |
| 행동 가지 (0.5B, 층마다 영상 특징에 cross-attn) | GE-Act [29], EnerVerse [20], DiT [33] | 코드 계산으로 0.51B (`03` V11) |
| flow matching 손실 | Lipman 외 [30] | 영상과 행동에 독립 잡음 수준 |
| 행동 조건 주입 (시간 임베딩 + AdaLN 변조) | Cosmos [2] | 코드: 시간 임베딩과 6배 변조에 더함 |
| 보상 확산 헤드 | 논문 고유 (보상도 flow matching으로 생성) | 코드의 보상 가지는 1536차원, 0.91B (`03` V11) |
| 이질적 데이터와 감독 마스크 | UWM [46]의 독립 시간 단계, GE [29] 등과 유사 | 마스크 구현은 공개 코드에 없음 (`04` D1) |
| 후보 선택 | best-of-N 계열, Cosmos Policy [24]의 계획 | RCS는 논문 고유 |
| 가이던스 비교 | CFG [18], ACG [32] | 코드에 없음 (`04` D9) |
| 데이터 | DROID [23], Open X [31], BridgeData V2 [38] 등이 인용되나 코퍼스 구성은 로봇(자체) + UMI(GenRobot 10Kh RealOmni-Open [13]) + egocentric(EgoDex [19], Egoverse [35], Xperience-10M [36]) | 사전학습 데이터 파일은 공개되지 않음 |
| 가까운 대안 | Fast-WAM [45]: 추론 때 미래 영상 생성을 제거하는 쪽 | 같은 장에서 기준선으로 쓰임. 공식 τ0-WM의 행동 전용 추론도 미래 영상을 노이즈 제거하지 않음 (`04` D5) |

## 4. GitHub와 자료 표 (확인 2026-11-26)

| 자료 | 링크 | 라이선스 | 별 / 마지막 push | 비고 |
|---|---|---|---|---|
| **공식 코드** | https://github.com/sii-research/tau-0-wm | README는 Apache-2.0(링크는 diffusers 저장소의 LICENSE), 저장소에 LICENSE 파일 없음, API license 없음 | 523 / 2026-10-13 | 커밋 `e03902a4a8e0cf89b681e89920784ae8ca7b4b3c`로 얕은 clone, 정적 읽기만. 열린 이슈 2 |
| **공식 가중치** | https://huggingface.co/sii-research/tau-0-wm | Apache-2.0 (카드 표기) | 다운로드 1,021, 좋아요 18 (확인 시점), 수정 2026-07-30 | VAM 11.0 GB(파일 둘), 시뮬레이터 11.9 GB. 받지 않음 (HF API의 파일 크기만 조회) |
| 프로젝트 페이지 | https://tau0-wm.github.io/ | 확인 안 함 | 저장소 `tau0-wm/tau0-wm.github.io` push 2026-08-30 | 논문, GitHub, Hugging Face 링크와 영상 |
| Wan2.2 (기반) | https://github.com/Wan-Video/Wan2.2 | Apache-2.0 | 18,931 / 2026-09-21 | 코드와 가중치 파일의 기반. Wan2.2-TI2V-5B의 umT5-XXL 인코더 파일 11.4 GB, VAE 2.8 GB (HF API 크기) |
| Genie Envisioner V1 (GE-Act 기반) | https://github.com/AgibotTech/Genie-Envisioner-V1 | API 없음 | 592 / 2026-09-10 | README가 일부 코드를 여기서 가져왔다고 밝힘 |
| GE-Sim-V2 | https://github.com/AgibotTech/GE-Sim-V2 | Apache-2.0 | 166 / 2026-10-20 | 같은 계열 시뮬레이터 후속 |
| EnerVerse-AC | https://github.com/AgibotTech/EnerVerse-AC | API 없음 | 185 / 2025-07-14 | 행동 조건 영상 모델 선행 |
| openpi | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 14,767 / 2026-08-24 | 정책 서버(websocket) 클라이언트가 여기서 옴 |
| Cosmos Policy | https://github.com/NVlabs/cosmos-policy | Apache-2.0 | 901 / 2026-01-23 | VAM 계열 비교 |
| UVA | https://github.com/ShuangLI59/unified_video_action | MIT | 419 / 2026-08-21 | VAM 계열 비교 |
| EgoDex 코드 | https://github.com/apple/ml-egodex | NOASSERTION | 419 / 2026-09-11 | egocentric 데이터 소스 중 하나 |
| 학습 예제 데이터 | https://huggingface.co/datasets/lerobot/taco_play | 확인 안 함 | - | 공식 사후학습 예제가 가리키는 공개 데이터. 받지 않음 |

- 프로젝트 페이지에 코드와 모델 링크가 있어, 코드와 가중치는 같은 조직(sii-research)에서 공개됩니다.
- 사전학습 데이터(17.8K h 로봇, 롤아웃·실패 궤적)는 어느 목록에도 없습니다. UMI 계열(6.5K h)과 egocentric(3.0K h)은 논문이 인용한 공개 데이터셋에서 필터링한 것으로 적혀 있고, 필터 기준과 최종 목록은 확인하지 못했습니다.

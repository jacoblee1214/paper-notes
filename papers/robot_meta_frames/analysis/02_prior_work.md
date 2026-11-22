# 선행연구와 공식 저장소

## 1. 계보와 가까운 대안

MetaWAM은 "사전학습 영상 확산 모델을 로봇 행동 예측에 맞추는" WAM 계열이다. 논문 2절은 WAM을 두 갈래로 나눈다. 세계 모델을 먼저 만들고 행동을 따로 복원하는 **cascaded** WAM (Du 외 [2], Genie [3], World-VLA-loop [4])과, 하나의 생성 백본이 행동·미래 관측·가치를 한 번에 내는 **joint** WAM이다 (Cosmos Policy [5], DreamZero [6], UWM [9], UVA [10]).

가장 가까운 대안은 세 가지다.

| 대안 | 신호 주입 방식 | 논문과의 관계 | 근거 |
|---|---|---|---|
| Cosmos Policy [5] (ICLR 2026) | 현재·미래 고유감각, 행동, 가치마다 잠재 프레임 하나 (공간 복제) | 같은 백본 (Cosmos-Predict2-2B)과 같은 RoboCasa·LIBERO 설정. 제안 방식의 직접 기준선 (표 2, 3, 4, 5) | 논문 2절과 4.1절 |
| DreamZero [6] | 행동용 전용 인코더·디코더를 영상 백본 주변에 둠 | 부록 C.2에서 MLP 인코더·디코더 변형과 행동 L1 학습 곡선을 비교 | 논문 C.2절, 그림 10 |
| AgiBot-World [7] | 신호를 이미지로 그려 VAE로 인코딩 | 표 1의 "VAE Encoding" 비교 대상 | 논문 3절 |

신호 주입을 주파수 영역에서 하는 선행 연구도 있다. FAST [18]는 행동 시퀀스에 1D DCT를 쓰고, 웨이블릿 토크나이저 [19], FA-VAE [20], DocPedia [21]는 시각 또는 문서 모델에서 주파수 표현을 쓴다. 논문은 이들과 달리 **채널별 2D DCT 대역**으로 여러 신호를 한 잠재 프레임에 싣고 학습 파라미터 없이 되돌린다고 구별한다 (2절 마지막 문단).

잠재 행동 표현 계열 (VQ-VAE 잠재 행동 [13, 14], 세계 모델 기반 잠재 행동 [15], 이산 행동 청크 [16])과 모달리티별 잠재 위치를 쓰는 방식 [17]은 신호를 **학습된** 이산·연속 코드로 바꾼다. MetaWAM은 학습된 인코더를 쓰지 않는다는 점이 다르다.

효율 중심 WAM으로는 테스트 시 미래 프레임 생성을 없앤 Fast-WAM [11]과 행동 중심 디코딩 GigaWorld-Policy [12]가 있다. 논문은 이들과 직교하며 디노이저 구조를 바꾸지 않는다고 말한다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 논문에서의 역할 |
|---|---|---|
| 영상 확산 백본 | Cosmos-Predict2-2B-Video2World [22] | 초기화와 동결 디노이저 (표 1) |
| 대안 백본 | Wan2.1-I2V-14B, Wan2.2-TI2V-5B [23] | 부록 A.2 교차 백본 확인 |
| 시뮬 벤치마크 | RoboCasa [24] (24 과제), LIBERO [25] (4 스위트) | 표 2, 표 8 |
| 강건성 벤치마크 | LIBERO-Plus [26] (일곱 교란) | 표 3 |
| 기준선 수치 | 원자료 arXiv:2603.22078 표 4 (LIBERO-Plus) [32] | 표 3 |
| 기준선 수치 | Cosmos Policy [5], GE-Act [40], VLA-JEPA [39], ABot-M0 [38] 등 | 표 2, 3 |
| 실로봇 기준선 | π0 [28], π0.5 [41], Cosmos Policy [5] | 표 4 |
| 학습 설정 | Cosmos Policy 저장소와 Cosmos-Predict2 | 부록 B |

논문이 쓴 기준선 수치는 대부분 원 논문에서 옮긴 값이다. 표 3의 12개 기준선 행은 원자료 arXiv:2603.22078 표 4와 모두 같다 (`analysis/recompute.py`). 원자료에 있지만 표 3에 없는 행은 π0.5와 π0 rerun이다.

## 4. 공식 저장소와 관련 저장소

| 저장소 | 관계 | 라이선스 | 최근 push | 별 | 상태 |
|---|---|---|---|---|---|
| [showlab/MetaWAM](https://github.com/showlab/MetaWAM) | 공식 구현 (README에 따르면 Cosmos Policy 코드 기반) | Apache-2.0 (LICENSE 파일 있음) | 2026-11-11 | 1 | 얕은 복제 완료 (`repo/`, 커밋 `4e52ede`, 2026-11-11). 정적 읽기만 |
| [NVlabs/cosmos-policy](https://github.com/NVlabs/cosmos-policy) | 기반 코드 (학습·평가 골격) | Apache-2.0 | 2026-01-23 | 899 | 복제 안 함. 저장소 README의 기반 관계만 확인 |
| [nvidia-cosmos/cosmos-predict2](https://github.com/nvidia-cosmos/cosmos-predict2) | 기반 백본 코드 | Apache-2.0 | 2025-10-29 | 793 | 보관 처리됨 (archived). 복제 안 함 |
| Hugging Face `Ethanliang99/MetaWAM-RoboCasa-Predict2-2B` | RoboCasa 체크포인트 | 메타데이터 "other" | 2026-11-11 | 0 | 받지 않음. `.pt` 3.91 GB, T5 임베딩 0.46 GB |
| Hugging Face `Ethanliang99/MetaWAM-LIBERO-Predict2-2B` | LIBERO 체크포인트 | 메타데이터 "other" | 2026-11-11 | 0 | 받지 않음. `.pt` 3.91 GB, T5 임베딩 0.04 GB |
| Hugging Face `Ethanliang99/MetaWAM-RealWorld-Predict2-2B` | 실로봇 체크포인트 (이 로봇 설정에 묶임) | 메타데이터 "other" | 2026-11-11 | 0 | 받지 않음. `.pt` 3.91 GB |

- **프로젝트 페이지:** 찾지 못했다. arXiv abs 페이지와 GitHub 저장소 설명 (homepage 필드 비어 있음)에 링크가 없다.
- **코드 라이선스 확인:** GitHub API 호출은 한도에 걸려 `gh` CLI로 확인했다. 저장소 설명, 라이선스 (Apache-2.0), 최근 push 시각을 같은 출처에서 얻었다.
- **체크포인트 크기:** HF API 메타데이터로만 확인했다. 가중치는 받지 않았다.

공식 저장소의 README는 "CoRL 2026 채택"을 밝히고, 인용 항목에 arXiv 버전을 쓰라고 한다. 커밋 메시지 "Cite the arXiv version until the CoRL proceedings are out"도 같은 내용이다.

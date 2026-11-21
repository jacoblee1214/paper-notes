# WAMotor 선행연구와 관련 자료

## 1. 공식 코드와 프로젝트 페이지 (GitHub 표)

| 항목 | 주소 | 라이선스 | 별 | 마지막 push | 복제 여부 |
|---|---|---|---|---|---|
| **WAMotor 공식 코드** | https://github.com/Overdue-Lin/WAMotor | 없음 (`license: null`) | 6 (2026-11-22 조회) | 2026-11-20 03:00 UTC | 얕은 복제 `repo/` (커밋 `2de16b8`, 정적 읽기만) |
| 프로젝트 페이지 | 찾지 못함. arXiv abs에도 없고, 저장소 `homepage`는 비어 있음 | — | — | — | — |
| 저장소 이력 | 생성 2026-11-17, 커밋 3개(11-17 `d954aec`, 11-20 `9260bc1`, `2de16b8`) | — | — | — | — |
| Faster-WAM (기반 모델) | https://github.com/hustvl/FasterWAM | 조회 실패 (API 한도) | 조회 실패 | 조회 실패 | 미복제 |
| Wan2.2 (백본) | https://github.com/Wan-Video/Wan2.2 | 조회 실패 | 조회 실패 | 조회 실패 | 미복제. 모델 `Wan-AI/Wan2.2-TI2V-5B` |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | 조회 실패 | 조회 실패 | 조회 실패 | 저장소가 `third_party/LIBERO`를 기대하지만 없음 |
| LIBERO-plus | https://github.com/sylvestf/LIBERO-plus | 조회 실패 | 조회 실패 | 조회 실패 | 저장소에 없음 |
| RoboTwin 2.0 | https://github.com/RoboTwin-Platform/RoboTwin | 저장소 안 `third_party/RoboTwin/LICENSE` 있음 | 조회 실패 | 조회 실패 | `repo/third_party/RoboTwin`에 들어 있음 |

WAMotor 저장소 README의 BibTeX는 제목을 "Faster-WAM: Efficient Inference-Time Future Conditioning for Robust World Action Models"로 적고 `journal = {arXiv preprint}`입니다. 제목이 잘못 복사되었습니다(`04` D14). 저장소 README에는 "Joint-WAM" 체크포인트 공개 경로가 없습니다.

웹 검색(2026-11-22)으로 WAMotor의 프로젝트 페이지, 학회 채택 정보를 찾지 못했습니다. 다만 같은 검색에서 관련 논문 WAMACHINE(arXiv:2609.34608)을 찾았습니다.

## 2. 선행연구 계보

### 2.1 투기적 추론 (언어 모델 → 확산·흐름 정책)

- **투기적 디코딩** Leviathan 외 (ICML 2023, [19]), Chen 외 (2023, [5]): 초안 모델 + 병렬 검증.
- **EAGLE 계열** [23–25]: 특징 수준 초안, 동적 트리.
- **확산·흐름 정책으로 옮긴 연구:** ParaStep [40] (Wang 외, NeurIPS 2025, 속도 재사용 방식), SpeCa [30] (ACM MM 2025, 특징 캐시 기반 투기), FlowCast [1] (ICLR 2026, 학습 없는 속도 재사용 투기).
- **VLA 투기:** SpecVLA [41], FLASH/Realtime-VLA Flash [37], KERV [54], HEISD [55], WASpecDec [42]. 모두 과제별 초안 모델을 학습합니다.

WAMotor는 FlowCast와 같은 "학습 없는 속도 재사용" 계열이고, 차이는 두 가지입니다. 앵커를 u_1으로 옮기고, 검증 임계를 적분 간격으로 가중합니다(§4.3–4.4).

### 2.2 캐시 기반 가속 (기준선)

- **VLA-Cache** [44] (NeurIPS 2025): 적응형 토큰 캐시.
- **TeaCache** [29] (CVPR 2025): 시각 확산 캐시, 시간 임베딩 기반.
- **BAC** [16] (ICLR 2026): 블록 단위 적응 캐시.
- **DeepCache** [33], Token-wise 캐시 [58], EfficientVLA [46], ActionCache [38], C³Cache [53], WAM-Cache [10] 등.

WAMotor의 부록 B.1은 이들과 직교적이라고 주장하고 C.5에서 결합 실험을 합니다.

### 2.3 월드 액션 모델 (WAM)

- **Faster-WAM** [52] (Zhao 외, 2026): KV 융합과 희소 비디오–행동 상호작용으로 미래 조건을 싸게 만듭니다.
- **Joint-WAM** 계열 [45, 48, 52]: 비디오와 행동을 함께 denoise합니다. WAMotor 저자들은 이 계열의 대표 구현을 직접 만들었습니다([49]를 인용하지만 실제로는 저자 구현입니다).
- 기타: Cosmos Policy [18], Video Prediction Policy [15], Motus2 [3], Causal World Modeling/LingBot-VA [21], UWM/GigaWorldPolicy [35], Efficient-WAM [20], SelfWAM [39], InternW0-δ [34].

## 4. 이 논문의 위치 (요약)

WAMotor는 "학습 없는 속도 재사용 투기"라는 FlowCast의 틀을 WAM에 옮기면서, 재사용 오차를 적분 간격으로 가중하는 검증을 추가했습니다. 새로운 것은 검증 규칙과 앵커 선택입니다. 샘플러 자체(속도 재사용 + 병렬 검증)는 FlowCast와 같습니다. 기준선 구성에서는 FlowCast가 가장 가까운 대조군이고, 제거 실험의 "neither" 행도 FlowCast와 같습니다(`03` §4).

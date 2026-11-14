# 02. 선행 연구와 관련 자료

분석 대상은 arXiv:2607.14236 v3입니다. 이 문서는 (1) 구성 요소별 출처, (2) 가까운 선행 연구 계보, (3) 이 저장소의 관련 분석, (4) GitHub와 코드 자료 표를 적습니다. 코드 대조는 `04_code_verification.md`에 있습니다.

## 1. 구성 요소별 출처

| LIFT의 구성 | 출처 | 논문 위치 | 이 논문이 새로 한 것 |
|---|---|---|---|
| 기반 VLA π0.5 (PaliGemma 2B + action expert 300M) | Physical Intelligence, π0.5 [5] | Section 4 | 없음 (기반) |
| 기존 expert를 복사한 반응 expert | 저자 설계 | O2.1 | 복사 + shifted causal attention의 조합 |
| 청크 안 인과 디코딩 | 저자 설계 (ImplicitRDP의 causal attention에서 영향 [11]) | O1.1 | π0.5 action expert를 행동별 인과 스트림으로 바꿈 |
| 힘 인코더 GRU | ImplicitRDP [11]의 GRU 힘 메모리 | O1.2, Appendix A | 지연 정렬 마스크 $L$ 추가 |
| 0 초기화 교차어텐션 | 제로 초기화 잔차 관행 (ControlNet 류, 본문 인용 없음) | O2.2 | 힘 경로에 적용 |
| 시각-언어 KV 캐시 재사용 | 저자 설계 | O1.3, Algorithm 1 | 청크 안에서 힘만 다시 인코딩 |
| 온라인 DAgger | DAgger [23], HG-DAgger [24], SIRIUS [25], Sirius-Fleet [26] | Section 2 | 힘 입력 정책에 적용, 개입 프레임만 사용 |
| 1:1 대칭 샘플링 | RLPD [31] | O3.3 | 시각 오프라인 데이터와 힘 온라인 데이터 혼합 |
| 시각 전용 데이터 수집 | 손으로 든 iPhone 수집기 (ARKit SLAM) | Appendix E | 없음 |
| 원격조작 교정 장치 | Flexiv TDK [32] (쌍방향 주종 원격조작) | Appendix D | 없음 (외부 도구) |
| 잔차 기준선 | ImplicitRDP [11], CR-DAgger [30] 참고 | Appendix H | 저자 구현 (재현 아님) |
| 온라인 후학습 인프라 | SOP [28], RoboPocket [29] | Section 4.4 | 체크포인트 동기화 방식 |

## 2. 가까운 선행 연구 계보

**힘을 쓰는 정책 (정책 입력에 힘)**
- FoAR [7], ForceMimic [8], ACP [9] (ICRA/RA-L 계열): 힘 게이트 확산, 힘 중심 모방, 순응 정책.
- FACTR [15] (힘 주의 커리큘럼), Force Policy [17] (상호작용 프레임 힘-위치 제어), DexForce [16] (운동감각 시연에서 힘 목표 추출).

**빠른 반응과 느린 계획의 분리**
- RDP [10]: 느린 시각 확산 + 빠른 촉각 조건 디코더.
- ImplicitRDP [11]: 시각-힘 확산 + GRU 메모리 + 인과 어텐션. 청크 안 힘 관측으로 행동을 갱신합니다. LIFT의 가장 가까운 선행 연구입니다. 논문도 "이 메커니즘을 기반으로 MOT 기반 정책을 반응형으로 바꾼다"고 씁니다(Section 2).
- FA-RDP [18]: 학습된 다중모달 지표로 추론 주기와 확산 스텝을 조절합니다.

**힘을 붙인 VLA**
- ForceVLA [19] (힘 MoE 행동 디코더), TA-VLA [20] (토크 어댑터와 보조 예측), ForceVLA2 [21] (힘 프롬프트와 하이브리드 출력), FAVLA [22] (느린 VLM + 빠른 힘 전문가, 예측 힘 변화로 갱신 주기 조절).
- LIFT의 차이: 사전학습 출력을 초기화 시점에 보존하고(O2), 후학습 단계에서 힘을 넣으며(O3), 온라인 교정 루프를 씁니다.

**교정 기반 후학습**
- CR-DAgger [30]: 순응형 사람 교정으로 위치와 힘 잔차 정책을 학습합니다. 논문의 잔차 기준선이 이 계열입니다.
- SOE [27]: 온-매니폴드 탐색으로 자기 개선합니다.

**이 계보에서 빠진 것**
- 힘 입력을 실제로 어떤 주파수로 받는지는 선행 연구마다 다릅니다. 이 논문은 10 Hz로 내립니다(Appendix D). 반응 주기 논의에서 FA-RDP와 달리 주기를 바꾸지 않습니다.
- 출력 보존의 "동일성"이 같은 노이즈에서만 성립한다는 점은 어디에도 적혀 있지 않습니다(`03` §3, `04_code_verification.md` D1).

## 4. 대체 구현과 파생 자료 (이 저장소 밖)

- openpi (Physical-Intelligence): LIFT 코드의 기반입니다. π0.5 학습과 추론 코드를 제공합니다. 데이터 형식과 체크포인트 로딩 방식이 LIFT의 전제입니다.
- LeRobot 형식: LIFT의 온라인 데이터는 LeRobot 데이터셋으로 변환됩니다(저장소 `scripts/nedf2_to_lerobot_*`). 변환기는 외부 NEDF2 SDK(`nmx_nedf_api`)가 있어야 합니다.

## 5. GitHub와 코드 자료 (확인 2026-11-14)

| 자료 | 링크 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| LIFT 공식 저장소 | https://github.com/y-wng/LIFT | Apache-2.0 (LICENSE). `LICENSE_GEMMA.txt` 동봉 | 17 | 2026-10-13 | 클론 커밋 `024243426c` (2026-10-13, "docs: highlight CoRL 2026 Spotlight"). 생성 2026-09-17. 정적 읽기만 함 (`repo/`, gitignore) |
| 프로젝트 페이지 | https://lift-policy.github.io/ | - | - | - | 논문 초록과 영상 소개. 코드 링크는 위 저장소 |
| openpi | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 14,646 | 2026-08-24 | LIFT의 기반 코드. 여기서 실행하지 않음 |
| Flexiv TDK | https://github.com/flexivrobotics/flexiv_tdk | Apache-2.0 | 19 | 2026-11-14 | 원격조작 교정 도구. LIFT 저장소에는 없음 |
| ImplicitRDP (저자 저장소로 보임) | https://github.com/Chen-Wendi/ImplicitRDP | 필드 "other" (본문 미확인) | 64 | 2026-11-06 | 공식 여부는 프로젝트 페이지 링크로 추정. 확인 필요 |
| Reactive Diffusion Policy | https://github.com/xiaoxiaoxh/reactive_diffusion_policy | 필드 "other" (본문 미확인) | 397 | 2026-04-12 | RDP 원 저장소로 보임. 확인 필요 |

- 공식 저장소 README: "Use Linux, Python 3.11.8 이상, CUDA GPU, uv". 학습 예제는 GPU 2장(Book), 8-GPU 노드(논문)를 씁니다.
- 가중치: 학습 설정은 `gs://openpi-assets/checkpoints/pi05_base/params`(Hanoi, Towel)와 `$OPENPI_CHECKPOINT_ROOT/30000/params`(Book)를 가리킵니다. 가중치는 이 분석에서 받지 않았습니다.
- 데이터: 저장소에 데이터셋은 없습니다. 변환 설정 YAML만 있습니다(`preprocess_data/configs/*.yaml`).
- FA-RDP, SOP, RoboPocket, CR-DAgger의 저장소는 이번 분석에서 확인하지 않았습니다(미확인).

# DexTouch-WM 선행연구와 관련 저장소 (02)

## 1. 논문이 놓은 계보

논문 §II(Related Work)를 기준으로 네 갈래로 정리합니다. 괄호 안은 논문의 참고문헌 번호입니다.

| 갈래 | 대표 연구 | DexTouch-WM과의 차이 |
|---|---|---|
| 촉각을 제어 신호로 쓰는 정책 | 3D-ViTac [5], Reactive Diffusion Policy [26], ViTacFormer [3], Contact-grounded policy [25], FTP-1 [29], TactAlign [22] | 촉각을 **예측 대상**으로 쓰지 않음. 논문은 이 점을 명시적으로 구분 |
| 사람이 모은 시각-촉각 데이터 | Touch and Go [28], OpenTouch [18], DexUMI [24], FreeTacMan [23], DexViTac [1], Touch in the Wild [34] | 모방·표현 학습이 목적. 논문은 같은 센서를 로봇에도 달아 **세계모델 학습 데이터로** 쓴다는 점이 다름 |
| 시각-촉각 세계모델 | SPOTS [14], VT-WM [4], OmniVTA [32], FeelWorld [13], ViTacWorld [6], Dream-Tac [12], VT-WAM [19], TacForeSight [30], TouchWorld [33] | 대부분 로봇 원격조작 데이터로 학습. 논문은 사람 촉각 데이터의 규모 효과를 **고정된 로봇 데이터**와 대비해 측정 |
| 세계모델 평가·생성 | WorldGym [16], GigaWorld [2], Interactive world simulator [21], WorldSample [27], MMRV [9], [8] | 평가 지표(MMRV)는 그대로 가져다 씀. 정책 3개로 계산하는 한계는 03 §2에서 다룸 |

논문의 중심 가설(§II 마지막 문단): "공유 접촉 동역학이 있으면 사람의 물리적 상호작용이 손 세계모델의 규모를 키운다." 이 가설을 검증하려면 촉각 전이 자체(사람 촉각 → 로봇 촉각)를 따로 떼어 보는 실험이 필요한데, 논문에는 그 실험이 없습니다(04 C9, C10).

## 2. 기반 기술

- **Wan2.2-TI2V-5B와 비디오 VAE [20]**: 시각 전문가의 초기화와 프레임 잠재. 공식 저장소 `Wan-Video/Wan2.2`(Apache-2.0).
- **Mixture-of-Transformers [10]**: 모달리티별 AdaLN·FFN을 두고 attention만 공유하는 구조.
- **Conditional flow matching [11]**: 식 (4)의 손실.
- **MMRV [9], [8]**: 식 (7). 정책 순위 불일치를 실제 점수 차로 가중.
- **WorldArena [17], GigaWorld [2]**: 시각 평가 지표 묶음(Aesthetic, Image Quality, JEPA similarity, Subject consistency, Trajectory accuracy). 논문은 이 프로토콜을 따른다고 하나 정의는 원 논문을 봐야 합니다.

## 3. 이 저장소의 다른 논문과의 관계

| 폴더 | 연구 | DexTouch-WM과 맞닿는 지점 | 쓸 만한 것 |
|---|---|---|---|
| `contact_grounded_policy/` | 미래 촉각·상태 예측과 접촉 일관성 | 미래 촉각 예측 | 예측 대상을 접촉 상태로 둔 설계 비교 |
| `restacvla/` | π0.5에 촉각 잔차를 VQ 토큰으로 주입 | 촉각을 "시각 예측과의 차이"로 표현 | 잔차 대 원시 ablation의 부재를 채우는 질문 |
| `flashdexretarget/` | 사람 손-물체 시연 → 로봇손 궤적(RL) | 사람→로봇 리타게팅 | 리타게팅 오차를 어떻게 재는지(04 C12의 빈칸) |
| `gnr/` | 생성형 리타게팅 잔차(MPC 보정) | 사람→로봇 행동 정렬 | 같은 IK 리타게팅 문제를 다른 방식으로 푼 비교 기준 |
| `dreamx/` | 사람 영상을 latent action으로 로봇 학습 | 사람 데이터를 로봇 행동 공간으로 옮김 | 사람 데이터 전이의 성공·실패 사례 |
| `openwam/` | World Action Model 설계 부품 비교, Wan2.2 DiT 기반 | 같은 백본(Wan2.2-TI2V-5B) 계열 | 백본 선택과 학습 설정의 기준선 |
| `agentic_wam/` | 코드·영상·행동을 한 MoT로 생성 | 세계행동모델 구조 | MoT 구조 비교 |
| `gpt6_astra_tactile/` | 고정된 대형 모델을 촉각 정책으로 씀 | 촉각 정책 대안 | 반사실 촉각 시험 설계 |

## 4. GitHub와 공개 자료 조사

조사 날짜 2026-11-10. `gh search repos`는 검색 API 404로 실패해 `gh api repos/...`로 확인했습니다. 웹 검색으로도 프로젝트 페이지를 찾지 못했습니다.

| 저장소 | 내용 | 공식? | 라이선스 | 스타 | 마지막 커밋 (hash) |
|---|---|---|---|---|---|
| DexTouch-WM 코드 | 논문 공식 구현 | **없음 (미확인)**. `DexTouch-WM`, `DexTouch`, `2609.20649`로 저장소 검색 0건, 웹 검색에도 프로젝트 페이지 없음 | - | - | - |
| `Wan-Video/Wan2.2` | 백본 Wan2.2-TI2V-5B | 공식(Wan 팀), 논문이 쓴 백본 | Apache-2.0 | 18,481 | 2026-09-21 (`1ea34ff`) |
| `wuji-technology/wujihandpy` | Wuji Hand C++/Python SDK | 공식(Wuji 팀), 논문의 Wuji 손 제어용으로 추정 | MIT | 40 | 2026-08-24 (`468e169`) |
| `wuji-technology/wujihandros2` | Wuji Hand ROS2 드라이버 | 공식(Wuji 팀) | 명시 없음 (`none`) | 16 | 2026-08-24 (`95000ed`) |
| `wuji-technology/wujihand-hmi` | Wuji Hand 모니터링·보정 GUI | 공식(Wuji 팀) | MIT | 2 | 2026-04-08 (`79d4f88`) |
| `Lee-Kangwon/dextouch` | DexTouch (LRA 2024, 촉각 전용 조작, arXiv:2401.12496) | **다른 연구**(이름만 같음) | - | 0 | 2024-01-12 |

검토 메모:
- 논문의 촉각 장갑(Moxian)과 사람 손 추적(MANUS, VIVE)용 SDK는 GitHub에서 찾지 못했습니다. 장치 제조사 자료는 공개 저장소로 확인되지 않았습니다.
- Wuji SDK 두 개(wujihandpy, wujihandros2)는 논문의 WujiHand2와 같은 제품군이라는 점만 확인했습니다. 두 저장소 모두 WujiHand2 모델을 명시하는지는 확인하지 않았습니다.
- 논문 공개 자료 확인: 프로젝트 페이지 링크 없음(arXiv 초록 페이지와 PDF 본문 모두). 코드·데이터·가중치 공개 문장 없음.
- 논문 이름과 같은 외부 블로그 요약(예: 영문·중문 논문 요약 저장소)은 있으나, 원 저자가 올린 자료가 아니므로 근거로 쓰지 않았습니다.

## 5. 공개 자료 상태 (04와 연결)

- 공식 코드 없음 → 클론 없음 → `repo/` 없음 → 04는 `04_claim_verification.md`(주장 C1–C20 재계산·검증)로 씀.
- 리타게팅·데이터 수집 하드웨어 자료가 공개되지 않아, 04 §4의 재현 판정은 "논문만으로 구현하는 쪽"으로 봐야 합니다.

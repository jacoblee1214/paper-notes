# 02. 선행연구와 관련 자료

## 1. 계보

1. **잔차 정책 학습.** 고정된 기반 정책 위에 작은 보정을 학습하는 흐름: Residual RL for Precise Assembly(참고문헌 [55]), PLD(Self-improving VLA via residual RL, [22])가 가장 직접적인 선행입니다. OmniTacTune의 잔차 크기 스케줄, 기반 행동 노출, 웜스타트 롤아웃은 PLD와 WSRL([23])의 계승입니다(§3.3 "Following PLD").
2. **실로봇 RL.** HIL-SERL([21])의 비동기 학습기와 실행기 구조, 사람이 주는 종료 보상. RL Token([24])은 VLA의 온라인 RL 부트스트랩.
3. **객체 중심 flow 인터페이스.** Im2Flow2Act([27]), GenFlowRL([33], 같은 1저자), Dex4D([59]). 생성된 객체 flow가 기반 정책 입력이자 보상 신호가 됩니다.
4. **촉각 표현.** AnyTouch2([64]), Sparsh([68]), T3([69])는 인코더 후보입니다. ControlTac([66], 같은 저자 그룹)는 힘 조건부 촉각 합성입니다.
5. **시각-촉각 정책(비교 대상).** RDP([10], 느린-빠른 확산 정책), ViTAL([63], "Touch begins where vision ends"), LeFlexiTac류 tactile token을 붙인 π0.5([67]).

## 2. 가장 가까운 대안과 차이

| 대안 | 같은 점 | 다른 점 |
|---|---|---|
| PLD [22] | 잔차 SAC, 기반 행동 입력, 스케일 스케줄, 웜스타트 | 시각 전용. critic을 오프라인 시연으로 초기화. OmniTacTune은 촉각 전이가 없어 로봇 롤아웃으로 critic과 인코더를 부트스트랩 |
| HIL-SERL [21] | 실로봇 SAC, 사람 종료 보상 | 사람이 개입 교정을 하는 구조. OmniTacTune은 자율 롤아웃 |
| CR-DAgger [49] | 접촉 풍부 과제의 잔차 교정 | 사람 개입(DAgger) 기반 |
| RDP [10] | 촉각으로 빠른 반응 | 모방 학습, 촉각 시연 필요 |
| ViTAL [63] | 시각-촉각 정책 | 모방 기반. 여기서는 "처음부터 RL"하는 변형으로 구현되어 원 방법과 구성이 다릅니다 |
| MimicTouch [7] | 접촉 풍부 과제, 같은 1저자 | 인간 촉각 시연 모방 후 실로봇 RL 개선 |

OmniTacTune의 새로운 부분은 (1) 촉각 인코더와 critic을 웜스타트에서 함께 맞추는 순서, (2) 궤적 단위 ControlTac 증강, (3) 객체 flow를 인터페이스와 보상 양쪽에 쓰는 구성입니다. 개별 부품은 모두 기존 방법입니다. 무엇이 성능에 얼마나 기여하는지는 절제가 10회 평가라 분리가 약합니다(`03` §7).

## 4. GitHub와 코드 자료 (확인 2026-11-23, `gh api`)

| 자료 | 링크 | 라이선스 | 별 / 마지막 push | 비고 |
|---|---|---|---|---|
| 프로젝트 페이지 | https://colinyu1.github.io/omnitactune-site/ | - | 별 1 / 2026-09-20 | 저장소 `ColinYu1/omnitactune-site`(템플릿 페이지). 페이지에 코드 링크 없음 |
| 공식 코드 | 찾지 못함 | - | - | 저장소 검색(`omnitactune`)과 페이지·논문 확인. 저자 계정 저장소 목록에도 없음 |
| 같은 1저자의 GenFlowRL | https://github.com/ColinYu1/genflowrl | 없음 | 별 0 / 2025-10-15 | 관련 선행. 코드 내용은 열어 보지 않음 |
| Im2Flow2Act | https://github.com/real-stanford/im2Flow2Act | MIT | 별 160 / 2024-10-17 | flow 생성기 원본 |
| HIL-SERL | https://github.com/rail-berkeley/hil-serl | Apache-2.0 | 별 1,600 / 2025-10-27 | 실로봇 SAC 구조 |
| WSRL | https://github.com/zhouzypaul/wsrl | 없음 | 별 153 / 2026-02-26 | 웜스타트 RL |
| RDP | https://github.com/xiaoxiaoxh/reactive_diffusion_policy | 판정 불가(NOASSERTION) | 별 398 / 2026-04-12 | 비교 기준선 |
| ViTaL (Zhao 외) | https://github.com/Exiam6/ViTaL | MIT | 별 115 / 2025-07-01 | 비교 기준선 [63]의 원 구현으로 보임(동일성 미확인) |
| AnyTouch2 | https://github.com/GeWu-Lab/AnyTouch2 | MIT | 별 69 / 2026-04-17 | 촉각 인코더 |
| Sparsh | https://github.com/facebookresearch/sparsh | 판정 불가(NOASSERTION, 분석에서 CC BY-NC) | 별 225 / 2025-02-27 | 촉각 인코더 |
| T3 | https://github.com/alanzjl/t3 | MIT | 별 64 / 2024-06-21 | 촉각 인코더 |
| ControlTac | https://github.com/DongyuLUO/ControlTac | 없음 | 별 5 / 2026-09-16 | 촉각 증강 |
| CR-DAgger | https://github.com/yifan-hou/cr-dagger | MIT | 별 98 / 2025-12-10 | 관련 |
| CoTracker | https://github.com/facebookresearch/co-tracker | 판정 불가(NOASSERTION) | 별 5,211 / 2026-03-03 | 키포인트 추적 |
| DINOv2 | https://github.com/facebookresearch/dinov2 | Apache-2.0 | 별 13,563 / 2026-11-05 | |
| SAM | https://github.com/facebookresearch/segment-anything | Apache-2.0 | 별 55,170 / 2024-09-18 | |
| openpi (π0.5) | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 별 14,743 / 2026-08-24 | |

PLD와 OpenTeach, Dex4D는 저장소 주소를 확인하지 못했습니다(검색 결과 없음 또는 404). 공식 코드가 없어 복제하지 않았고 `repo/`는 비어 있습니다.

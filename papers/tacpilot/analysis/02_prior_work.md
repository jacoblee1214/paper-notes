# 02. 선행연구, 공식 저장소, 관련 분석

## 1. 계보

- **시각 정책 기반.** ACT [27]이 고정된 시각 정책의 기저입니다. Diffusion Policy [5], OpenVLA [12], π0.5 [3]는 같은 계열의 대안입니다.
- **촉각 주입.** AT-VLA [14]는 촉각 주입 시점을 학습하고 빠른 촉각 스트림을 둡니다. 시간 구조(느린 시각, 빠른 촉각)는 TacPilot의 동기와 같습니다. Reactive Diffusion Policy [23]도 같은 구조입니다. See, Hear, and Feel [13], 3D-ViTac [10], NCF-v2 [8]가 융합 방식의 예입니다. Matak & Hermans [15]는 정밀 파지에서 시각과 촉각을 나눠 씁니다.
- **잔차 정책.** Johannink et al. [11]와 ResiP [1]는 잔차 강화학습의 원형입니다. TacPilot에 가장 가까운 것은 OmniTacTune [24] (DTR로 재구현, 온라인 SAC)과 ViTaR [20] (arXiv:2608.15816, 결과에 따라 잔차 크기를 조절)입니다. EquiTac [29]는 촉각 법선 맵으로 방향 보정을 합니다.
- **촉각 표현과 선택적 계산.** T-DEX [7], UniT [22], T3 [26], AnyTouch [6], Sparsh [9]가 촉각 표현 학습의 계보입니다. 촉각 잔차에서 선택적 계산을 쓴 예는 TacPilot의 S2에 가까운 DeltaCNN [17] (영상 프레임 차이 전파)과 신경형 촉각 [19]입니다. TouchWGNN [16]는 그래프로 촉각 시공간을 다룹니다.
- **TacPilot의 자리.** 기존 연구는 공간 구조와 시간 구조를 설계 선택으로 썼습니다(§2.1). TacPilot은 그 위에서 특징 갱신과 행동 개입을 따로 게이트합니다. 새 요소는 이 분리와 영역 재사용입니다. 시각 정책은 고정합니다.

## 2. 공식 저장소와 프로젝트 페이지 (2026-11-21 확인)

| 저장소·페이지 | 관계 | 라이선스 | 최근 push | 상태 |
|---|---|---|---|---|
| 논문 본문·arXiv 초록 | 프로젝트 페이지 링크 없음, 코드 링크 없음 | — | — | 찾지 못함 |
| `Edge-Intelligence-Lab/tactile_robotics_manipulation` | 제3자 저장소(`openarm-runtime`의 `THIRD_PARTY.md`가 "TacPilot 저장소"로 가리킴) | 알 수 없음 | — | **404 (비공개 또는 없음)**, 복제하지 않음 |
| [Edge-Intelligence-Lab/openarm-runtime](https://github.com/Edge-Intelligence-Lab/openarm-runtime) | 관련 저장소 (공식 아님). `src/openarm_policy/models/act.py`와 `tacpilot.py`가 TacPilot의 ACT 로더·정규화·쿼터니언 차트를 적응했다고 명시 | MIT (LICENSE 파일 확인) | 2026-11-19 (생성 2026-10-31), 별 2개 | 메타데이터와 적응 파일 세 개만 읽음. 복제 없음, 실행 없음 |

- `openarm-runtime`은 OpenArm 로봇용 Python 런타임입니다. TacPilot 논문의 방법을 구현한 저장소가 아니라 TacPilot 측 ACT 어댑터와 업라이트 준비 스크립트를 가져다 쓴 downstream 저장소입니다. 그 README는 "TacPilot 저작권 2026 contributors"의 MIT 고지를 옮겨 둡니다.
- 따라서 **공식 코드는 공개되어 있지 않습니다.** `04_claim_verification.md`는 코드 대조 대신 주장 검증으로 씁니다.
- ACT 공식 구현과 OpenVLA, π0.5의 저장소는 이 분석에서 확인하지 않았습니다.


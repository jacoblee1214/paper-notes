# 02. 선행연구와 구성 요소 출처: HumanX (arXiv:2602.02473)

참고문헌 번호는 v2의 번호입니다. 논문이 말하는 계보와, 코드 저장소 README의 감사 목록(ASAP, HumanoidVerse, BeyondMimic, SkillMimic)을 함께 봅니다.

## 1. 계보

HumanX는 세 줄기가 만나는 자리에 있습니다.

1. **물리 기반 캐릭터 모방(참조 추종 보상).** DeepMimic [23, 77]의 참조 운동 추종 + 조기 종료, AMP [55]의 자연스러움 보상, SFV [35]의 영상으로부터 기술 학습. 사람-물체 상호작용으로 확장한 것이 **PhysHOI [16]와 SkillMimic [36]** 입니다. 둘은 같은 저자 그룹의 이전 작업입니다. SkillMimic-v2 [73]는 부록의 접촉 그래프 보상 출처로 인용됩니다.
2. **휴머노이드 전신 추종과 영상 기반 데이터.** ASAP [24], BeyondMimic [28], 일반 전신 추종 [26, 27, 33, 34, 61, 62], VideoMimic [37](영상에서 사람-장면 상호작용 데이터를 뽑아 실로봇에서 모방), GMR [38](일반 모션 리타게팅).
3. **휴머노이드 상호작용 모방.** OmniRetarget [22](상호작용 보존 리타게팅으로 로봇 데이터 생성), HDMI [39](영상에서 상호작용 전신 제어를 배움), PhysHSI [7](사람-장면 상호작용, 과제별 보상). 이 셋이 표 1의 비교 대상 또는 가장 가까운 동시기 작업입니다.

대안 계보는 (a) 텔레오퍼레이션 + 행동 복제(HumanPlus [1], GR00T N1 [2], Open-TeleVision [3])로 데이터가 비싸고, (b) 과제별 보상 RL(축구 [4], 배드민턴 [5], 일어나기 [6], 골키핑 [53], 상자 운반 [54, 7], 집어 옮기기 [17, 18])로 보상 설계가 과제마다 필요하다는 것이 저자들의 동기입니다(1절).

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | HumanX에서의 쓰임 |
|---|---|---|
| 영상 → 사람 자세 | GVHMR [11] (공식 코드와 README의 권장 경로도 GVHMR) | 단안 영상에서 세계 좌표계 SMPL 동작 |
| 사람 → 로봇 리타게팅 | GMR [38] (논문), UMR (코드 README의 권장) | G1 관절 각도. 논문과 README가 서로 다른 리타게터를 가리킴 |
| 물체 메시·자세 추정 | SAM 3D [12], 대안으로 수동 | 물체와 앵커 상대 회전 ϕ |
| 접촉 최적화 | force closure 계열 [65–67] | 접촉 구간 로봇 자세 보정 |
| 비접촉 물체 궤적 | Isaac Gym 시뮬레이터 [68] | 논문은 시뮬레이터, 공개 코드는 해석적 포물선 (`04` D4) |
| 모방 보상 | DeepMimic [77] 지수형 추적 보상, AMP [55] | 식 3 γ·exp(−λe), 몸 보상 7항 + AMP |
| 접촉 그래프 보상 | PhysHOI [16], SkillMimic [36, 73] | 식 6–7 |
| 상대 위치 보상 | PhysHOI·SkillMimic의 상대 위치 항 | **IV 보상.** 이름과 확장 점(손가락 끝, 라켓)이 새롭고, v1에서는 "relative motion reward"였음 |
| 조기 종료 | DeepMimic [23] | IVT. 확률적 적용이 차이 |
| 교사-학생 | 일반적인 DAgger식 증류 + 비대칭 actor-critic [70] | RL 보상과 BC 손실을 함께 씀 |
| 무작위화 | Tobin 등 [76], ASAP·HumanoidVerse 계열 설정 | 그림 14 |
| PD 이득 | BeyondMimic [28]의 고유 진동수·감쇠비 방식 | k_p = Iω², ζ = 2, ω = 10 Hz (`04` K1) |
| 학습 프레임워크 | HumanoidVerse (공식 코드 README 감사 목록), Isaac Gym | 코드는 HumanoidVerse 구조(Hydra 설정, `humanoidverse/`) |

새로운 것은 개별 부품이 아니라 **(1) 영상에서 앵커 기반으로 상호작용 데이터를 합성하고 증강하는 파이프라인을 한 틀로 묶은 것, (2) 상대 벡터(IV) 보상과 접촉 조건부 조기 종료, 초기·외력 교란의 조합, (3) 고유감각만으로 물체를 다루는 NEP 배포**입니다. 저자도 개별 요소의 독자성을 주장하지 않고 "주요 설계 세 가지"로 묶습니다(1절).

## 4. 코드와 저장소 표

2026-11-23 GitHub API와 프로젝트 페이지로 확인했습니다.

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| [wyhuai/HumanX](https://github.com/wyhuai/HumanX) | **공식 코드** (xgen + xmimic) | `Apache-2.0 with Commons Clause` (GitHub API는 NOASSERTION; `setup.py`는 BSD-3-Clause 표기) | 83 | 2026-10-22 | 얕은 복제 `02634bf` (2026-10-22 "Update documentation"), 정적 읽기 전용. 커밋 8개, 첫 공개부터 이 상태 |
| [wyhuai.github.io/human-x](https://wyhuai.github.io/human-x/) | 프로젝트 페이지 | - | - | - | 초록이 v1 문구("10 skills", "over 8×", "task-agnostic"). 하단 BibTeX와 감사 문구는 다른 논문(HumanPlus)의 것 |
| [wyhuai/SkillMimic](https://github.com/wyhuai/SkillMimic) | 선행 연구(같은 저자), 표 1 기준선 | Apache-2.0 | 455 | 2025-06-30 | HumanX 코드 감사 목록에 있음 |
| [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) | 영상 → 사람 자세 | NOASSERTION | 2,312 | 2026-09-15 | 논문 [11] |
| [YanjieZe/GMR](https://github.com/YanjieZe/GMR) | 리타게팅 (논문 [38]) | MIT | 2,841 | 2026-10-22 | 논문 3절 |
| [hanyang9/UMR](https://github.com/hanyang9/UMR) | 리타게팅 (코드 README 권장) | MIT | 531 | 2026-10-02 | 공식 README의 단계 2 |
| [LeCAR-Lab/HDMI](https://github.com/LeCAR-Lab/HDMI) | 표 1 기준선 [39] | 없음 | 684 | 2026-01-17 | 라이선스 파일 없음 |
| [LeCAR-Lab/HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse) | 코드 기반 프레임워크 | MIT | 475 | 2025-06-12 | 공식 코드 감사 목록 |
| [LeCAR-Lab/ASAP](https://github.com/LeCAR-Lab/ASAP) | 선행 [24], 감사 목록 | MIT | 2,146 | 2026-01-06 | |
| [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | BeyondMimic 코드 [28] | MIT (보관됨, 2026-10-16) | 2,443 | 2026-10-16 | PD 이득 방식의 출처 |

- 공식 코드가 **데이터와 에셋**으로 포함한 것: 예시 모션 4개(`xmimic/data/motions/example/*.pkl`, 가장 큰 것은 축구 64개 증강 10.5 MB), XGen 예시 npz 3개, G1 URDF/MJCF와 메시, 물체 URDF(농구공, 축구공, 셔틀콕, 상자, 빈). **정책 체크포인트와 실로봇 배포 코드는 없습니다.**
- 가중치와 데이터셋은 받지 않았고(저장소 얕은 복제 226 MB는 코드와 에셋), 설치와 실행은 하지 않았습니다.

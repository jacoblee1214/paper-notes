# 02. 선행연구, 공식 저장소, 구성 요소별 출처

## 1. 계보

X-WBC는 세 갈래를 합칩니다.

| 갈래 | 대표 | X-WBC가 가져온 것 |
|---|---|---|
| 모션 추적 기반 휴머노이드 WBC | H2O (IROS 2024), OmniH2O (CoRL 2024), HumanPlus, BeyondMimic (2025), SONIC (2025) | 사람 모션을 추적 목표로 쓰는 인터페이스, 추적 보상군, 실패율 기반 적응 샘플링 |
| 교차 신체 로봇 기초 모델 | RT-2, OpenVLA, π0·π0.5·π0.7, GR00T N1, HPT, CrossFormer, X-VLA | "로봇 데이터를 섞어 공유 백본을 학습하고 신체별 입출력 모듈을 둔다"는 틀(HPT의 stem, X-VLA의 소프트 프롬프트와 같은 계열) |
| 사람 모션 리타게팅 | AMASS, LAFAN1, BONES-SEED, SMPL, GMR | 사람 모션을 로봇별 기준 모션으로 바꾸는 입력 파이프라인 (GMR을 수정해 사용) |

- 비교 대상이 되는 가장 가까운 연구는 **SONIC**(단일 로봇 G1, 대규모 모션 추적, 같은 BONES-SEED 데이터 사용)과 **TWIST**(SMPL→GMR 입력 텔레옵 시스템)입니다. 표 2의 기준선이 이 둘입니다.
- X-WBC의 새로움은 **"학습 단위를 로봇 하나에서 로봇 여럿으로 바꾸고, 사람 중심 토큰으로 정렬"**한 구조입니다. 개별 구성 요소(추적 보상, 적응 샘플링, Transformer 정책, PPO)는 기존 것을 그대로 씁니다(§4 표).
- 가까운 대안: 로봇별 정책을 따로 학습한 뒤 증류하는 방법(예: 단일 로봇 학생 정책 여럿을 한 정책으로 증류), 형태 조건부 정책(로봇 기술 입력). 논문은 이 대안과 직접 비교하지 않습니다.

## 2. 공식 저장소와 관련 저장소 (GitHub 조회일 2026-11-23)

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [LogosIntelligenceLab/x-wbc](https://github.com/LogosIntelligenceLab/x-wbc) | 공식 저장소(이름만 구현 저장소). **프로젝트 페이지와 출시 예고만 있음** | 없음. README가 "미래 코드·모델의 라이선스는 미정, 미공개 산출물은 허가 없음"이라고 명시 | 21 | 2026-10-25 | 커밋 `c0dbf302` (커밋 1개). 얕은 clone 후 정적 읽기만. 학습·평가 코드, 체크포인트, 배포 예시는 체크리스트에서 "미완료" |
| [프로젝트 페이지](https://logosintelligencelab.github.io/x-wbc/) | 논문 v2의 주소. 영상 링크(YouTube) 포함 | 해당 없음 | - | - | 페이지의 수치는 표 1·2와 일치(98.60/93.22, 91.13/93.00/75.62). 이전 주소는 이동 안내 |
| [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) | SONIC 공식 코드, 표 2 기준선 | NOASSERTION (저장소 별도 약관, 가중치는 별도 모델 라이선스) | 3,939 | 2026-11-03 | 받지 않음 |
| [YanjieZe/TWIST](https://github.com/YanjieZe/TWIST) | 표 2 기준선 TWIST | MIT | 824 | 2025-11-01 | 받지 않음 |
| [YanjieZe/GMR](https://github.com/YanjieZe/GMR) | 리타게팅 파이프라인(논문은 "수정해 사용") | MIT | 2,841 | 2026-10-22 | 받지 않음 |
| [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | BeyondMimic 공식 코드. 보상·종료·적응 샘플링의 원형 | MIT | 2,443 | 2026-10-16 (아카이브됨) | 받지 않음 |
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | 학습 환경 | NOASSERTION("Other") | 8,700 | 2026-11-22 | 받지 않음 |
| [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions) | 논문 참고문헌 [34], 물리 기반 캐릭터·휴머노이드 학습 | Apache-2.0 | 2,512 | 2026-11-10 | 받지 않음 |
| [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed) (HF 데이터셋) | 학습 데이터. 142,220 시퀀스, 288시간(논문은 약 200시간 사용) | `bones-seed-license`(태그 other) | 다운로드 17,167 | 수정 2026-05-03 | 접근 승인(gated: auto). HF API의 저장 사용량 약 759 GB. 받지 않음 |

- 공식 저장소는 **코드가 아니라 홍보 페이지**입니다(`docs/`의 HTML, 이미지, 영상 60 MB). 따라서 `04`는 코드 대조가 아니라 **주장 검증**으로 씁니다.
- 100STYLE(Mason et al. 2022)은 외부 평가 모션이며 이 저장소에서는 받지 않았습니다. 라이선스는 확인하지 않았습니다.

## 4. 구성 요소별 출처

| 구성 요소 | 논문이 밝힌 출처 | 비고 |
|---|---|---|
| PPO, GAE | Schulman 외 2017 | 표 A3: 5 에폭, 4 미니배치, γ 0.99, λ 0.95, 클립 0.2, 학습률 적응(KL 0.01) |
| 인과 Transformer + KV 캐시 | Vaswani 외 2017 | 창 32, 깊이 3 |
| 추적 보상군(루트·몸체 위치·방향·속도 가우시안 커널) | 부록 A.1은 출처 미표기 | BeyondMimic 공식 설정과 동일 (`03` K9) |
| 종료 조건 0.25 m, 0.8 | 부록 A.2는 출처 미표기 | BeyondMimic과 동일 값 |
| 실패율 기반 적응 샘플링 | [4] BeyondMimic, [5] SONIC | 1초 구간, 균등 혼합 0.1, 상한 50배(표 A4)는 수정·확장 |
| 교차 신체 인터페이스(공유 백본 + 신체별 입출력) | [17] HPT, [18] CrossFormer, [19] X-VLA | 신체별 인코더·디코더를 로봇 ID로 선택 |
| 리타게팅 | [37] GMR | "평활화와 로봇별 파라미터로 적응" |
| 데이터 | [8] BONES-SEED, 외부 평가 [40] 100STYLE | 약 200시간, 50 Hz |
| 시뮬레이터 | [39] Isaac Lab | 로봇 9종 한 환경 |
| 새로 제안 | 사람 중심 명령 토큰 정렬(식 2의 쌍별 MSE), 다중 로봇 한 배치 PPO | 정렬 손실의 정규화 방식(L2 정규화인지 층 정규화인지)은 미기재 |

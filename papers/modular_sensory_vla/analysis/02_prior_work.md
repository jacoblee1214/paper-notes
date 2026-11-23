# 02. 선행연구와 관련 저장소

MoSS의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공개 자료가 무엇인지 정리합니다. 대괄호 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| MoSS의 요소 | 출처 | 관계 |
|---|---|---|
| 확산·flow matching 행동 전문가 + 행동 청킹 | π0 [23], GR00T N1/N1.5 [34, 6], ACT [22], Diffusion Policy [17] | MoSS가 얹히는 베이스. 식 1은 π0 계열의 조건부 flow matching |
| 층마다 스트림별 Q, K, V를 이어 붙이는 공동 어텐션, 스트림별 가중치 | Mixture-of-Transformers [33], Transfusion [32], Unified-IO [31] (부록 C "unified tokenization") | 모달리티별 가중치를 두고 어텐션만 공유하는 구조의 계열. 논문은 π0의 "VLM + 행동 전문가"가 이미 두 스트림 구조라는 점은 쓰지 않지만, MoSS는 그 구조에 스트림을 더하는 것으로 읽을 수 있음 |
| 교차 모달 융합 일반 | Attention Bottlenecks [26], Perceiver [28], BLIP-2 [29], MMA [30], 교차 어텐션 [27] | 부록 C. 시각-언어 융합 기법을 행동 예측으로 옮겼다고 서술 |
| 센서 모달리티별 스트림을 새로 붙이고 기존 가중치를 동결한 채 정렬한 뒤 전체 학습 | 논문에 인용 없음 | 어댑터 정렬 후 전체 미세조정이라는 일반적 학습 관행과 닿아 있으나 이 논문은 출처를 달지 않음 |
| 미래 센서 값 예측 보조 과제 | TA-VLA [13] (미래 토크 예측), 서론에서 직접 인용 | MoSS는 이것을 모든 센서 스트림의 flow-matching 손실로 확장(식 3) |
| 촉각을 VLM 입력 토큰으로 | Tactile-VLA [11], VTLA [14], OmniVTLA [15] | 기준선 Tactile-VLA의 계열 |
| 힘을 행동 전문가 쪽에 조건화 | ForceVLA [12] (MoE 융합), VLA-Touch [19] | 기준선 ForceVLA의 계열 |
| 관절 토크를 행동 전문가에 | TA-VLA [13] | 기준선 TA-VLA. 단일 트랜스포머에 토큰 추가 |
| 촉각·힘을 쓰는 VLA 외 정책 | Reactive Diffusion Policy [35], Tactile-conditioned DP [36], see-through visuotactile [37], AnySkin 정책 [38], Touch begins where vision ends [39], ForceMimic [40], Feel the Force [20], PP-Tac [21] | 부록 C. 단일 센서·고정 구성·처음부터 학습한 정책으로 구분 |
| 센서와 플랫폼 | AnySkin [25], DROID 구성 [24], RLBench [41] | 실험 환경 |

**가까운 대안과의 차이.** 가장 가까운 것은 (1) 단일 모달리티를 VLA에 넣는 기준선 세 가지, (2) 모달리티별 스트림을 공동 어텐션으로 묶는 MoT 계열 구조(MoSS가 센서에 적용)입니다. 이 논문의 새로움은 구조 자체보다 **"센서 모달리티를 추가해도 기존 행동 전문가를 그대로 두는 모듈 방식"과 그것을 두 모달리티 이상으로 확장했다는 점**입니다. 두 개 이상의 물리 신호를 한 VLA에 넣은 사례를 "처음"이라고 쓰는 문장(§1)은 이 논문 내 비교로 확인되는 범위가 촉각+토크 한 조합입니다(`04` C4).

## 2. 코드와 저장소 (GitHub 표)

2026-11-23에 `gh api`와 프로젝트 페이지로 확인했습니다. 일부 칸은 네트워크 오류로 확인하지 못해 "미확인"으로 적었습니다.

| 저장소 | 관계 | 라이선스 | 별 / 마지막 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|
| **공식 코드 없음** | 프로젝트 페이지의 Code 버튼이 "Code (Coming Soon)" | — | — | — | 코드 대조 대신 주장 검증(`04`) |
| [jiminlx/MoSS](https://github.com/jiminlx/MoSS) | 프로젝트 페이지(GitHub Pages)의 소스. 코드 검색에서 .py 파일 0건 | LICENSE 없음 (API `null`). 페이지 푸터는 템플릿(Nerfies)의 CC BY-SA 4.0 링크만 있음 | 1 / 2026-10-13 | 마지막 커밋 `47c0cb9b` ("Add hits counter to footer…"), 2026-04-30 커밋 메시지 "Update Code button label to Coming Soon" | 복제하지 않음. 페이지 HTML만 읽음 |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | 베이스 모델 저장소(논문은 GR00T N1.5 [6]) | Apache-2.0 | 8,411 / 2026-11-17 | 현재 태그는 n1.6, n1.7 등. N1.5 태그는 확인하지 못함 | 참조만 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 베이스 π0 구현(JAX) | Apache-2.0 | 14,743 / 2026-08-24 | — | 참조만 |
| [ft-robotic/ForceVLA](https://github.com/ft-robotic/ForceVLA) | 기준선 ForceVLA [12]의 공식 코드로 판단 | Apache-2.0 | 121 / 2026-01-27 | — | 참조만 |
| [ZZongzheng0918/TA-VLA](https://github.com/ZZongzheng0918/TA-VLA) | 기준선 TA-VLA [13]의 코드로 보임(이름 일치, 소유자 확인 안 함) | Apache-2.0 | 127 / 2025-10-25 | — | 참조만 |
| Tactile-VLA [11] | 기준선 | — | — | GitHub 검색에서 공식 저장소를 찾지 못함 (목록형 저장소만 나옴) | 미확인 |
| [raunaqbhirangi/anyskin](https://github.com/raunaqbhirangi/anyskin) | 촉각 센서 AnySkin [25] | MIT | 91 / 2024-11-06 | — | 외부 의존 |
| [enactic/openarm](https://github.com/enactic/openarm) | 손 로봇 플랫폼 OpenArm | Apache-2.0 | 3,718 / 2026-11-23 | — | 외부 의존 |
| RLBench [41] | 시뮬 벤치마크 | 미확인 | 미확인 | — | 외부 의존 |

- **공개 범위 요약:** 이 논문의 코드, 체크포인트, 시연 데이터는 공개되어 있지 않습니다(프로젝트 페이지 "Code (Coming Soon)", 논문 본문에 코드·데이터 주소 없음). 프로젝트 페이지는 과제별 비교 영상(GR00T N1.5 실패 대 + MoSS 성공)을 제공합니다.
- 재구현에 필요한 베이스(GR00T, openpi)와 기준선 일부(ForceVLA, TA-VLA)는 공개 저장소가 있습니다.

## 4. 참고

- 번호가 붙은 참고문헌은 `paper/paper.txt`의 References를 따릅니다.

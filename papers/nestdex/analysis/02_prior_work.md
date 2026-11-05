# 02. 선행연구와 계보

번호 [n]은 논문의 참고문헌 번호입니다. arXiv ID는 논문 참고문헌 목록에 적힌 값을 옮겼습니다. 각 논문을 따로 열어 검증하지는 않았습니다(§2의 ACT 가중치 규약은 예외로 아래에 적었습니다).

## 1. 계보도

```
[다지 손 텔레옵: 사람이 손가락까지 직접 지정]
  모션캡처 장갑 DexMV/Rajeswaran [7], DexCap [8], DOGlove [9]
  카메라 손 자세 추정 DexPilot [10], Robotic Telekinesis [11], AnyTeleop [4] ← 이 논문의 기준선이자 리타겟팅 출발점
        │
[손은 정책이, 팔은 사람이: shared autonomy]
  Cui et al. [12] arXiv:2511.00139 — VR 팔 텔레옵 + 자율 손 VLA로 데이터 수집   ← 가장 가까운 선행 연구(비교 없음)
        │
[기술 기반 계층 정책]
  options [13], 잠재 기술 공간 [14]–[16], SayCan [18], 교차 모달 지시 [19]
        │
  ★ NestDex: proprio 손 기술(BC) + 1-DoF 클러치 진행도 + VLM 기술 선택으로 수집
            → 수집 장치를 모두 뺀 outer visuomotor 정책(H-VAE 손 latent)으로 배포
```

**차별점 요약:** shared autonomy [12]처럼 손을 정책에 맡기지만, (1) 작업자가 1-DoF 클러치로 기술의 **진행도를 앞뒤로** 조절하고, (2) 단계 전환을 **VLM**이 고르며, (3) 그 장치들은 **배포에서 빠진다**는 점이 다릅니다. 계층 정책 연구 [13]–[19]에서는 선택기와 기술이 배포에도 남는데, NestDex는 수집에만 씁니다.

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 손 리타겟팅 | AnyTeleop [4] 벡터 리타겟팅 | 잔차를 Huber [28]로 바꾸고 여러 카메라 삼각측량을 더함(Eq. 3) |
| action chunk + temporal ensembling | ACT [27] (Zhao et al., RSS 2023) | Eq. 1–2. 가중치 w_i = exp(−m i), i = 0이 가장 오래된 예측으로 ACT 규약과 같음(ACT 원 논문의 서술과 대조, 내 사전 지식) |
| 잠재 행동 | VAE [26]; 관련 PLAS [20], LASER [21], VQ-ACE [22], BeT [23], VQ-BeT [24], Liconti et al. [25] | 손 명령 20 → 10차원(H-VAE), posterior 평균을 목표로 사용 |
| 시각 표현 | DINOv3 [30] arXiv:2508.10104 | outer policy 입력 임베딩 |
| BC | ALVINN [29] | Eq. 10 |
| 하드웨어 | Piper Nero 7-DoF 팔, WujiHand I 20-DoF 손 | leader-follower 플랫폼 |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| **DexScrew** (`dexscrew/`) | **가장 가까운 아이디어입니다.** 둘 다 "작업자는 팔(손목)을 조종하고 손가락은 학습된 기술이 맡는" 텔레옵으로 데이터를 모은 뒤, 그 데이터로 팔·손을 함께 제어하는 BC 정책을 학습합니다. 다른 점: DexScrew의 기술은 **시뮬레이션 RL**(회전 gait)이고 트리거로 켜고 끕니다. NestDex의 기술은 **리타겟팅한 사람 시연의 BC**(시연 10개)이고, 클러치로 진행도를 앞뒤로 조절하며, VLM이 여러 기술 중 하나를 고릅니다. DexScrew는 촉각을, NestDex는 관절 effort를 기술의 입력으로 씁니다. |
| **Contact-Grounded Policy** (`contact_grounded_policy/`) | CGP는 촉각을 예측해 compliance 목표로 바꿉니다. NestDex는 effort를 **inner policy 입력**으로만 쓰고, outer policy의 목표는 관절 위치입니다. 접촉력을 목표로 삼지는 않습니다. 공개 데이터에서 유지 노력이 성공을 거의 가른다는 결과(`03` §4)는 CGP처럼 힘을 목표에 넣는 방향의 근거가 됩니다. |
| **DexForce** (`dexforce/`) | DexForce는 kinesthetic 시연의 힘으로 "force-informed" 행동 목표를 만듭니다. NestDex는 effort를 기록하지만 행동 목표에 반영하지 않습니다. 두 방법을 합치면, 클러치 수집 시연의 effort로 목표 관절 위치를 보정하는 실험이 가능합니다. |
| **GPT-as-Tactile-Policy** (`gpt6_astra_tactile/`) | 둘 다 접촉 신호(촉각/effort)를 정책 입력으로 씁니다. GPT-as-Tactile-Policy는 LLM이 정책이고, NestDex는 VLM을 **기술 선택기**로만 씁니다. |
| **PaceVLA** (`pacevla/`) | 둘 다 action chunk를 어떻게 실행하느냐가 성능을 바꾼다는 문제를 다룹니다. NestDex는 temporal ensembling(Eq. 2)으로 부드럽게 하고, PaceVLA는 시간 양자화로 빠르게 합니다. NestDex의 앙상블은 오래된 예측에 가중을 두므로 접촉 뒤 반응이 늦어질 수 있습니다(`03` §3). |
| **GNR, FlashDexRetarget** (`gnr/`, `flashdexretarget/`) | 사람 손 → 로봇 손 리타겟팅 계열입니다. NestDex는 운동학 리타겟팅(AnyTeleop식)을 **기술 시연 수집**에만 쓰고, 물리 기반 리타겟팅은 쓰지 않습니다. 대신 실물 로봇에서 실행한 결과를 기록하므로 물리 정합성이 자동으로 보장됩니다. |

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| Real-Time Chunking(RTC, Physical Intelligence) | 공개 데이터의 조건 id가 `online_rtc`입니다. chunk 경계를 inpainting으로 잇는 방법이라 ACT식 평균과 다릅니다. 어느 쪽을 썼는지 확인이 필요합니다 | 공개 데이터 id + 내 사전 지식 |
| 공유 자율의 arbitration(사람·정책 명령 혼합) 연구 | 클러치 진행도는 "사람이 정책의 시간축을 쥐는" 특수한 arbitration입니다 | 내 해석 |
| 작업 부하 평가(NASA-TLX 등) | "작업자 부담 감소" 주장을 직접 재는 표준 도구입니다. 논문에 없습니다 | 내 사전 지식 |

**한계:** 4절과 ACT 가중치 규약은 내 사전 지식에 근거합니다. 게시 전에 원문을 직접 확인해야 합니다.

## 5. GitHub와 공개 자료 (2026-11-05 확인)

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | ★ | 마지막 push | 확인 커밋 |
|---|---|---|---|---|---|---|
| (NestDex 코드) | inner·outer policy, 클러치, H-VAE | 미공개 | — | — | — | 프로젝트 페이지·논문에 코드 링크 없음 |
| cheese-zj/nestdex-site | 프로젝트 웹사이트(React 정적 사이트, 영상, `paper.pdf`, 공개 데이터 `public/data/inner-policy-effort.json`) | 공식(웹사이트) | 표기 없음 | 0 | 2026-08-17 | `63f96e6` |
| dexsuite/dex-retargeting | AnyTeleop 리타겟팅 구현(Eq. 3의 출발점, 기준선) | 기준선 구성 요소 | MIT | 1,311 | 2025-08-21 | `3f56141` |
| tonyzhaozh/act | ACT, temporal ensembling(Eq. 2) 원 구현 | 구성 요소 출처 | MIT | 2,323 | 2024-07-23 | `742c753` |
| facebookresearch/dinov3 | 시각 임베딩 | 구성 요소 | DINOv3 License(GitHub 표기 NOASSERTION) | 11,682 | 2026-11-02 | `6876159` |
| wuji-technology/wujihandpy | WujiHand SDK(C++ + Python) | 하드웨어 | MIT | 40 | 2026-08-24 | `468e169` |
| agilexrobotics/piper_sdk | Piper 팔 SDK(Nero 전용 SDK는 찾지 못함) | 하드웨어 | MIT | 420 | 2026-10-14 | `b847e53` |

가장 가까운 선행 연구인 [12] arXiv:2511.00139의 코드 저장소는 abs 페이지와 검색에서 찾지 못했습니다.

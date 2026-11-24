# FM-VLA 선행연구, 구성 요소 출처, 공식 저장소

## 1. 계보

### 1.1 기반: π0.5 계열 VLA

FM-VLA는 π0.5 [4] 위에 지어집니다. π0.5는 PaliGemma [35]와 SigLIP [36] 비전 인코더, flow matching [37]·rectified flow [38] 행동 전문가를 쓰는 VLA로, π0 [3]의 후속입니다. 공개 구현은 openpi이고, 논문은 openpi 체크포인트에서 시작해 Zhiyuan Challenge 데이터로 15만 스텝 더 사전학습한 모델을 모든 실험의 출발점으로 씁니다(부록 C). RT-2 [1], OpenVLA [2], GR00T N1 [17] 등은 같은 문제를 "현재 관측 → 행동"의 무기억 매핑으로 다룹니다(§1, §2).

### 1.2 기억을 가진 VLA (시각·언어 중심)

- **MemoryVLA [5]**(ICLR 2026, arXiv:2508.19236): 지각·인지 기억 은행에서 과거 시각 토큰을 꺼내 확산 행동 전문가에 섞습니다.
- **MEM [6]**(Torne 외, arXiv:2603.03596): 단기 비디오 기억(최근 프레임을 시공간 분리 어텐션으로 인코딩)과 장기 텍스트 요약을 함께 씁니다. FM-VLA의 시각 기억 기준선 π-MEM이 이 구조를 π0.5 위에 다시 구현한 것입니다(부록 D).
- **그 밖의 계열 [20–27]:** ReMem-VLA(재귀 질의), EchoVLA(선언적 기억), VPWEM(작업·일화 기억), VQ-Memory, TempoFit(층별 KV 기억), Notes-to-self(스크래치패드), 이중 기억(CVPR 2026), 객체 중심 진행 상태. 논문은 이들이 모두 시각·언어 토큰을 다룬다고 정리합니다(§2).

FM-VLA의 차이는 기억의 **재료**입니다. 시각 변화가 거의 없는 접촉 사건(버튼 클릭, 닦기 왕복)을 힘 신호로 센다는 점입니다.

### 1.3 힘·촉각을 넣은 VLA (현재·단기 중심)

- **ForceVLA [7]**(NeurIPS 2025, arXiv:2505.22159): 6축 wrench를 VLM 뒤에서 전문가 혼합(MoE) 블록으로 융합합니다.
- **TA-VLA [8]**(CoRL 2025, arXiv:2509.07962): 토크 인지 VLA의 설계 공간을 정리하고 행동 디코더 쪽 토크 어댑터와 미래 힘 예측 보조 손실을 제안합니다. FM-VLA의 TA-VLA 기준선은 이 설계를 π0.5 위에 다시 구현한 것입니다(최근 0.9 s의 wrench 10프레임을 토큰 하나로, 부록 D).
- **후속 [28–34]:** ForceFlow, TacVLA, HapticVLA, AT-VLA, TaF-VLA, VTLA, DreamTacVLA. 논문은 이들이 현재 행동과 같은 시점의 짧은 힘 창만 쓴다고 정리합니다(§2).
- **같은 시기의 이력형 힘 주입:** 이 분석에서 별도로 읽은 LIFT(arXiv:2607.14236)는 힘을 GRU로 요약해 반응 전문가에 주입하고, FACT(arXiv:2608.01402)는 힘 이력 토큰을 씁니다(이력의 길이와 에피소드 전체를 덮는지는 확인하지 않았습니다). 두 논문은 FM-VLA(2026-07-20)와 비슷한 시기에 공개되었고, "힘 이력을 기억으로 쓴 첫 VLA"라는 서술은 저자의 "to our knowledge"가 붙은 주장으로 읽습니다(`04` C4).

### 1.4 FM-VLA의 위치

| 축 | 시각 기억 VLA | 힘 VLA (TA-VLA 등) | FM-VLA |
|---|---|---|---|
| 기억 재료 | 과거 RGB, 언어 요약 | 현재·단기(약 1 s) 힘 | 에피소드 전체 wrench + 짧은 관절 상태 창 |
| 주입 위치 | VLM 입력, 비전 인코더 | 행동 전문가 접두 토큰 | 행동 전문가 접미 토큰(잡음 행동 뒤) |
| 압축 | 프레임 선택, 요약 | 창을 선형 투영 | 동결한 VAE로 8토큰 |
| 추론 비용 | 프레임 수에 비례 | 거의 없음 | 거의 없음(+3.3 ms) |

## 1.5 구성 요소별 출처

| 구성 요소 | 출처 | 논문에서의 사용 |
|---|---|---|
| Perceiver-IO 인코더·디코더 | Jaegle 외 [10] | 잠재 질의 8개에 교차 어텐션, 시간 질의로 복원 (§3.2.2) |
| VAE | Kingma·Welling [9] | 평균·로그분산 헤드, 재매개변수화 (식 2) |
| free-bits KL | 사후 붕괴 방지 기법(Kingma 외, 2016 계열). 논문은 별도 인용을 달지 않음 | λ = 0.5 nats (식 4) |
| 푸리에 위치 인코딩 | Perceiver 계열 관례 | 32 밴드, f_max = 1500 (부록 E.1) |
| 분위수 정규화 | openpi(π0.5)의 q01/q99 정규화 | wrench와 행동·상태 정규화 (§3.2.2, 표 4) |
| 1차 EMA 평활 | 신호처리의 일반 기법 | α = 0.3 (§3.2.1) |
| 무작위 사전 패딩 | 이 논문의 설계 | 길이 지름길 제거 (§3.2.1) |
| 0 초기화 선형 투영 | 사전학습 모델에 새 조건을 안전하게 더하는 일반 기법 | 힘·상태 토큰 투영 (§3.2.3) |
| rectified flow 손실 | Liu 외 [38], π0.5 | 식 5 |
| GRU, Q-Former | Chung 외 [40], BLIP-2 [41] | 구조 절제 기준선 (표 1, 부록 E.4) |
| MEM 시공간 분리 어텐션 | Torne 외 [6] | π-MEM 기준선 (부록 D) |
| AgiBot G1, Zhiyuan Challenge | Zhiyuan Robotics [39] | 로봇과 사전학습 데이터 (§4.1, 부록 C) |

## 2. 공식 저장소와 관련 저장소 (확인 2026-11-24)

| 저장소 | 관계 | 라이선스 | 스타 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [qft-333/FM-VLA](https://github.com/qft-333/FM-VLA) | 논문 공식 저장소(프로젝트 페이지의 Code 링크) | 없음(API license null, LICENSE 파일 없음) | 7 | 2026-09-05 | **코드 없음.** README, `demo-fmvla.mp4`(17 MB), `demo-fmvla-preview.gif`, `assets/corl2026-banner.svg`만 있음. 커밋 `51c0aac35ac23dac035925f8a47b8e403b35b734`(2026-09-05), 정적 확인만 |
| [프로젝트 페이지](https://qft-333.github.io/FM-VLA-Page/) | 데모 영상, 과제 설명, 표 1 재게시 | 페이지 하단에 Nerfies 템플릿(CC BY-SA 4.0) 표기 | — | — | HTML을 정적으로 읽음. 새 수치는 없고 표 1과 같은 값 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 기반 모델 π0.5 구현·체크포인트 | Apache-2.0 | 14,745 | 2026-08-24 | 정적으로만 확인, 실행 안 함 |
| [ft-robotic/ForceVLA](https://github.com/ft-robotic/ForceVLA) | 관련 힘 VLA(ForceVLA [7]) 구현 후보 | Apache-2.0 | 121 | 2026-01-27 | 공식 여부 확인 안 함 |
| [ZZongzheng0918/TA-VLA](https://github.com/ZZongzheng0918/TA-VLA) | 기준선 TA-VLA [8] 원 저장소 | Apache-2.0 | 127 | 2025-10-25 | 정적으로만 확인 |
| [shihao1895/MemoryVLA](https://github.com/shihao1895/MemoryVLA) | 시각 기억 VLA [5] 구현 | 라이선스 표기 없음 | 383 | 2026-06-13 | 정적으로만 확인 |
| MEM [6] | π-MEM의 원 방법 | — | — | — | 공개 저장소를 찾지 못함(GitHub 검색 한 번) |

공식 저장소 README는 "FM-VLA is accepted to CoRL 2026"(2026-09), 개요 문단, BibTeX만 담고 있어 학습·평가 코드, Force-VAE 체크포인트, 데이터에 대한 언급이 없습니다. 이 저장소들은 설치하거나 실행하지 않았습니다.

## 4. 정리 (게시본에도 남는 절)

- FM-VLA의 기여는 기억의 재료를 힘으로 바꾼 것과, 긴 wrench 이력을 재구성 사전학습 VAE의 8토큰으로 줄여 행동 전문가 접미에 붙인 설계입니다. 지연은 +3.3 ms로 보고됩니다.
- 시각 기억 방식(MemoryVLA, MEM)은 과거 프레임을 쓰고, 힘 VLA(ForceVLA, TA-VLA)는 현재·단기 힘을 씁니다. FM-VLA는 그 둘 사이의 칸(긴 힘 이력)에 놓입니다. 같은 시기에 힘 이력을 순환 요약이나 이력 토큰으로 쓰는 논문들도 공개되어 "첫 VLA" 서술은 저자 판단으로 읽는 편이 좋습니다.
- 공식 GitHub 저장소는 있으나 코드가 없습니다(README와 데모 영상). 기준선의 구현(openpi, TA-VLA)은 Apache-2.0입니다.
- 구성 요소 대부분(Perceiver-IO, VAE, rectified flow, Q-Former, GRU)은 기존 기법이고, 새로움은 조합과 적용 대상(접촉 횟수 기억)에 있습니다.

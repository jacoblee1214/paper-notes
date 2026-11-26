# 02. 선행연구, 구성 요소의 출처, 관련 자료

## 1. 계보

ACE-Ego는 세 줄기의 연구를 한 틀에 넣습니다.

1. **교차 구현체 VLA 사전학습**: RT-1, RT-2, Open X-Embodiment(RT-X), Octo, OpenVLA, π0/π0.5, RDT, CogACT, GR00T N1/N1.6, X-VLA, HPT 등(논문 2.1절). 이 줄기는 로봇 시연 수집 비용에 묶여 있습니다. ACE-Ego의 비교 대상 중 JoyAI-RA, ABot-M0, FLARE, Motus, LingBot-VLA, Hy-VLA, DIAL은 2026년의 같은 계열 모델입니다.
2. **1인칭 사람 영상에서 행동 감독 얻기**: 표현·보상 학습(R3M, VIP, LIV, MVP 등)에서 출발해 손 궤적을 행동 대리로 쓰는 쪽(EgoMimic, EgoVLA, HumanPlus, OKAMI, Phantom, EgoZero, ZeroMimic, EMMA, H2R, EgoScale)으로 옮겨 왔습니다. 잠재 행동(LAPA)과 잠재 세계모델(DIAL)은 다른 갈래입니다(2.2절).
3. **손 복원 + 카메라 포즈**: HaMeR/WiLoR(국소 MANO), Dyn-HaMR(동적 카메라의 전역 4D), VIPE(카메라 포즈), SAM 3(추적)을 데이터 파이프라인 구성 요소로 씁니다(4.2절 Stage 3).

이 논문이 내놓는 차이는 두 가지입니다. 하나는 사람 손 의사 행동과 로봇 말단 행동을 **같은 카메라 좌표 형식**에 넣고, 형태 토큰과 물리 시간 청킹까지 붙여 하나의 정책에 먹이는 것입니다. 다른 하나는 사람 의사 행동을 **주 손실에서 빼고** 위치 채널 위주의 보조 손실로만 쓰는 설계입니다.

## 2. 구성 요소별 출처

| 구성 요소 | 논문의 서술 | 출처로 보이는 것 | 비고 |
|---|---|---|---|
| VLM | Qwen3-VL-4B-Instruct | Qwen3-VL | 코드 저장소는 토크나이저·설정 파일만 포함, 가중치는 별도 |
| 행동 전문가 | 흐름 정합 DiT 600M | GR00T N1 계열의 교차 주의 DiT(코드가 GR00T 데이터 경로를 따른다고 고지) | 층별 VLM 은닉 상태에 교차 주의하는 구조는 논문 본문에 없음(04 D5) |
| 델타 행동 청크 | π0를 따름(3.2절) | π0 | |
| 상태 입력 | 논문에 없음 | RDT2 방식의 타임스텝 조건화(코드의 `rdt2` 프리셋) | 04 D4 |
| 6D 회전 | [45] | Zhou 외 2019 | 식 12 |
| 손 복원 | HaMeR, 전역 최적화는 Dyn-HaMR에서 영감 | HaMeR, Dyn-HaMR | 식 10 |
| 카메라 포즈 | VIPE | VIPE | |
| 추적 | SAM3 | SAM 3 | |
| 형태 조건화 | URDF 그래프 + 대리 임베딩 | X-VLA의 소프트 프롬프트, HPT의 임베딩별 stem과 같은 문제의식 | 그래프 인코더는 이 논문의 것 |
| 시간 정렬 청킹 | 식 3–5 | 새로 제안했다고 서술 | 공개 코드에는 없음(04 D2) |
| 신뢰도 가중 Huber | 식 6–9, 20–24 | 로버스트 회귀의 일반 기법 | 공개 코드에는 없음(04 D7) |
| 데이터 | Ego4D, EgoExo4D, EPIC-KITCHENS-100, HOI4D, EgoDex, Xperience-10M | 각 데이터셋 논문 | 표 1 |
| 로봇 데이터 | AgiBot Alpha/Beta·DigitalWorld, Galaxea R1Lite, Galbot 자체, RoboCasa GR1 | | 자체 수집 Galbot 1,800+ h는 공개되지 않음 |

## 4. GitHub와 자료 (확인 2026-11-26)

| 자료 | 링크 | 라이선스 | 별 / 마지막 푸시 | 비고 |
|---|---|---|---|---|
| **공식 코드 (ACE-Ego-0)** | https://github.com/ACERobotics-VLA/ACE-Ego-0 | MIT (저장소 LICENSE) | 별 58, 포크 3 / 2026-09-14 | HEAD `4ee59acd25070dbc248b2ef44597ba1114693239`. 공개 범위는 GR1 RoboCasa 24 추론·평가와 RoboCasa 24·ARX SFT. 사전학습, 사람 영상 파이프라인, RoboTwin 평가, 실로봇 배포는 포함하지 않는다고 README가 밝힘 |
| 프로젝트 페이지 | https://acerobotics2025.github.io/ACE-Ego-0/ | - | - | 논문 요약, 영상. 2026-11-26 확인 시 "Code Coming Soon" 문구가 남아 있음 |
| 코드 보조 저장소 | https://github.com/acerobotics2025/ACE-Ego-0 | 라이선스 파일 없음 | 별 1 / 2026-07-23 | 프로젝트 페이지가 링크한 주소 |
| 체크포인트 | https://huggingface.co/ACERobotics/ACE-Ego-0 | Apache-2.0 | 내려받기 0 / 2026-09-14 | 사전학습 초기화 2개 + RoboCasa 24 정책 2개, 각 약 12.3 GB(합계 49.2 GB). 받지 않음 |
| RoboCasa 24 SFT 데이터 | https://huggingface.co/datasets/ACERobotics/Robocasa_Camera_Space_Eef | Apache-2.0 | 내려받기 231 / 2026-09-14 | 파일 48,126개, 약 53.4 GB. 받지 않음 |
| 자매 논문 코드 ACE-Ego-Hand | https://github.com/ggxxii/ACE-Ego-Hand | MIT(체크포인트는 README가 CC BY-NC 표기) | 별 149 / 2026-11-24 | arXiv:2608.20308, 같은 그룹의 1인칭 손 복원. ACE-Ego-0 파이프라인의 HaMeR 단계와의 관계는 논문에 없음 |
| GR00T N1 | https://github.com/NVIDIA/Isaac-GR00T | Apache-2.0 | 별 8,420 / 2026-11-17 | 행동 전문가와 데이터 경로의 원형 |
| RoboTwin 2.0 | https://github.com/RoboTwin-Platform/RoboTwin | MIT | 별 3,081 / 2026-09-24 | 평가 환경. 공식 ACE-Ego 코드에는 RoboTwin 평가가 없음 |
| starVLA | https://github.com/starVLA/starVLA | NOASSERTION | 별 3,977 / 2026-11-18 | 공식 README가 설치 설명에서 언급하는 상위 코드베이스 |
| HaMeR | https://github.com/geopavlakos/hamer | MIT | 별 1,199 / 2026-02-07 | 손 복원 |
| VIPE | https://github.com/nv-tlabs/vipe | Apache-2.0 | 별 2,222 / 2026-11-24 | 카메라 포즈 |
| Qwen3-VL | https://github.com/QwenLM/Qwen3-VL | Apache-2.0 | 별 20,261 / 2026-01-30 | VLM (모델 라이선스는 가중치 별도) |

논문 첫 쪽에 적힌 저장소·페이지 주소는 2026-11-26에 확인되지 않았습니다. 위 표의 `ACE-Ego-0` 주소는 저장소 README와 프로젝트 페이지가 서로 가리키는 주소입니다.

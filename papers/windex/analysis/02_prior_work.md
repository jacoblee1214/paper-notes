# 02. 선행연구와 관련 저장소

## 1. 계보

WinDEX가 풀려는 문제(사람 손의 미터 단위 세계 좌표 궤적)는 세 줄기에서 왔습니다.

1. **손 자세 추정 (단안, 이미지).** HaMeR, WiLoR, HaWoR, Dyn-HaMR(§II.C, 참고문헌 26–29). WiLoR는 WinDEX가 초기화에 쓰고 Table IV에서 비교합니다. HaWoR는 세계 좌표 비교의 기준선입니다. 단안이라 절대 스케일이 모호합니다.
2. **다중 시점과 웨어러블 기록.** POEM, POEM-v2(다중 뷰 스테레오, 참고문헌 30–31), UmeTrack(VR 헤드셋, 25), Project Aria, Quest(23–24). POEM 계열은 보정된 다중 뷰를 씁니다. 논문은 머리에 쓴 스테레오 한 쌍에 적용하려면 짧은 기저선, 어안 왜곡, 카메라와 손의 결합 움직임, 가림을 함께 다뤄야 한다고 적습니다(§II.C).
3. **데이터와 원격 조작.** DexCap(4), DexUMI(18), AnyTeleop(2), DexMV(5), ManipTrans(7), EgoVLA(9), Being-H0(10), Dexterous World Models(11). 대부분 장갑이나 외골격, 또는 실험실 RGB-D를 씁니다.

WinDEX의 차별점은 **(a) 어안 스테레오를 입력으로 쓰는 학습 모델, (b) 광학 mocap으로 독립 검증하는 평가 집합, (c) 반복 라벨링을 고정 임계값으로 묶는 데이터-모델 순환**입니다. 세 가지 모두 개별로는 선행 기법이 있습니다(알고리즘 1은 자기 학습과 모델 보조 주석의 변형, §II.D, 참고문헌 38–39). 새로운 것은 이 조합을 장치(CyberCap)와 함께 제시한 점입니다.

## 2. 가까운 대안과 구성 요소별 출처

| 구성 요소 | 논문이 쓴 출처 | 비고 |
|---|---|---|
| MANO 손 모델 | Romero 외 2017 (40) | 표준 |
| 어안 투영 모델 | Kannala & Brandt 2006 (41) | 보정 |
| VIO 세계 좌표 | cuVSLAM (43), VINS-Mono (44), ORB-SLAM3 (45) | cuVSLAM은 CUDA 가속. 이 맥에서 같은 구현은 불가 |
| 자세 사전 | DPoser-X (48) | 학습 사전이라 라벨 생성과 독립 |
| 특징 백본 | DINOv3 (49), VGGT-Omega (50) | Omega는 CVPR 2026 참고문헌 |
| 증류 | Hinton 외 (52), FitNets (53) | 중심화 코사인(식 18)은 이 논문의 변형 |
| 경량 인코더 | MobileNetV3 (54) | Stream |
| 손 키포인트 검출 | YOLO11-pose (46), ViTPose (47), Sapiens2 (51) | Table VI 비교 |
| 시뮬 조작 | ManipTrans (7) | Hand2Policy의 모방기 |
| 손 리타게팅 | dex-retargeting 계열 (1, 2), Wuji 벤더 구현 (55) | 실로봇 손 |

## 4. 공식 코드와 저장소

| 저장소 | 관계 | 라이선스 (GitHub API, 2026-11-22) | 최근 push | 별 | 상태 |
|---|---|---|---|---|---|
| WinDEX 공식 저장소 | — | 찾지 못함 | — | — | 검색 결과 없음 (`04` §1) |
| [wuji-technology/wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting) | 논문 [55]. Wuji 손 리타게팅 벤더 구현 | MIT | 2026-08-17 | 80 | 정적 읽기 대상 아님. 논문 요약에만 씀 |
| [rolpotamias/WiLoR](https://github.com/rolpotamias/WiLoR) | 논문 [27]. 초기화에 쓰임 | Other | 2026-11-19 | 746 | 실행하지 않음 |
| [ThunderVVV/HaWoR](https://github.com/ThunderVVV/HaWoR) | 논문 [28]. 세계 좌표 기준선 | Other | 2026-04-16 | 392 | 실행하지 않음 |
| [ManipTrans/ManipTrans](https://github.com/ManipTrans/ManipTrans) | 논문 [7]. Hand2Policy 모방기 | GPL-3.0 | 2025-10-10 | 350 | 실행하지 않음 |

프로젝트 페이지(https://fruitrabbit.github.io/WinDEX/)에는 코드 링크가 없습니다. 기준선 저장소 세 개는 라이선스가 제한적입니다(Other는 명시 조건 확인 필요, GPL-3.0은 파생물 공개 의무). 이 논문 재현에 기준선 코드를 쓰려면 라이선스를 따로 확인해야 합니다.

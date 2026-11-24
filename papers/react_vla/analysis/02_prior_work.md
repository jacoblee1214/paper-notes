# 02. 선행연구와 관련 저장소

참고문헌 번호는 논문(arXiv:2610.12007 v1)의 번호입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| 흐름 | 대표 연구 | REACT와의 관계 |
|---|---|---|
| 행동 청크 정책 | ACT [5], Diffusion Policy [6], π0 [4], π0.5 [7] | REACT의 기반. 청크 전체를 한 관측으로 생성한다는 점이 출발점(§1) |
| 청크의 관측 신선도 문제와 비동기 실행 | 비동기 추론 [48–52], DynamicVLA [50], Xiaomi-Robotics-0 [51], 네이티브 이어가기 [52] | 청크 생성과 실행을 겹치지만 관측 입력률이 청크 실행 주기에 묶인다는 것이 논문의 비판(§2) |
| 청크 경계 연속성 | RTC(실시간 청크, 확정 행동 고정과 inpainting) [8], 학습 시 행동 조건화 [9], REMAC [54], VLASH [55], Delay-aware DP [53] | Async+RTC 기준선이 [9]. REACT는 "하나의 롤링 버퍼가 연속 관측으로 계속 다듬어진다"는 점으로 구분(§3.1 마지막 단락) |
| VLA 추론 가속 | VLA-Cache [10], 토큰 가지치기 [11], EfficientVLA [12], Running VLAs at real-time speed [13], FASTER [14], 소형 VLA [35–39] | 지연을 줄이는 직교 방향. 논문은 "지연만으로 동적 제어를 설명할 수 없다"(§1) |
| 롤링·스트리밍 확산 (영상·모션) | Rolling Diffusion [56], FIFO-Diffusion [57], StreamDiT [58], Causal DiT [59], TEDi [60] | 노이즈 수준이 다른 버퍼를 한 칸씩 밀어내는 구조의 직접 출처 |
| 롤링 행동 버퍼 (로봇, 비VLA) | Streaming Diffusion Policy [61], Noise-Relaying DP [62] | 같은 발상을 확산 정책에서. 논문은 "VLA의 VLM 인코딩과 DiT가 결합된 병목은 다루지 않는다"고 구분(§2) |
| 토큰 단위 독립 노이즈 학습 | Diffusion Forcing [63] | 위치별 노이즈 수준으로 학습한다는 §3.3의 직접 출처(인용 [63]) |
| 학습 시 여러 flow 샘플 | MolmoAct2 [64] | "청크당 여러 timestep 지도"의 선행 사례. REACT는 결정적 격자로 고정(§3.3) |
| 벤치마크 | RoboTwin 2.0 [65] | 7과제 부분집합 사용(부록 C.6) |

## 2. 가까운 대안과 비교 대상

- **Async+Training RTC [9]**: 논문의 유일한 비동기·연속성 기준선. 부록 A는 이어 붙임 경계의 jerk를 지적합니다.
- **FlashVLA(arXiv:2608.27384, CoRL 2026)**: π0.5 위에서 계단형(스태거드) 노이즈 버퍼와 청크 단위 인과 마스크, RoboTwin 2.0 평가를 쓰는 가까운 동시기 연구입니다. REACT 참고문헌 목록에서 이 논문을 찾지 못했습니다(본문 검색). 두 논문의 선후와 차이는 이 분석에서 판단하지 않습니다.
- **FASTER [14]**: 처리량 분석의 기준으로 부록 B가 인용합니다(Φ_in, Φ_out 정의 방식).
- **VLASH [55]**: 미래 상태 조건화 비동기 방법. 논문 실험의 기준선에는 들어가지 않았습니다(관련 서술만).

## 4. 코드와 저장소 (GitHub 표)

조회일 2026-11-24, `gh api`로 확인했습니다.

| 이름 | 역할 | 라이선스 | 별 / 포크 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| [houlong-xiong/REACT](https://github.com/houlong-xiong/REACT) | 공식 구현 (openpi PyTorch 포크, 훈련·롤링 추론·DD·ARX X5 ROS2 예제) | Apache-2.0 (+ `LICENSE_GEMMA.txt`) | 4 / 1 | 2026-10-18 | 커밋 `eaad1d1` (clone으로 확인). 열린 이슈 1건(체크포인트 문의). 정적 읽기만 (`repo/`, 실행·설치 안 함) |
| [react-vla.github.io](https://react-vla.github.io/) | 프로젝트 페이지 | - | - | - | 요약, 영상, BibTeX. 코드 링크는 위 저장소 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 포크의 원본 (분기 커밋 `215abfb`, 2026-08-24) | Apache-2.0 | 14,745 | 2026-08-24 | 기반 코드 |
| [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | 시뮬 벤치마크 | MIT | 3,080 | 2026-09-24 | 벤치마크 코드 |
| [Physical-Intelligence/real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) | RTC 원 구현 (Kinetix) | MIT | 637 | 2025-12-08 | Async+RTC 기준선의 원리 참고. REACT 저장소에는 기준선 구현이 없음 |
| 체크포인트, 데이터 | 공개 여부 | - | - | - | 공식 저장소와 프로젝트 페이지에서 찾지 못했습니다. 저장소 이슈 #1이 같은 질문(2026-11-07)입니다 |

## 5. 학회 확인 기록

- arXiv 코멘트: "Accepted to CoRL 2026 as Spotlight."
- 프로젝트 페이지 헤더: "CoRL 2026 · Spotlight". GitHub README: "CoRL 2026 (Spotlight)".
- corl.org 공식 채택 목록(2026-11-14 수집본): 제목 완전 일치, spotlight, 세션 S2 "Efficient execution, world models and learning", 2026-11-09 월요일, 포스터 61번, OpenReview 포럼 FFB8qNJwV2.
- 결론: 공식 목록에서 확인했습니다 → `venue: "CoRL 2026"`.

## 6. 참고: 이 논문의 주장이 기대는 선행 결과

- 청크가 길수록 시간적 일관성이 좋고 짧을수록 반응이 빠르다는 대비(§1): ACT, Diffusion Policy 이래의 통념. 논문의 H50E50/H50E10/H10E10 표가 이를 같은 모델에서 재확인합니다(표 1, 2).
- 롤링 버퍼가 순수 잡음에서 깨끗한 행동까지 한 칸씩 간다는 구조 자체는 영상·모션에서 알려져 있습니다([56–60]). 새로운 점은 π0.5의 VLM 접두와 DiT를 분리해 처리량을 올리는 DD와, 이를 π0.5 체크포인트에서 이어 학습한다는 점입니다.
- "학습 시 여러 timestep을 지도하면 빨리 수렴한다"는 §3.3, §4.4의 주장은 MolmoAct2 [64]가 선행 근거로 인용됩니다. 이 분석은 이 주장의 인과를 따로 검증하지 않았습니다(03 C9).

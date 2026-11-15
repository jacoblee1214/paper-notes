# 선행연구와 관련 자료 (PACE, arXiv:2610.16308)

참조 번호 `ref n`은 .bib 파일의 키(`paper/src_v1/PACE_camera_ready_v5_refs.bib`)다. 논문 본문의 `[n]` 번호와 다르다(예: ACT는 본문 [11], 이 문서 ref 14). 이 문서는 조사일 2026-11-15 기준이다.

## 1. 문제 계보

PACE가 푸는 문제는 "접촉이 많은 삽입에서 촉각을 청크 정책에 어떻게 넣을 것인가"이다. 계보는 세 갈래다.

1. **접촉이 많은 조립의 고전 제어와 학습.** 임피던스, 하이브리드 힘/위치 제어, 연산 공간 힘 제어(ref 1–5)가 있다. 학습 쪽은 가이드 정책 탐색과 RL 조립(ref 6, 7, 10), 촉각과 힘을 쓰는 정책(ref 11, 43, 52, 53), 산업용 커넥터 학습(ref 12, 13)이 있다. 이 가운데 ref 11(Touch Begins Where Vision Ends)과 ref 53(TA-VLA)이 가장 가까운 최근 비교 대상이다.
2. **청크 정책.** ACT(ref 14)가 청크 정책의 기준이다. Diffusion Policy(ref 15)는 확산 기반 대안이다. π0(ref 24), OpenVLA(ref 32), FAST(ref 16) 같은 VLA가 청크를 쓴다. 청크 실행을 개선한 RTC(ref 20), REMAC(ref 22), DCDP(ref 21)는 청크 경계와 갱신 시점을 다룬다. PACE는 이 계열의 백본 위에 올라간다. 청크 경계 문제가 아니라 촉각의 역할을 다룬다는 점이 차이다.
3. **시각촉각 융합.** 토큰 결합(ref 38, 41), 교차 주의(ref 35, 40), 모달리티 가중(ref 43, 44), 예측 목표(ref 27, 55, 57, 58), 촉각 표현 사전학습(ref 25, 60)이 있다. 반응형 시각촉각 정책 RDP(ref 23)가 실행 중 촉각을 쓰는 가장 가까운 계열이다. 논문은 이 모두가 촉각을 **생성 시점의 한 입력**으로 본다고 주장한다. 이것이 PACE가 겨냥하는 "단일 경로" 설계다.

PACE의 독창적 주장은 촉각을 세 시간 역할로 나누는 것이다. 역할별 근거는 인간 촉각 신경생리(ref 45, 46, 47)에서 가져왔다. 이 신경생리 인용은 설계 동기일 뿐 실험 근거가 아니다.

## 2. 구성 요소별 출처

| PACE 구성 | 출처 | 메모 |
|---|---|---|
| 청크 백본 (ACT, CVAE-Transformer) | ref 14, 28, 29 | 백본 자체는 원 논문과 같다 |
| 시간 앙상블 | ref 14 (ACT) | 부록 A.6 |
| 상태 EMA와 FiLM 변조 | ref 33 (FiLM), ref 34 (LayerNorm) | 변조 식과 층별 적용은 논문 설계 |
| 사건 게이트 잔차 | 논문 설계 (인용 없음) | 잔차 정책 학습(ref 8, 9)은 bib에만 있고 본문 인용은 없다. 목적도 다르다 |
| 촉각 증분 예측 (Pred.) | 논문 설계. 시각촉각 세계 모델 계열 ref 27, 58 | 학습 때만 쓰는 보조 목표 |
| Diffusion Policy 이식 | ref 15 | 부록 D. DDPM(ref 30)은 bib에만 있고 본문 인용은 없다 |
| 기준선 π0 | ref 24, 31 | "미세조정"만 설명. 구현 세부 없음 |
| RDP 비교 | ref 23 | 같은 시연과 평가 조건으로 2과제 |
| 평가 벤치마크 | ref 39 (ManipulationNet) | 표준 peg-in-hole 간극 |
| 힘 피드백 원격조작 | ref 48–51 (FACTR, ACE-F, Haptic-ACT, Immersive Demonstrations) | 시스템은 논문 설계 |

요약하면 구성 요소 대부분은 기존 방식이다. 새로운 것은 **같은 촉각 스트림을 세 경로에 나누고, 각 경로의 위치를 생성 전·생성 후·학습 때로 정한 설계**다. 이 설계를 단독 기여로 볼지는 절제 실험으로 판단해야 한다(01 §4, 03 §2).


## 4. GitHub, 프로젝트 페이지, 관련 저장소 (2026-11-15 조회)

조회 방식: GitHub 공개 API 메타데이터(조직, 검색, 저장소 조회)와 프로젝트 페이지 HTML. 코드는 clone하지 않았다. 공식 코드가 없어서 clone 대상이 없다.

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | 별 | 최근 push | 대조 |
|---|---|---|---|---|---|---|
| [pace-insertion.github.io](https://pace-insertion.github.io/) (사이트) | PACE 프로젝트 페이지 | 공식 사이트 | 페이지 템플릿 CC BY-SA 4.0 (논문 라이선스 아님) | — | — | 페이지의 "Code (coming soon)" 확인 |
| [pace-insertion/pace-insertion.github.io](https://github.com/pace-insertion/pace-insertion.github.io) | 위 사이트의 저장소 | 공식 (웹사이트만) | 없음 | 0 | 2026-10-25 | 코드 아님 |
| (PACE 코드 저장소) | — | 없음 | — | — | — | 조직 `pace-insertion`에는 웹사이트 저장소 하나뿐 |
| [ManipulationNet/mnet_client](https://github.com/ManipulationNet/mnet_client) | ManipulationNet 공식 클라이언트 | 벤치마크 공식 | Apache-2.0 | 37 | 2026-07-18 | 대조 안 함 |
| [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | ACT (ref 14) | 원 저자 | MIT | 2339 | 2024-07-23 | 대조 안 함 |
| [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | Diffusion Policy (ref 15) | 원 저자 | MIT | 4731 | 2024-12-24 | 대조 안 함 |
| [xiaoxiaoxh/reactive_diffusion_policy](https://github.com/xiaoxiaoxh/reactive_diffusion_policy) | RDP (ref 23) | 원 저자 추정 | 표시 NOASSERTION (확인 필요) | 397 | 2026-04-12 | 대조 안 함 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0 (ref 24) | 원 저자 | Apache-2.0 | 14649 | 2026-08-24 | 대조 안 함 |

- PACE 자체 구현은 공개되지 않았다. 논문이 공개를 약속한 "code, models, and datasets"도 아직 없다.
- 부록 D의 DP+PACE는 원 DP 코드 위에 구현했다고 보이지만 공개되지 않았다.
- 원 ACT 저장소는 2024년 이후 push가 없다. 재현용 백본으로 쓸 때 버전 차이를 확인해야 한다.

## 5. 메모

- 기준선 π0 설명(부록 C.4)은 "같은 시연에서 미세조정"뿐이다. 프롬프트, 행동 표현, 학습 스텝은 없다. π0.5 같은 최신 버전이 아니다.
- RDP 비교(부록 E)는 "공개 구현 사용"이라고만 적혀 있다. 2-pin 19/30, Key 12/30은 PACE와 같은 조건인지 판단할 자료가 없다.
- 논문의 촉각 센서 GF225와 힘 피드백 장치 sigma.7은 상용 부품이다. 반면 ManipulationNet은 공개 벤치마크라 같은 간극 조건을 다른 로봇에서 재현할 수 있다.

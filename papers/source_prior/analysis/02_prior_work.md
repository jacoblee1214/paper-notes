# 02. 선행연구와 관련 저장소

이 문서는 LeaP의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드와 프로젝트 페이지가 어디 있는지를 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| LeaP의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 행동을 조건부 생성으로 학습하는 정책 | Diffusion Policy [1], DP3 [2], FlowPolicy [3], DDPM [4] | 모두 N(0, I)에서 시작합니다. LeaP가 바꾸는 부분입니다(§1) |
| 흐름 매칭과 선형 보간 경로 | Flow Matching [5], Rectified Flow [18], Stochastic Interpolants [19] | 식 1의 x_t = (1−t)z₀ + t a, u_t = a − z₀. 소스 z₀만 바꿉니다 |
| 흐름 기반 로봇 정책 | π0 [6], π0.5 [25], GR00T N1 [26], SmolVLA [27], AdaFlow [20], MP1 [21], ManiFlow [22] 등 | 같은 N(0, I) 소스. VLA에서의 효과는 논문이 시험하지 않았다고 밝힙니다(§7) |
| 관측에서 시작점을 만드는 정책 | A2A [7] (고유수용감각 시퀀스의 결정적 잠재), VITA [8] (시각 잠재), BridgePolicy [9] (확산 브리지, 시각+고유수용감각), Streaming Flow Policy [33] (직전 행동 중심의 좁은 가우시안, 분산은 고정 하이퍼파라미터이고 테스트에서 0) | 가장 가까운 대안. LeaP는 이들이 점 추정이거나 손으로 정한 분포라고 보고, 분산을 상태 적응형으로 학습한다는 점을 차별점으로 듭니다(§2) |
| 소스(잡음)를 정책 밖에서 조작 | DSRL [32] (고정된 정책의 잠재 잡음 공간에서 강화학습) | 독립된 소스를 학습하지는 않습니다 |
| 이미지 생성의 초기 잡음 최적화 | [28–31] (lottery ticket, 희귀 개념, ReNO, "A noise is worth diffusion guidance") | 개방 루프 합성이라는 점이 다릅니다 |
| 생성형 정책의 힘이 표현력이 아니라는 분석 | Pan 외 "Much ado about noising" [10] (ICLR 2026) | 반복 계산과 확률성 주입이 핵심이라는 주장. LeaP의 "확률적 소스" 동기입니다(§1) |
| 재매개화 | VAE [34] | 식 3 |
| 대조 정렬 손실 | CLIP [11], CPC [12] | 식 6 |
| adaLN 생성기 | DiT [35] | A.2의 SimpleFlowNet 변조 방식 (코드는 A2A에서 옴, `04` D10) |
| 점군 인코더 | PointNet [36], DP3 [2] | 모든 방법이 공유 |
| 확산 브리지 | UniDB [37], UniDB++ [38] | §5.4의 브리지 생성기와 BridgePolicy 재구현의 솔버 |
| 벤치마크 | RoboTwin 2.0 [13] | 15과제 선택 기준은 논문에 없음 |

**가까운 대안과의 차이.** A2A와 NoPrior는 LeaP와 **같은 SimpleFlowNet 생성기**라서 소스만 다릅니다(A.2). 따라서 표 1의 "LeaP 대 A2A"는 "상태에서 나온 결정적 잠재" 대 "상태 조건부 가우시안과 NLL·정렬 손실"의 비교이고, 손실 구성도 함께 바뀝니다(A2A는 자체 오토인코더 손실을 씀, 표 3 캡션). VITA는 "공식 conditioning-free 구성"이라 생성기 입력이 다릅니다. 이 차이는 비교에서 소스 이외의 변수입니다(`03` §4 C3).

## 2. 코드와 저장소 (GitHub 표)

논문 PDF에는 코드나 프로젝트 링크가 없습니다. 공식 저장소와 페이지는 GitHub 검색으로 찾았고(`gh search repos "learnable source prior"`), README와 프로젝트 페이지가 서로를 가리키며 같은 제목과 저자를 씁니다.

| 자료 | 링크 | 라이선스 | 별 / 포크 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| **공식 코드 (본 저장소)** | https://github.com/daimeipo/LeaP | MIT (LICENSE 파일 있음, A2A 유래 부분은 Apache-2.0) | 2 / 1 | 2026-09-20 | `repo/` 얕은 clone, 커밋 `8eb3bc9` ("Host the official code and project website under daimeipo"). 정적 읽기만 |
| 조직 쪽 사본 | https://github.com/SEU-VIPGroup/LeaP | 파일 없음 (API license null) | 1 / 0 | 2026-09-20 | 같은 코드와 사이트("Use organization repository as a landing page"). daimeipo 쪽이 최신 커밋 `8eb3bc9`이고 이쪽은 `b32f2ad` |
| **프로젝트 페이지** | https://daimeipo.github.io/LeaP/ | MIT (저장소 `docs/`) | — | 2026-09-20 | 발표 영상(4분 11초), 슬라이드(PPTX 13.7 MB, 받지 않음), 결과 요약. "모델 가중치, 데모 데이터, 시뮬레이터 자산은 포함하지 않았다"고 명시 |
| CoRL 2026 공식 채택 목록 | https://www.corl.org/program/accepted-papers (`conferences/corl2026/papers.json`) | — | — | 수집 2026-11-14 | 포스터 74번, OpenReview `AO1shrWa8H` |
| RoboTwin 2.0 (평가 환경) | https://github.com/RoboTwin-Platform/RoboTwin | MIT | 3,074 | 2026-09-24 | LeaP 저장소는 커밋 `9b5491d`로 고정해 외부에서 받게 함 (구버전 `script/eval_policy.py` API) |
| A2A Flow Matching (SimpleFlowNet 출처) | https://github.com/JIAjindou/A2A_Flow_Matching | Apache-2.0 | 248 | 2026-06-01 | LeaP 생성기의 출처. 논문 [7]은 arXiv:2602.07322 |
| 3D Diffusion Policy (PointNet 인코더 출처) | https://github.com/YanjieZe/3D-Diffusion-Policy | MIT | 1,475 | 2025-10-17 | |
| FlowPolicy (연구판 골격 출처) | https://github.com/zql-kk/FlowPolicy | MIT | 193 | 2025-10-14 | THIRD_PARTY_NOTICES.md가 근거 |
| BridgePolicy 공식 구현 | https://github.com/jianghcsr/BridgePolicy | 파일 없음 (API license null) | 2 | 2026-05-26 | **논문 A.2는 "BridgePolicy has no public code"라고 쓰지만 이 저장소는 2026-05-25에 생겼고(LeaP v1은 2026-06-16), 설명은 "official implementation of BridgePolicy"입니다.** LeaP 실험 시점에 공개였는지는 확인하지 못했습니다(`04` D15) |
| VITA 공식 코드 | 찾지 못함 | — | — | — | `gh search`로 후보를 찾지 못했습니다. 논문 [8]은 ICLR 2026입니다 |

- 저장소 내용(추적 파일 60개, 작업 트리 약 48 MB, 대부분 `docs/assets`의 슬라이드와 영상. 파이썬 코드는 약 1,550줄): `leap/`(정책, 속도 네트워크, 흐름 매처, PointNet, 정규화), `leap/robotwin/`(데이터 변환, 학습, 체크포인트, 배포), `configs/leap.yaml`, `scripts/`(RoboTwin 설치·수집·평가), `tests/`(CPU 단위 테스트 12개, README와 VALIDATION.md의 기록).
- 소스 분포 이외의 비교 대상(NoPrior, A2A, VITA, BridgePolicy), 절제 변형, 실로봇 코드는 저장소에 없습니다(`04` D13).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- Pan 외 [10]의 "생성형 제어의 힘은 반복 계산과 확률성 주입에서 온다"는 분석이 LeaP의 동기입니다(§1). LeaP는 이를 "확률적이면서 관측으로 정보를 준 소스"로 구현했다고 말하지만, 그 분석을 이 실험에서 직접 검증하지는 않습니다.
- A2A [7]와 VITA [8]는 "관측에서 소스를 만든다"는 점에서 LeaP의 직접 선행입니다. 논문 §2는 A2A가 고정 분산 가우시안 잡음을 입력에 넣으면 "견고성은 올라가나 정상 과제 성능이 떨어진다"고 서술합니다. 이 서술은 [7]을 읽어 확인하지 않았습니다.

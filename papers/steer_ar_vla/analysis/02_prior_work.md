# 02. 선행연구와 관련 저장소

TS의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 코드와 프로젝트 페이지가 어디 있는지 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| TS의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 기반 정책 π0-FAST, DROID 체크포인트 | π0-FAST·FAST [14] (Pertsch 2025), π0 [5], DROID [13] | TS는 이 정책을 건드리지 않고 디코딩 루프만 바꿉니다. 실험의 모든 수치가 이 한 모델에 대한 것입니다(§3.3) |
| 행동 토큰화(DCT + BPE, 저주파 우선 평탄화) | FAST [14], 순차 모델링 관점 [28] (Decision Transformer), [29] (BeT) | §4.2의 "저주파 토큰이 전역 의미를 가진다"는 FAST의 평탄화 순서에서 온 해석입니다 |
| 자기회귀 VLA | RT-1 [1], RT-2 [2], OpenVLA [4] (행동을 칸 나눈 토큰) | TS는 "토큰 열이 곧 행동"이라는 구조를 이용합니다 |
| 확산 정책 조향 | DSRL [16] (잠재 잡음에 RL), DiSCo [19] (부분 시연 인페인팅), DynaGuide [15] | 가장 가까운 대안. 모두 확산 정책의 잡음/잠재/가이드 위에서 일하고 TS는 이산 토큰 위에서 일합니다(§2.1) |
| 공유 자율, 인간 개입 | Javdani [20], Reddy [21], Dragan [22], LILA [24], "No, to the right" [25], HG-DAgger [26] | 인간 입력을 정책과 섞는 계보. TS는 섞는 위치를 "토큰 접두"로 정합니다(§2.2) |
| 추론 시 인간 상호작용 조향 | Wang 외 [17] (ICRA 2025) | 같은 문제(정책을 다시 학습하지 않고 인간이 방향을 준다)를 확산 정책으로 풉니다 |
| 비동기 청크 실행 | RTC [34] (Black 2025) | §5.1에서 TS와의 결합을 미래 과제로 둡니다 |
| 브레인–컴퓨터 인터페이스 | [31]–[33] | §5의 동기. 이 논문 실험에는 BCI가 없고 키보드만 씁니다 |

**가까운 대안과의 차이.** TS는 학습 없이 한 번의 디코딩 루프 수정으로 끝나는 가장 가벼운 조향입니다. 대신 입력이 정책의 토큰 문법(FAST의 DCT 계수 순서)에 묶여 있고, 방향 이외의 의도(힘, 속도 프로파일)를 토큰으로 쓰는 방법은 보이지 않습니다. 확산 정책 조향과의 정량 비교, 또는 정책 출력에 인간 속도를 더하는 단순 혼합 기준선과의 비교는 논문에 없습니다(`03` C 항목).

## 2. 코드와 저장소 (GitHub 표)

2026-11-22 GitHub API(`gh api`)와 얕은 복제로 확인했습니다.

| 저장소 | 관계 | 라이선스 | 별 / 마지막 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|
| [jasontchan/openpi-fast-steering](https://github.com/jasontchan/openpi-fast-steering) | **공식으로 추정되는 구현.** 저자 개인 계정의 openpi 포크(README 제목 "Autoregressive FAST Token Editing for Inference-Time Steering"). 논문과 프로젝트 페이지는 이 저장소를 링크하지 않습니다 | Apache-2.0 (openpi 상속) + Gemma 라이선스 파일 | 0 / 2026-05-08 | `8fb0a577ba3ec815338696d7229a028bfffba2bc` (2026-05-08 커밋, 얕은 복제, 업스트림 기준 `fdc03f5`) | 정적 읽기만 (`04`). 실행, 설치, import 안 함 |
| [jasontchan.github.io/token-steering](https://jasontchan.github.io/token-steering/) | 프로젝트 페이지 (arXiv 초록의 링크) | 페이지 하단: 템플릿 Nerfies, CC BY-SA 4.0 | 저장소 `jasontchan/jasontchan.github.io` MIT, 0 / 2026-10-28 | HTML 한 번 읽음 | 영상·그림은 받지 않음. 표와 초록은 논문 v1과 같고 "w=2가 가장 좋은 경로 효율"이라는 문장이 추가됨(`03` K4) |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 포크의 원본. π0-FAST, DROID 설정, 서버/클라이언트 | Apache-2.0 | 14,731 / 2026-08-24 | 포크 기점 `fdc03f5`(2026-03-29)와 diff | 업스트림은 읽지 않고 포크의 diff만 읽음 |
| [HF physical-intelligence/fast](https://huggingface.co/physical-intelligence/fast) | FAST+ 범용 토크나이저 | Apache-2.0 | 좋아요 191 / 2025-01-16 | 메타데이터만 | 다운로드 안 함 |
| [droid-dataset/droid](https://github.com/droid-dataset/droid) | DROID 로봇 제어·데이터 수집 스택(키보드·IK 클라이언트가 붙을 자리) | 없음(API `null`) | 474 / 2025-09-15 | — | 이 논문의 클라이언트 코드가 어디에 있는지 공개 저장소에서 찾지 못함 |
| [jasontchan/robo_copilot_vela](https://github.com/jasontchan/robo_copilot_vela) | 같은 저자의 π0.5 + Franka 신경 공유 자율(2025-11) | 없음 | 1 / 2025-11-14 | — | 방법이 다름(조향 토큰 아님) |
| [MaxDu17/DynaGuide](https://github.com/MaxDu17/DynaGuide) | 확산 정책 조향 비교 대상 [15] | MIT | 72 / 2025-12-10 | — | 이 논문은 정량 비교를 하지 않음 |

저자 계정 검색(2026-11-22): 클라이언트(키보드 → 말단 속도 → 관절 속도 → `observation/arrows` 키)가 담긴 공개 저장소는 찾지 못했습니다. 코드 검색에서 `observation/arrows`는 openpi 포크의 서버 쪽에만 나옵니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- "저주파 계수가 전역 운동을 결정한다"는 FAST 논문이 평탄화 순서를 정한 이유(저주파 먼저 예측해야 롤아웃이 안정)와 같은 방향입니다. 그러나 FAST 논문이 보인 것은 학습 안정성이지 "그 위치의 토큰을 편집하면 의미 있는 조향이 된다"는 아닙니다.
- π0-FAST DROID 체크포인트가 이산 토큰 문법을 지키며 접두에 외부 토큰이 들어와도 정상 문법으로 계속한다는 점은 TS의 전제이고, 논문은 이를 그림 4a·7의 시퀀스로만 보입니다.

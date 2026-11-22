# 02. 선행연구와 관련 저장소

이 문서는 OaP의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드가 있는지를 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다. 조사일 2026-11-22.

## 1. 선행연구 계보 (구성 요소별 출처)

| OaP의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| VLM으로 목표나 비용을 만들어 MPC에 넣기 | Language to Rewards [12] (Yu 2023), VLMPC [7], Hess 외 [13] (IROS 2025) | 보상·목표를 LLM/VLM이 씁니다. OaP는 **고정 잔차 라이브러리 안에서만** 쓰게 해 탐색 공간을 줄입니다(§II-A) |
| 3D 가치 지도, 키포인트 제약 | VoxPoser [4], ReKep [5] | 기준선. ReKep은 잡은 키포인트가 엔드이펙터와 함께 움직인다고 가정하는 전방 모델을 씁니다. OaP는 접촉 동역학을 MPC에서 직접 시뮬레이션합니다(§I) |
| 단계별 비용과 VLM 수정 루프 | CoRAL [6] (RSS 2026) | 가장 가까운 대안. 1 Hz로 LLM을 갱신합니다. OaP는 실행 전에 프로그램을 고치고 실행 중 갱신은 하지 않습니다(§II-A, §IV-C) |
| 작업·동작 계획의 기호 명세 | AutoTAMP [10], LLM3 [11] | 시간 논리와 기호 행동으로 번역합니다. OaP와 같은 목표가 아니지만 "LLM이 쓰고 검사기가 거른다"는 틀이 같습니다 |
| 샘플링 MPC | MPPI [15], predictive sampling [14], CEM [16] | MPPI는 식 2의 가중 평균을 씁니다. 가중치는 λ로 조정합니다(§II-B, §III-C) |
| 접촉이 많은 MPC | DROP [1] (ICRA 2025), 접촉 상보성 MPC [2] | 같은 접촉 MPC 계열. 기준선이 아니라 배경입니다 |
| 잔차 비용 설계 | 잔차 기반 비용 설계 [14] (predictive sampling) | 식 3의 pseudo-Huber 변환과 식 4의 끝 항 dead band가 여기서 옵니다 |
| 실제-시뮬 변환 | 미분 가능 시뮬레이션 real-to-sim [23] (RA-L 2025), SAM 3D [17], Any6D [18], FoundationPose [20] | 디지털 트윈을 만드는 파이프라인 전체입니다 |
| VLM 물성 추정 | Qwen3-VL [19] | 밀도와 마찰을 외형에서 추정합니다. 이 추정이 실로봇 파지 실패와 연결된다고 저자가 봅니다(§IV-D) |
| 시뮬레이터 | MuJoCo [21], MuJoCo Warp [22] | 계산 모델과 롤아웃에 씁니다 |
| VLA 기준선 | OpenVLA [3] (CoRL 2025), OpenVLA-OFT [26] (RSS 2025), π0.5 [27] (CoRL 2025) | 학습된 정책 기준선. LIBERO 체크포인트를 미세조정 없이 씁니다(§IV-C) |
| 세계모델·보상 모델 | Robometer [9] (RSS 2026) | 실행 진행을 보상으로 씁니다. OaP와 다른 쪽(학습 기반 보상)입니다 |

**가까운 대안과의 차이.** 가장 가까운 것은 **CoRAL [6]**입니다. 둘 다 VLM/LLM이 단계형 비용을 만들고 MPC가 실행합니다. 차이는 세 가지입니다. (1) OaP는 비용을 **고정 라이브러리의 조합**으로 제한합니다. (2) 수정은 **실행 전 자기 검토와 시뮬레이션 피드백**에서만 합니다. (3) CoRAL은 공개 코드가 없어 OaP가 재구현했습니다(`01` §4.3). 논문 Table III에서 OaP 2라운드(36/40)는 CoRAL 재구현(31/40)보다 높지만, 검정으로는 구별되지 않습니다(`03` §2).

## 2. 코드와 저장소 (GitHub 표)

조회 방법: GitHub API를 `gh api`로 조회했고(2026-11-22), 웹 페이지로 교차 확인했습니다. 저장소 clone은 `--depth 1`로 한 번 했습니다.

| 저장소 | 관계 | 라이선스 | 생성 / 마지막 push / 별 | 대조 기준 | 상태 |
|---|---|---|---|---|---|
| [ZihaoLu001/OaP](https://github.com/ZihaoLu001/OaP) | 논문 제목과 같은 설명. 공식 구현 (Python 패키지 `oap`) | Apache-2.0 (LICENSE 파일 있음) | 2026-09-29 / 2026-09-29 / 2 | 커밋 `2132575` "Initial release" (2026-09-29, 단일 커밋) | `repo/oap_code/`에 얕은 clone. **정적 읽기만**(`04`). 실행, 설치, import 안 함 |
| [ZihaoLu001/oap-project](https://github.com/ZihaoLu001/oap-project) | 프로젝트 페이지 소스 (영상 파일 포함, 크기 38.7 MB) | GitHub API는 NOASSERTION. 페이지 하단은 CC BY-SA 4.0 | 2026-10-27 / 2026-11-19 / 0 | 클론하지 않음 (영상이 대부분) | 페이지 본문만 확인. 공식 결과의 근거가 아님 |
| [프로젝트 페이지](https://zihaolu001.github.io/oap-project/) | 저자 페이지. arXiv 링크, 코드 링크, BibTeX 포함 | 페이지 하단 CC BY-SA 4.0 | - | 본문 확인 | 학회 표기 없음. 결과 요약은 논문과 같습니다 |
| 공식 코드의 예시 설정 `oap_pipeline/config.example.json` | `"model": "gpt-5.6-sol"`로 논문 VLM과 같음 | Apache-2.0 | - | 정적 읽기 | `04` D1 참고 |
| 외부 의존 (SAM 3D, Any6D, FoundationPose, Qwen3-VL, MuJoCo-warp, GPT-5.6-sol) | 파이프라인 구성 요소 | 각각 다름 | - | 설치 안 함 | 가중치와 API 키는 받지 않음 (구현 가능성은 `04` §4와 README) |

**요약.** 공식 코드는 Apache-2.0이고 2026-09-29 단일 커밋입니다. 논문보다 6주 앞섭니다. 프로젝트 페이지 저장소는 2026-11-19에 마지막으로 갱신되었습니다. 논문 실험 결과를 담은 노트북, 표 생성 스크립트, 장면 자산은 저장소에 없습니다(`04` D10).

## 4. 공식 코드가 있는지에 대한 판정

공식 코드가 **있습니다**(`ZihaoLu001/OaP`, Apache-2.0). 따라서 분석 문서는 `04_code_verification.md`(논문 ↔ 코드, 차이 D1–Dn)입니다. `04_claim_verification.md`는 만들지 않았습니다.

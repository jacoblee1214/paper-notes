# 02. 선행연구와 관련 저장소

AIM의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드와 프로젝트 페이지가 어떤 상태인지 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| AIM의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 세계모델 구조 | Dreamer 4 [26], Diffusion Forcing [18], Stable Diffusion VAE [20], DiT [77], VideoRoPE [98] | 정책 데이터가 아니라 데이터를 연구하려고 "구조와 학습 레시피를 고정한 강한 기준 모델"로 씀(부록 D) |
| 제어 가능성이라는 문제 | Ctrl-World [27], PlayWorld [1], VLAW [2] | 행동 조건부 세계모델이 행동을 무시하거나 환각하는 문제. 가장 가까운 선행은 PlayWorld(자율 플레이 데이터로 보강)와 VLAW(실패 데이터) |
| 데이터 수집을 바꾼다 | 로봇 데이터 큐레이션 [3], DAgger [4], RaC 복구 [5], 능동 학습 [58, 59], Toneva 외 [60] | "데이터의 유용성은 모델과 독립"이라는 관점(§2)과 복구 기준선(부록 E)의 출처 |
| 점별 상호정보와 empowerment | Klyubin 외 [67], Salge 외 [68], Barber–Agakov 하한 [69] | 식 (3)은 한 단계 empowerment와 같다고 명시(§3.2). 정리 2의 하한이 Barber–Agakov |
| 상호정보 기술 학습 기준선 | DIAYN [54], DADS [44], METRA [55], MISL 분석 [56] | AIM의 점별 MI 항과 가장 가까운 비교 대상. 기준선으로 직접 구현(표 6) |
| 호기심, 예측 오차 | ICM [50], RND [51], 불일치 [53], Plan2Explore [45], 카운트 기반 [49] | ICM만 기준선으로 구현. 모델 불일치 계열(RND, 불일치, Plan2Explore)은 구현하지 않음 |
| 가장 가까운 동시기 연구 | Hansen & Wang [47] (예측 불안정성을 호기심 신호로 능동 수집) | 본문이 "contemporaneous"로 언급. 비교 실험은 없음 |
| 밀도 추정 | 흐름 정합 [17], FFJORD–Hutchinson [73, 74], 공동 증류 [19] | q̂, β̂, d̂의 정확한 우도를 학습 시간 자기 증류로 상각(§3.4) |
| 정책 개선 | IQL [71], AWR [72, 97], Q-chunking [96], 확산 정책 [70] | 조건부 상호정보 정책 개선을 "보수적 오프라인 RL"로 근사 |
| 신뢰 영역 | KL 제약 정책 개선 [81, 82, 102] | 정리 2 증명의 라그랑주 형태(η = 1)와 AWR 해석 |
| 적응 제어의 지속 여기 | Boyd–Sastry [65], Demetriou–Rosen [66] | 서론의 비유로만 인용 |

**가까운 대안과의 차이.** PlayWorld(자율 플레이로 얻은 목표 지향 롤아웃)는 "다양한 데이터를 더 모은다"는 점에서 가깝지만 수집 정책에 목적 함수가 없습니다. AIM은 수집 정책이 "현재 데이터셋이 설명하지 못하는 행동-결과"를 직접 찾습니다. 모델 불일치(앙상블)나 예측 오차 기반 탐험은 **모델**이 어려워하는 곳을 찾고, AIM은 **데이터**의 밀도에서 어려운 곳을 찾는다는 점이 저자의 구분입니다(§2: "the model and its data are largely separable").

## 2. 코드와 저장소 (GitHub 표)

2026-11-26에 GitHub API 메타데이터로 확인했습니다.

| 저장소 | 관계 | 라이선스 | 별 | 최근 push | 상태 |
|---|---|---|---|---|---|
| [princeton-prism/action-influence-maximization](https://github.com/princeton-prism/action-influence-maximization) | **공식 저장소.** 논문 초록의 Code 링크. README는 "Code coming soon!" | 없음 (LICENSE 파일 없음, API license: null) | 7 | 2026-10-23 | 대조 기준 커밋 `11e0da0a0b43ddb4c02f0140d4804cdc65c69101` (2026-10-23). 학습·수집 코드 없음. `docs/`는 프로젝트 사이트 소스와 기록 데이터. 정적 읽기만(`repo/`), 실행·설치 안 함 |
| [프로젝트 페이지](https://princeton-prism.github.io/action-influence-maximization/) | 위 저장소의 `docs/`를 GitHub Pages로 게시 | (저장소와 같음) | - | 2026-10-23 | 열림(2026-11-26). 그림 1·5, 선호 점수(표 1), 시드 1 기록 롤아웃 브라우저 |
| [nicklashansen/mmbench2](https://github.com/nicklashansen/mmbench2) | 동시기 연구 [47]의 저장소로 보이는 프로젝트(이름만 대조, 내용 미확인) | MIT | 121 | 2026-11-04 | 참고. 읽지 않음 |
| [EdanToledo/Stoix](https://github.com/EdanToledo/Stoix) | 부록 C의 Four Rooms PPO 전문가 학습 프레임워크 [84] | Apache-2.0 | 425 | 2026-03-18 | 참고 |
| [Farama-Foundation/Minigrid](https://github.com/Farama-Foundation/Minigrid) | Four Rooms 환경 [75, 76]의 후보 구현(참고문헌 [76]의 저자가 Minigrid 저자) | NOASSERTION (표준 SPDX 아님) | 2547 | 2026-11-23 | 참고 |
| [mani-skill/ManiSkill](https://github.com/mani-skill/ManiSkill) | Push-T 환경 [83] (ManiSkill3) | Apache-2.0 | 3552 | 2026-08-04 | 참고 |
| [buoyancy99/diffusion-forcing](https://github.com/buoyancy99/diffusion-forcing) | Diffusion Forcing [18] 공식 코드 | NOASSERTION | 1308 | 2026-07-06 | 참고 |
| [seohongpark/METRA](https://github.com/seohongpark/METRA) | METRA [55] 공식 코드 | MIT | 98 | 2023-10-15 | 참고 (AIM 기준선의 출처 구현이 이것인지는 논문에 없음) |
| [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4) | Dreamer 4 [26]의 비공식 구현 | MIT | 423 | 2026-07-09 | 참고 |

- 부록 D는 모든 모델을 JAX로 구현했다고 적습니다. 공식 저장소에는 JAX 코드가 없습니다.
- 사이트 소스에 있는 기록 데이터(`docs/static/data/`)는 논문 표 1을 시드 1로 다시 만든 값을 담고 있어 재계산에 쓸 수 있습니다(`03` V3, `04` §1).
- 이 저장소에서 따로 받지 않은 것: 영상(mp4), 포스터 이미지 등 사이트 미디어. 복제본에 포함된 것은 읽지 않았습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- Barber–Agakov 하한 [69]은 I(A; S'|S) ≥ E[log q(a|s,s') − log π(a|s)]이고 q가 참 사후분포일 때 등호입니다. 정리 2는 이 사실을 β_k로 만든 q_k에 적용합니다(`03` §5).
- KL 제약 정책 개선의 닫힌 형 해 π* ∝ β exp(r/η) [81, 82]는 표준 결과입니다. 정리 2 증명은 η = 1일 때 A.12의 형태가 이것과 같다고 씁니다.
- 한 단계 empowerment 목적 [67, 68]은 I(A; S'|s)를 최대화합니다. AIM의 식 (3)은 이를 ρ_eval 위에서 평균한 것이고, 새로운 점은 이를 **리플레이 데이터 밀도**로 근사해 수집 정책의 보상으로 쓴다는 것입니다.
- "데이터 유용성은 모델과 독립적"이라는 능동 학습 관점 [59, 60]은 AIM 서술의 동기이지만, 세계모델의 재현율이 이 관점을 따르는지에 대한 직접 검증은 논문에 없습니다(`03` K2).

# 02. 선행연구, 구성 요소의 출처, 자료와 학회

## 1. 문제의 계보

ForceHarness는 두 흐름이 만나는 자리에 있습니다.

- **frozen 모델을 로봇에 붙이는 harness.** GPT-6 Astra를 자세 인터페이스로 쓰는 연구(논문 refs [27], [43]), 의미 단위 행동(ref [5]), 가상 카메라 시점(ref [32]), 모델이 쓰는 도구와 코드(refs [13], [17]), 저수준 VLA 기술(ref [44]) 등입니다. 저자들은 이들이 주로 "보고 움직이는" 쪽을 돕고 힘은 거의 다루지 않는다고 봅니다(2절 첫 단락).
- **힘을 학습에 넣는 정책.** 힘을 입력으로 받는 시각운동 정책과 VLA(FoAR [15], ForceVLA2 [18], FORGE [23], Diffusion Policy [6], π0.5 [24], ForceVLA [39])는 시연과 가중치 갱신이 필요합니다.
- **힘을 추론 시점에 쓰는 언어 모델.** 힘을 문장으로 읽히는 방식(ref [45]), 힘 피드백이 있는 코드 생성(GenCHiP [4], ref [22]), 임피던스 조정(refs [40], [41]), 학습된 기술 위의 회복 행동(ref [25], FORGE-plus), 원시 wrench·촉각을 직접 주는 방식(refs [29], [34], [35], [38], [46]), FORCA(ref [28]). ForceHarness는 제어와 인지를 **함께** 두고, 프리미티브가 모델 아래에서 힘을 조절하며 모델은 보정된 물리량을 읽는다는 점에서 이들과 다릅니다(2절 마지막 단락).

고전 기반은 하이브리드 위치·힘 제어(Raibert & Craig, 1981, ref [26])와 임피던스 제어(Hogan, 1985, ref [16])입니다. 힘-토크 측정으로 적재 하중을 추정하는 방법(Atkeson 외, ref [1])과 접촉 위치를 찾는 방법(Bicchi 외, ref [2])도 있습니다.

## 2. 구성 요소별 출처와 논문의 차이

| 구성 요소 | 논문의 출처 | 논문에서 새로운 부분 | 확인 메모 |
|---|---|---|---|
| 관절 토크에서 외력 추정 (식 1) | 인용 없음 (표준 기법으로 서술) | 영점을 읽는 시점으로 해결(tare, settle, 변화량 보고) | 표준 최소제곱 식. 영점 드리프트 대응이 기여 |
| 임피던스 스프링으로 힘 만들기, 리드 제한 (식 2) | 임피던스 제어 [16] | 리드 상한을 프리미티브별로 두어 힘 상한을 간접 설정 | 상수는 부록 표 A2와 코드 일치(`04` §2) |
| admittance 힘 제어 (식 3) | 하이브리드 힘 제어 [26] (일반 개념) | 모델 결정 주기보다 빠른 루프, 사건 종료 | 이득과 속도 한계는 표 A2 |
| 사건으로 끝나는 도구 | 없음 (저자 설계) | 모델이 시간이 아니라 접촉 사건을 기다림 | 가드는 모델이 못 바꿈 |
| 물리량 추정 (식 4–5) | 인용 없음, 최소제곱과 오차 전파 | 작업 임계값 없이 물리량 + σ + 플래그로 반환 | 코드 공식과 일치(`04` §2) |
| 코드 생성 기준선 | Code as Policies [20] | 힘 API가 있는 한 번 생성 프로그램(부록 G.3, 원 CaP와 다름) | 5.4절 결과는 같은 모델과 표 6 조건 |
| 힘을 입력으로 쓰는 학습 기준선 | Diffusion Policy [6], π0.5 [24], ForceVLA [39] | HiddenPhys에서 같은 시드로 재학습 | 학습 시드 1개(부록 G) |
| ForceBench 벤치마크 | ForceBench [14] (arXiv:2610.19604) | 평가자는 그대로, 보고 시드 300–309 사용 | 학습 기준선 수치는 ForceBench Table I과 동일 |

## 4. GitHub와 자료 표 (확인 2026-11-21, GitHub API와 Hugging Face API)

| 자료 | 링크 | 라이선스 | 별 / 포크 | 마지막 활동 | 비고 |
|---|---|---|---|---|---|
| **공식 코드 (ForceHarness)** | https://github.com/showlab/ForceHarness | MIT (LICENSE 파일, GitHub 표기 MIT) | 48 / 4 | push 2026-11-18, 최신 커밋 `733407c` (2026-11-17 커밋 날짜) | `repo/`에 얕은 clone(depth 1), 정적 읽기만 |
| 프로젝트 페이지 | https://showlab.github.io/ForceHarness/ | - | - | - | 실로봇 영상, 결과 표(개발 시드와 보류 시드 모두 표시) |
| 데이터셋 | https://huggingface.co/datasets/Jack97/ForceHarness-Data | MIT (Hugging Face 태그 `license:mit`) | - | 수정 2026-11-17 | 시연 데이터(raw, LeRobot). 평가 결과 파일은 확인하지 못함. 받지 않음 |
| Hugging Face 논문 페이지 | https://huggingface.co/papers/2611.21099 | - | - | - | 저자 연결 확인 |
| ForceBench 저장소 | https://github.com/forcebench-platform/ForceBench | 표기 없음 (GitHub는 NOASSERTION, "Other") | 5 / 1 | push 2026-10-30 | 논문 기준 시뮬레이션 벤치마크(Isaac Lab). 받지 않음 |
| ForceBench 논문 | arXiv:2610.19604 | arXiv non-exclusive (abs 페이지 확인) | - | 2026-10-14 제출 | Table I의 학습 기준선 수치와 같음(`03` §2) |
| HiddenPhys 기반 | robosuite 1.5.1, MuJoCo 3.2.7 | 외부 라이선스 | - | - | 설치하지 않음 |
| 실로봇 기반 | Polymetis (Franka FR3) | 외부 라이선스 | - | - | 하드웨어 필요 |

공식 저장소의 학습 기준선 코드(`scripts/baselines/`)는 Diffusion Policy, π0.5, ForceVLA의 수정본을 담고 있습니다. 그대로 쓰면 논문과 같은 설정인지는 `04` §2에서 확인했습니다.

## 5. 학회 판정과 라이선스

**학회: 표기 없음 (arXiv).** 근거는 다음과 같습니다. arXiv 댓글은 "31 pages ... Project page"만 있습니다. 프로젝트 페이지와 저장소의 README, docs에서 CoRL, ICRA, RSS, NeurIPS, ICLR, ICML 등 학회 이름을 찾지 못했습니다. 로컬 `conferences/corl2026/papers.json`(755편)에도 2611.21099나 ForceHarness 항목이 없습니다. 따라서 `venue: "학회 표기 없음 (arXiv)"`로 둡니다.

**라이선스: non-exclusive distribution.** arXiv abs 페이지의 "view license" 링크는 `http://arxiv.org/licenses/nonexclusive-distrib/1.0/`을 가리킵니다(확인 2026-11-21). CC BY 4.0이 아닙니다. PDF 첫 쪽과 끝 쪽에서 ©, copyright, All rights reserved 같은 저작권 문구를 찾지 못했지만 라이선스 자체가 CC BY가 아니므로 논문 그림은 공개본에 넣지 않습니다. 공개본의 그림은 우리가 그린 그래프만 씁니다(`report/figures/own_*.png`).

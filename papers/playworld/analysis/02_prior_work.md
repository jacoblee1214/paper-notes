# 02. 선행연구와 관련 자료 (PlayWorld)

## 1. 계보

PlayWorld는 새 모델이 아니라 **학습 데이터를 바꾸는** 논문입니다. 모델은 Ctrl-World를 따르고, 새로운 것은 (a) 자율 놀이 수집 시스템, (b) CLIP 거리 기반 곡선, (c) 접촉 중심 평가 벤치마크입니다.

| 줄기 | 대표 선행 (논문 번호) | PlayWorld에서의 역할 |
|---|---|---|
| 행동 조건 영상 세계모델 | Ctrl-World [9], Cosmos Policy [10], WorldEval [11], Gemini Robotics 보고서 [14], IRASim [54], EnerVerse [55] | 모델 구조와 비교 대상. 사람 시연 데이터 미세조정이 기준선 |
| 영상 모델 백본 | Stable Video Diffusion [15] (Cosmos [1], Wan [2] 인용) | SVD UNet을 DROID로 사전학습한 체크포인트에서 시작(§3.3, §4.1. 저장소 문서는 UNet 크기를 1.5B로 표기) |
| 세계모델로 정책 평가 | WorldEval [11], WorldGym [19], 확장 가능한 정책 평가 [57], World-in-World [59] | 그림 7의 평가 설계(정책 순서와 성공률 상관)의 선례 |
| 세계모델 안의 RL·정책 개선 | DreamGen [8], VLAW [18], WMPO [50], DiWA [72], World-Gymnast [92] | 보통 합성 데이터 SFT. PlayWorld는 DSRL [90]로 모델 안 RL을 시도(§4.5, 부록 D) |
| 놀이 데이터 | Learning latent plans from play [36], From play to policy [37], 잠재 계획 오프라인 RL [60], 촉각 놀이 [64] | 사람이 조작한 놀이 데이터가 주류. PlayWorld는 로봇 자율 놀이를 세계모델 학습에 사용 |
| 자율 수집·자기개선 | Pinto & Gupta [69], Levine 외 [68], RoboCat [70], Walke 외 [71], 파운데이션 모델로 지시 수행 자율 개선 [38, 73], Tether [91](동시 연구) | VLM 제안기와 VLA 실행기 구성은 [38, 73]의 계열 |
| 정책 다양성의 원천 | LIBERO-Plus [77] | 지시 변주에 따른 행동 분산 증가를 다양성으로 사용(§3.2) |
| RL 알고리즘 | DSRL [90] | 확산 정책의 초기 잡음을 학습(부록 D 식 8–12) |

## 2. 구성 요소별 출처

| 요소 | 출처 | 논문 내 위치 |
|---|---|---|
| 모델 구조, 프레임별 행동 조건, 3시점 | Ctrl-World [9] (Cosmos Policy [10], Gemini Robotics [14]의 다시점 예측 참조) | §3.3 |
| DROID 사전학습 가중치 | Ctrl-World 체크포인트 [9], DROID 데이터 [31] | §4.1 |
| 놀이 실행기 | π0.5-DROID [76] | 부록 A |
| 과제 제안기 | VLM (GPT-4 보고서 [75] 인용) | §3.2, 부록 A |
| 정책 평가 대상 | 확산 정책 [86], π0 [87] | §4.4 |
| 곡선 학습 | 곡선 학습 일반 [80, 81], CLIP [82], K-means | 식 4–6 |
| 시각화 | t-SNE [83] | §4.2 |
| RL | DSRL [90]. 보상 모델은 직접 만든 진행도 모델 | §4.5, 부록 D |

## 4. 공식 코드와 관련 저장소 (2026-11-26 GitHub API 확인)

| 저장소 | 관계 | 라이선스 | 스타 | 마지막 push | 커밋 | 비고 |
|---|---|---|---|---|---|---|
| [irom-princeton/open-world](https://github.com/irom-princeton/open-world) | 프로젝트 페이지 Code 링크, "OpenWorld" 범용 플랫폼 | 없음(API license null, 저장소에 LICENSE 파일 없음) | 31 | 2026-07-30 | `457942696e322cb24b00450d854fbf9875e01888` (2026-07-20, 얕은 clone) | 학습·정책 평가·RL 틀. PlayWorld 전용 부분은 `04` 참조 |
| [robot-playworld.github.io](https://robot-playworld.github.io/) | 프로젝트 페이지 | 해당 없음 | - | - | - | CoRL 2026 표기, 영상 비교, Code·Data 링크 |
| Hugging Face `tennyyyin/playworld_dataset_preview` | 프로젝트 페이지 Data 링크 | 카드를 확인하지 못함 | - | 2026-03-29 | - | 파일 목록 메타데이터로 에피소드 6,868개(3개 폴더 2,260 / 1,993 / 2,615개 주석), 총 저장 용량 약 39.3 GB(체크포인트 1개 9.3 GB 포함). 받지 않음 |
| [Robert-gyj/Ctrl-World](https://github.com/Robert-gyj/Ctrl-World) | 세계모델 원형 | MIT | 587 | 2026-04-08 | - | open-world의 `CrtlWorld`가 이 코드를 가져와 씀(README 감사의 글) |
| [ajwagen/dsrl](https://github.com/ajwagen/dsrl) | DSRL 원저자 코드 | 없음 | 245 | 2025-08-05 | - | open-world README가 기반으로 명시 |
| [tenny-yinyijun/dsrl](https://github.com/tenny-yinyijun/dsrl) | 저자의 DSRL 포크(open-world 문서가 clone 지시) | 없음 | 0 | 2026-05-22 | - | 확산 정책 사전학습용(dppo 포함) |
| [irom-princeton/dppo](https://github.com/irom-princeton/dppo) | 확산 정책 학습 | MIT | 880 | 2025-02-04 | - | |
| [tenny-yinyijun/openpi](https://github.com/tenny-yinyijun/openpi) | openpi 포크(서브모듈) | Apache-2.0 | 0 | 2026-09-02 | - | π0 / π0.5 정책 |
| [arnavkj1995/WEAVER](https://github.com/arnavkj1995/WEAVER) | 서브모듈(자기회귀 세계모델 관련) | MIT | 48 | 2026-10-30 | - | |
| [robometer/robometer](https://github.com/robometer/robometer) | 저장소의 보상 모델(subprocess) | 없음 | 287 | 2026-08-13 | - | 논문의 진행도 보상 모델과 다른 도구 |

- 이름이 같은 저장소 [hku-sail/PlayWorld](https://github.com/hku-sail/PlayWorld)(스타 94)는 **이 논문과 무관합니다.** 에이전트 플레이어로 세계모델을 벤치마킹하는 다른 논문(arXiv:2608.13552)의 코드입니다.
- 논문 본문(v1–v3)에는 코드 주소가 없고, 프로젝트 페이지에만 링크가 있습니다.

## 5. 가까운 대안과 차이

- **정책 롤아웃을 데이터로 쓰는 방식** (Ctrl-World의 정책 개선 실험, WorldGym 계열): 특정 정책의 분포에 맞춘 데이터입니다. PlayWorld는 한 놀이 정책(π0.5-DROID + 언어 변주)에서 나온 분포가 여러 정책을 포괄한다고 가정합니다(식 2의 주장). 이 가정의 직접 검증은 정책별 오차 분해(어떤 정책에서 틀리는지)이지만 논문은 정책 6개의 분포만 보여 줍니다.
- **사람 놀이 데이터:** 같은 6시간에서 사람 놀이는 baseline 중 가장 나쁘게 나옵니다(그림 7). 저자는 "정책 행동과 맞지 않는 분포 다양성은 해가 될 수 있다"고 해석합니다(§4.4). 다양성이라는 변수와 수집 주체(사람 대 로봇)가 함께 바뀌어 있어 분리되지 않습니다.
- **불확실성 기반 탐색:** 부록 A는 모델 불확실성을 보상으로 쓰는 능동 수집을 시도하지 않은 이유(불안정, 운영 부담)를 적습니다. 비교 실험은 없습니다.

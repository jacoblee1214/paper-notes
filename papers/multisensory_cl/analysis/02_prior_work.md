# 02. 선행연구와 관련 저장소

MuSe의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 코드가 있는지를 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| MuSe의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 백본: 영상-행동 공동 모델 | UVA [21] (Li, Gao, Sadigh, Song, 2025), MAR [22] (Li 외, NeurIPS 2024) | MuSe는 UVA를 MAR-Large로 초기화한 모델을 그대로 쓰고 F/T 경로만 더합니다(§3.5, 부록 6.3) |
| F/T 센서와 수집 장치 | UMI-FT [2] (Choi 외, 2026) | 데이터 수집과 배포 모두 같은 장치(UR5e, WSG50, 손가락 CoinFT) |
| 적응 보상 제어 | Adaptive Compliance Policy [1] (Hou 외, ICRA 2025) | 예측 렌치에서 강성 행렬과 가상 목표를 복원하는 방식을 그대로 따름(§3.5) |
| 새 센서를 사전학습 정책에 붙임 | FuSe [8] (Jones 외, ICRA 2025, "Beyond Sight") | 가장 가까운 선행연구로 인용됨. 차이는 (a) 접촉이 많은 조작에 초점, (b) 새 모달리티가 비전 전용 과제로 전이되는지를 평가(§2) |
| 힘을 잔차 보정으로 씀 | CR-DAgger [20] (Xu 외, 2025) | 힘을 기존 정책의 잔차로 쓰는 점이 다름(§2) |
| 시각-촉각·오디오 정책 | ViTaCFormer [3], ManiWAV [4], Touch in the Wild [6], OSMO [7], 온라인 어드미턴스 잔차 학습 [5] | 새 모달리티를 처음부터 함께 학습하는 설정(§2) |
| 시각-촉각 세계 모델 | OmniVTA [9], VTAM [10] | 접촉 동역학 예측의 가치를 보임. MuSe는 여기에 "사후 추가" 설정을 더함(§2) |
| 지속 학습 | Thrun & Mitchell [11], Lesort 외 [12], LIBERO [13] | 센서 구성이 고정된 설정. MuSe는 센서 구성 자체를 바꿈(§1, §2) |
| 사전학습 VLA의 망각 억제 | [14] (VLA는 망각에 강함), Breaking lock-in [15], Knowledge insulating [17], Actions as language [18], RT-2 [16] | 정규화·가이던스·공동학습 계열. MuSe는 재생과 공동학습의 정신을 따르되 모달리티 마스킹을 더함(§2) |
| 경험 재생과 모달리티 마스킹 | 경험 재생(지속 학습 일반), 마스크 토큰 | 새 요소는 아니고 "새 모달리티 없음"을 마스크 토큰과 손실 끄기로 처리하는 조합입니다 |
| 미래 예측을 보조 목적으로 사용 | UVA의 영상-행동 공동 학습 | MuSe는 F/T 미래 예측 헤드를 같은 방식으로 추가 |

**가까운 대안과의 차이.** 논문이 직접 비교하는 방법은 Diffusion Policy 기준선(부록 6.2) 하나뿐이고, FuSe나 CR-DAgger와는 수치 비교가 없습니다. 비교 대상은 모두 같은 데이터 위에서 MuSe 저자가 만든 변형(No F/T, No Pretrain, No ER, Pretraining Only, DP)입니다. 따라서 "기존 방법보다 낫다"가 아니라 "내부 절제보다 낫다"를 보이는 실험입니다(`01` §4.3).

## 2. 코드와 저장소 (GitHub 표)

2026-11-24에 GitHub API(`gh api`)와 프로젝트 페이지로 확인했습니다. 별 수와 마지막 push는 그때 값입니다.

| 저장소 | 관계 | 라이선스 | 별 / 마지막 push / 최신 커밋 | 상태 |
|---|---|---|---|---|
| [jadenvc/multisensory_wm](https://github.com/jadenvc/multisensory_wm) | 프로젝트 페이지의 "Code" 링크가 가리키는 곳. README는 "Code to be released" 한 줄 | 없음 | 9 / 2026-07-02 / `6813201` ("Update README.md") | **공식 코드는 아직 없음.** 링크는 대기 페이지 |
| [jadenvc/multisensory-continual-learning](https://github.com/jadenvc/multisensory-continual-learning) | 프로젝트 페이지(`jadenvc.github.io/multisensory-continual-learning/`)의 소스. 템플릿 기반 정적 사이트(`content.js`, `index.html`, 영상) | Apache-2.0 | 0 / 2026-11-17 / `97b2949` | 모델 코드 아님. 사이트 문구 확인에만 사용 |
| [jadenvc/multisensory-world-model](https://github.com/jadenvc/multisensory-world-model) | 같은 계정의 다른 저장소. README는 LingBot-VA(arXiv:2601.21998)의 것이고 `compare_uva_lingbot_actions.py`, LeRobot 변환 스크립트 등이 있음 | Apache-2.0 | 0 / 2026-06-01 / `1d5fdeb` (2026-05-23) | 논문이 쓰는 UVA/MAR 구현이 아니라 LingBot-VA 기반으로 보임. 논문과의 관계는 확인하지 못함. 내용은 읽지 않음 |
| [real-stanford/UMI-FT](https://github.com/real-stanford/UMI-FT) | 데이터 수집·배포 장치와 소프트웨어([2]) | MIT | 113 / 2026-08-10 / `5e41969` | 공개. 하드웨어 전제 |
| [ShuangLI59/unified_video_action](https://github.com/ShuangLI59/unified_video_action) | UVA 공식 구현([21]). MuSe의 백본 | MIT | 419 / 2026-08-21 / `fe3dec8` | 공개. MuSe가 이 위에 F/T를 더한 판은 없음 |
| [LTH14/mar](https://github.com/LTH14/mar) | MAR([22]) 구현 | MIT | 1956 / 2026-02-20 / `c6d53f7` | 공개 |
| [yifan-hou/adaptive_compliance_policy](https://github.com/yifan-hou/adaptive_compliance_policy) | ACP 공식 구현([1]). 보상 제어 스택의 기반 | MIT | 145 / 2026-04-18 / `74d68ec` | 공개 |
| [yifan-hou/cr-dagger](https://github.com/yifan-hou/cr-dagger) | CR-DAgger 공식 구현([20]) | MIT | 98 / 2025-12-10 / `00700d6` | 공개 |
| [jadenvc/force_control](https://github.com/jadenvc/force_control) | 6D 어드미턴스 제어 라이브러리(C++). 저자 계정의 사본으로 보이며 논문과의 직접 연결은 확인 못 함 | MIT | 0 / 2026-10-02 / `460bc3b` (2025-07-17) | 공개 |

- 체크포인트와 데이터셋: 프로젝트 페이지와 README에 링크가 없습니다. 없다고 확정할 수는 없고 "링크 없음"입니다.
- FuSe([8]) 구현은 GitHub 검색 두 번으로 찾지 못했습니다(검색 한계, 미확인).
- 어떤 저장소도 복제하거나 실행하지 않았습니다. 이 분석에서 clone한 코드는 없습니다.

프로젝트 페이지 문구(확인용): CoRL 2026 표기 아래 "Pretrain to Adapt Spotlight", "NeurIPS Robot Learning with World Models Best Paper Winner"라고 적혀 있습니다. 두 문구는 채택 목록으로 확인하지 못했습니다. CoRL 2026 본 학회 채택은 공식 목록(포스터)으로 확인했습니다.

## 4. 이 논문이 기대는 가정

- **F/T가 사전학습 데이터에 이미 기록되어 있습니다.** 부록과 §4에서 사전학습 21과제에서도 F/T는 수집되었으나 학습에서 가렸다고 씁니다. "교차 모달 일반화"는 이 숨긴 신호를 정답으로 비교하기 때문에 가능한 평가이고, 실제 "센서 없이 모은 과거 데이터"에서는 정답이 없어 같은 방식으로 평가할 수 없습니다.
- **원 사전학습 데이터를 보관하고 재생할 수 있어야 합니다**(§5가 한계로 인정).
- **새 모달리티가 UVA의 토큰 시퀀스에 붙을 수 있는 형태여야 합니다**(F/T 6D 시계열). 이미지 같은 큰 모달리티나 촉각 이미지로 같은 설계가 통하는지는 실험하지 않았습니다.

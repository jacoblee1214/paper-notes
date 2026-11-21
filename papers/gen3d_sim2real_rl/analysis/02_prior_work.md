# 02. 선행연구, 구성 요소의 출처, 저장소와 코드

## 1. 계보: 이 논문이 이어 받는 흐름

1. **VLA 사전학습과 모방 미세조정.** OpenVLA [14], SpatialVLA [15], π0 [16], π0.5 [17]이 기반입니다. 모방 미세조정은 분포 밖 상태에서 약하다는 지적이 있습니다 [19, 20]. 이 논문은 π0(BridgeV2 사전학습)을 출발점으로 씁니다.
2. **VLA의 RL 미세조정, 실물.** ConRFT [3], π0.6 [4], 보상 모델이나 사람 라벨러를 쓰는 연구들 [21–26]입니다. 실물 RL은 장면을 몇 개만 씁니다. 이 논문은 그 "장면 과적합" 문제를 지적합니다(§1).
3. **VLA의 RL 미세조정, 시뮬.** SimpleVLA-RL [5]은 대규모 병렬 시뮬로 희소 보상 미세조정이 된다고 보입니다. 이 논문은 SimpleVLA-RL의 근거를 "장면 분포를 넓히는 것"으로 옮깁니다. ReBot [27]은 실제-시뮬-실제 영상 합성을, VLA-RFT [28]은 세계 모델 안의 RL을 씁니다.
4. **흐름 정책의 RL.** ReinFlow [7]은 흐름 정책에 학습 가능한 노이즈를 넣어 온라인 RL을 합니다. DPPO [8]와 "Much ado about noising" [9]은 확산형 정책에 다중 모달이 꼭 필요하지 않다고 주장합니다. 이 논문의 K = 1 주장은 [9]의 방향과 맞습니다. PPOFlow는 ReinFlow의 PPO 변형입니다.
5. **일관성 정책과 흐름 정책의 단일 단계화.** Consistency Policy [10, 11], ConRFT [3]이 단일 단계 추론을 씁니다. 이 논문은 같은 효과를 K = 1 PPOFlow로 얻습니다(부록 B).
6. **3D 세계 생성.** TRELLIS [38], WorldGen [39]은 정적 자산 중심입니다. Holodeck [40], RoboGen [41]은 상호작용 장면을 절차적으로 만듭니다. EmbodiedGen [42, 43]은 자연어로 상호작용 3D 장면을 만들고, 이 논문의 생성 백엔드입니다.
7. **시뮬-실물 전이.** 도메인 무작위화 [29, 33], 실물-시뮬 장면 재구성 [36, 31, 37], 물리 파라미터 추정 [35]이 있습니다. 이 논문은 무작위화와 PD+중력 보상을 씁니다.

**가장 가까운 대안.** (a) 실물 RL로 장면 몇 개에 맞추기 [3, 4, 21–25]. (b) 수작업 시뮬 장면(SimplerEnv [48] 3장면)으로 학습하기. 논문 표 1은 (b)가 자기 장면에서 96.7%이지만 생성 장면과 외부 세트에서 36–38%로 떨어진다고 보입니다. 이 비교가 "장면 다양성"의 핵심 근거입니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 이 논문에서 |
|---|---|---|
| 시뮬레이터 | ManiSkill 3 [44] | GPU 병렬 시뮬(RTX 계열 전제) |
| 장면 생성 | EmbodiedGen [42, 43] | 텍스트 → 자산 → 장면 조립. 확장했다고 씀 |
| 자산 생성 | EmbodiedGen 내부 생성기(논문이 모델명을 밝히지 않음), Hunyuan3D-2.1 [50] (외부 평가 자산) | 외부 평가 자산은 학습에 쓰지 않음. TRELLIS [38]는 관련 기법으로만 언급 |
| 장면 설계기 | GPT-4o [45] | 과제 → 장면 그래프, QA 루프 |
| 사전학습 VLA | π0 [16] (제3자 open-pi-zero 체크포인트), BridgeV2 [12] | πpre |
| RL 알고리즘 | PPO [6], GAE [46], ReinFlow [7] → PPOFlow | 식 3–7 |
| 중요도 비율 지수 | GSPO [47] | 식 6의 s |
| 외부 평가 | SimplerEnv [48], Google Scanned Objects [49], Hunyuan3D-2.1 [50] | 표 1 블록 2, 3 |

## 4. 공식 코드와 저장소

조사 경로와 판정 근거는 `analysis/04_claim_verification.md` §0입니다. 요약하면 **논문의 학습 코드는 공개되지 않았습니다**. 아래 표의 저장소는 모두 *관련* 저장소입니다. 이 논문 코드가 아닙니다.

| 저장소 | 관계 | 라이선스 | 별 | 최근 push | 상태 |
|---|---|---|---|---|---|
| [HorizonRobotics/gail](https://github.com/HorizonRobotics/gail) | 프로젝트 페이지 소스(GitHub Pages). 파일 트리에 코드(.py 등)가 없음. `scale-vla-rl/`은 HTML, JS, 영상, 이미지 | 없음 (API null) | 0 | 2026-11-07 | 트리와 메타데이터만 조회. 복제하지 않음 |
| [HorizonRobotics/EmbodiedGen](https://github.com/HorizonRobotics/EmbodiedGen) | 생성 백엔드 [42, 43] (이 논문이 확장) | Apache-2.0 | 726 | 2026-08-24 | 조회만. 논문 전용 코드 아님 |
| [allenzren/open-pi-zero](https://github.com/allenzren/open-pi-zero) | πpre 체크포인트 출처(부록 D). π0 재구현, PyTorch(torch 2.5.0 고정, Python 3.10) | MIT | 1,541 | 2025-01-31 | 조회만 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 공식 π0 / π0.5 (부록 D에서 비교용으로 미세조정) | Apache-2.0 | 14,725 | 2026-08-24 | 조회만 |
| [mani-skill/ManiSkill](https://github.com/mani-skill/ManiSkill) | 시뮬레이터 [44] | Apache-2.0 | 3,532 | 2026-08-04 | 조회만. README는 Linux를 주로 지원하고 macOS는 제한적이라고 씀 |
| [ReinFlow/ReinFlow](https://github.com/ReinFlow/ReinFlow) | PPOFlow의 원류 [7] (NeurIPS 2025) | MIT | 382 | 2026-04-24 | 조회만. 이 논문의 PPOFlow 구현은 아님 |
| [simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv) | 외부 평가 세트 [48] | MIT | 1,187 | 2026-09-30 | 조회만 |
| [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | 외부 평가 자산 생성기 [50] | GitHub 자동 판정 NOASSERTION (LICENSE 원문 확인 필요) | 4,514 | 2025-10-17 | 조회만 |
| [openvla/openvla](https://github.com/openvla/openvla) | 실로봇 격차 분석 이슈(부록 D 각주 1) | MIT | 7,327 | 2025-03-23 | 조회만 |

- 복제한 저장소는 없습니다. `repo/`는 이 폴더에 없고, `.gitignore`에만 적어 두었습니다.
- 논문 전용 코드가 없으므로 04는 코드 대조(D-항목)가 아니라 **주장 대조(C-항목)**입니다.

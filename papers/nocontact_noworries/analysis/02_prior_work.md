# 02. 선행연구와 관계

논문: NoContactNoWorries (arXiv:2606.24450v2, IROS 2026 채택). 참고문헌 번호는 논문 References의 번호입니다.

## 1. 계보

| 갈래 | 대표 연구 (논문 번호) | 이 논문이 가져온 것 |
|---|---|---|
| 촉각 센서 기반 in-hand 회전 | AnyRotate [2], "Rotating without seeing" [3] | 과제 설정(in-hand 회전), 평가 지표 CRR·CRA([3]의 표준 절차) |
| 시각 + 촉각 정책 융합 | General in-hand rotation with vision and touch [21] | 손목 RGB-D, 관절 상태, 촉각을 넣는 정책 구조의 비교 기준 |
| 자기 고정 시각 정책 | Visual dexterity [13], Learning deep visuomotor policies [12], Diffusion Policy [14] | 시각 전용 기준선의 맥락 |
| 시각-관절 정책의 실패 | "When would vision-proprioception policies fail?" [16] | 시각이 과제 단계별로 덜 쓰인다는 관찰 (Introduction, Related Work) |
| 시각 → 촉각 예측 / 촉각 표현 | 시각-촉각 교차 예측, 마스크 시각-촉각 사전학습 [22–28] | "시각에서 접촉을 맞힌다"는 방향. 논문은 이들이 실물 촉각을 입력으로 가정한다고 구분 |
| 보상으로만 촉각 사용 | UpVital [29] | 예측 모델을 보상 계산에만 쓰는 방식과 대비. 논문은 **예측값을 정책 관측에 직접 넣는다**고 차별화 |
| 보상 형성 | Ng 1999 [30] | 보상 형성은 목표를 바꿀 수 있다는 논거 |
| 시각 인코더 | AsymFormer [31] | 고정된 RGB-D 의미 분할 백본 (Φ) |
| 손 플랫폼·정책 | LEAP Hand [32], RMA [35] | 시뮬레이션 손, 파지 캐시, PPO 기반 정책의 보상 구성 |
| 힘·토크 추정 | TransForce [36] | 이진 접촉에서 힘으로 확장하는 미래 연구 (Limitations) |

## 2. 가까운 대안

### 2.1 촉각을 정책 입력에 넣는 방법 (논문이 언급)
- 촉각 센서를 직접 쓰는 in-hand 회전 정책 [2, 3, 21]. 하드웨어 비용과 취약성이 문제라고 논문이 말합니다(Intro).
- 시각-촉각 사전학습 [22–28]. 실물 촉각을 배포 시에도 입력으로 가정합니다. 논문은 이것을 차별점으로 씁니다.

### 2.2 접촉을 추정하는 다른 방법 (논문이 직접 비교하지 않음)
- **기하 기반 접촉 추정:** 논문의 기준선 (g) "Kinematic depth"가 이 갈래입니다. 운동학 + 깊이 근접도로 접촉을 예측합니다. 가림에서 무너진다고 보고합니다(Table II: 0.54).
- **관절 추종 오차 기반:** 논문의 (d) pose-delta와 Fig. 6이 이 갈래에 가깝습니다. 리더-팔로워 데이터를 다루는 연구 [저장소 `beyond_implicit_force/`]와 연결됩니다.

### 2.3 논문이 인용하지 않은 가까운 연구
- 접촉 정보로 정책 입력을 게이트하는 방법(예: FoAR류), 시각으로 촉각을 예측하는 Connecting Touch and Vision(arXiv:1906.06322, 2019). 이 목록은 `restacvla/analysis/02_prior_work.md`의 §2.2를 참고하면 됩니다. 이 논문은 이들을 인용하지 않습니다.

## 3. 이 저장소의 다른 논문과의 관계

| 저장소 폴더 | 논문 | 이 논문과 맞닿는 부분 | 차이 |
|---|---|---|---|
| `beyond_implicit_force/` | Beyond Implicit Force (arXiv:2607.14578, IROS 2026) | **관절 추종 오차 → 접촉 힘 (proprioception 단서).** P 서보의 정상상태 편차 δ = 힘/강성이 이 논문의 δ·c 항과 같은 원리입니다(`03` §5). 이 논문은 δ를 "잔차"로 쓰고 Fig. 6으로 상관만 보입니다. 그 논문은 구동 여유가 목표에 들어 있을 때 δ가 힘이 됨을 보입니다 | 이 논문은 학습된 분류기, 그 논문은 물리 모델과 실물 2×2 실험 |
| `contact_grounded_policy/` | Contact-Grounded Policy (arXiv:2603.05687, RSS 2026) | 촉각을 미래 예측과 접촉 일관성 사상으로 다룸. 예측값을 제어기 목표로 씀 | 그쪽은 다지 촉각 정책 학습, 이 논문은 접촉 추정기를 분리 |
| `reactive_grasp/` | Real-Time Force Regulation for Whole-Hand Grasping (arXiv:2609.30082) | 촉각 없이 메쉬·추적 자세·관절각으로 접점을 계산(기하 기반) | 메쉬가 필요하고 학습이 없음. 이 논문은 메쉬 없이 학습 |
| `gpt6_astra_tactile/` | GPT-as-Tactile-Policy (arXiv:2610.28457) | 고정된 큰 모델을 촉각 정책으로 씀. 논문의 future work("pseudo-tactile 기반 모델")와 같은 방향 | 언어 모델 기반, 이 논문은 작은 학습 헤드 |
| `restacvla/` | ResTacVLA (arXiv:2607.03387, IROS 2026) | 영상으로 촉각을 예측하고 예측 오차(잔차)를 입력하는 점 | ResTacVLA는 촉각 센서가 있고 잔차를 VQ로 이산화. 이 논문은 센서 없이 이진 접촉만 예측 |

- 이 논문의 핵심 가설(시각 + 관절로 접촉을 맞힌다)은 `beyond_implicit_force/`의 δ 물리(관절 편차가 힘을 품는다)와 서로 보완합니다. 두 논문 모두 "관절 편차"를 접촉 단서로 봅니다. 그러나 이 논문은 편차의 속도 교란(명령 속도와의 결합)을 다루지 않습니다(`03` §6.2).

## 4. 구성 요소별 출처

| 구성 요소 | 출처 | 이 논문에서 새로운 점 |
|---|---|---|
| RGB-D 인코더 Φ (고정) | AsymFormer [31] | 고정하여 시간·자세 추론만 분리 |
| 시각 토큰 Ψconv | 논문 자체 | 두 번의 스트라이드 합성곱 |
| 자세 질의 ψ(q), ψcom(qcom) 교차 어텐션 | 논문 자체 | **현재와 명령을 다른 질의로 씀** ("query-asymmetry"). 가장 강조하는 설계 |
| 인과 transformer, 창 T = 8 | 표준 구조 | 이진 접촉 시계열 |
| BCE 손실 | 표준 | 프레임·접촉 지점 평균 |
| 시뮬레이션 접촉 라벨 | PhysX | 손가락 중심 4 지점, 실물 FSR 위치에 맞춤 |
| 정책 RL (GRU, PPO) | LEAP Hand [32]와 RMA [35]의 파지 캐시 | 접촉 추정기를 정책 관측에 연결 |
| 지표 CRR·CRA | [3] | 실물 CRA는 한 명의 관찰자가 수동 측정 |

## 5. GitHub와 공개 자료 (2026-11-09 확인)

논문의 공식 모델 코드는 없습니다. 프로젝트 페이지는 "Code (Coming Soon)"라고 씁니다(프로젝트 페이지 저장본). `gh search repos`와 `gh search code`로 제목, 저자, 논문 번호를 검색했습니다.

| 저장소 | 관계 | 라이선스 | ★ | 마지막 push | 대조 commit |
|---|---|---|---|---|---|
| `soham2560/no-contact-no-worries` | 저자의 **프로젝트 웹사이트 저장소**. index.html, 그림·GIF·영상, README만 있음. 모델 코드 없음 | 표기 없음 (웹사이트 본문에 CC BY-SA 4.0 표기) | 0 | 2026-06-24 | `f7c2b29` |
| `lucidrains/no-contact-no-worries` | 제3자 WIP "Implementation of NoContactNoWorries". README, LICENSE, .gitignore만 있음 | MIT | 4 | 2026-09-05 | `7f76a23` |
| `Fourier7754/AsymFormer` | 인코더 [31]의 AsymFormer 저장소. 논문 저자 공식 여부는 확인하지 않음 | Apache-2.0 | 57 | 2026-08-23 | `f1048c2` |
| `leap-hand/LEAP_Hand_Sim` | LEAP Hand v1 Isaac Gym 저장소. 논문의 시뮬레이션 플랫폼 [32]의 후보 | MIT | 215 | 2024-05-18 | `150bc3d` |
| `leap-hand/LEAP_Hand_Isaac_Lab` | LEAP Hand Isaac Lab 저장소 (대안) | 표기 없음 (API가 null) | 46 | 2025-09-04 | `c576e6a` |

- 검색 결과: `gh search repos "NoContactNoWorries"`는 `lucidrains/no-contact-no-worries`만 찾았고, `soham2560`의 저장소는 제목 검색으로 나오지 않았습니다. 이름으로 직접 조회해 찾았습니다.
- `gh search code "NoContactNoWorries"`는 요약·목록형 저장소(VLA-Handbook, Embodied-AI-Daily 등)와 저자 웹사이트 파일만 찾았습니다. 구현 코드는 없습니다.
- 공식 코드 저장소가 나오면 `repo/`에 얕은 clone을 받고 `04_code_verification.md`를 새로 씁니다.

## 6. 학회

- arXiv 댓글: "Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2026"입니다. 프로젝트 페이지도 IROS 2026을 씁니다. 따라서 **IROS 2026 (채택)**으로 적습니다.
- v1은 2026-06-23 제출, v2는 2026-11-06 개정입니다.

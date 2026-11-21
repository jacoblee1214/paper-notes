# 02. 선행연구, 가까운 대안, 관련 자료

## 1. 계보 (논문 본문과 참고문헌 기준)

| 축 | 대표 연구 (논문 인용 번호) | SemPhys와의 관계 |
|---|---|---|
| 텍스트 → 운동 생성 (kinematic) | MDM [33], MoMask [10], MotionGPT [19], T2M-GPT [39], TEMOS [28], MotionDiffuse [40], TMR 매처 [29], TM2T [9] | 생성기 계열. MDM·MoMask·MotionGPT3 [42]는 실험의 재학습 기준선 |
| 생성 → 실행 (추적 제어) | TextOp [35], ECHO [18], OMG [16], TeXedo [3] | 같은 G1 제어기로 실행해 비교하는 기준선. TeXedo는 "제어기 인지 테스트 시 스케일링", TextOp는 실시간 상호작용 |
| 실행 인지 생성 | RobotMDM (추적 보상 예측) [31], Sample-Simulate-Select [23], PhysMoDPO (실행 선호) [41], PhysDiff [38], Morph [20], ReinDiffuse [12], MaskedMimic [32] | 가장 가까운 대안. 모두 생성 후 선택·보상·선호 또는 제약으로 실행 정보를 쓴다. SemPhys는 생성기 파라미터를 실행 점수의 그래디언트로 직접 갱신 |
| 미분 가능 세계모델 + 제어 | SuperMimic [21] (동시 연구), 짧은 미분 롤아웃 [11, 15, 36], 미분 가능 시뮬 [36] | SemPhys는 SuperMimic의 "미분 가능 세계모델로 추적 보상을 역전파"를 가져와, 제어기를 고정하고 **생성기**에 적용 |
| 텍스트 기반 정렬 (그래디언트/보상) | DDPO [1], DRaFT [5], ReFL [30], Diffusion-DPO [34], MoDiPO [24] | 보상 정렬 계열. 보상이 실행 후 동작의 의미 점수라는 점이 다름 |
| 물리 기반 캐릭터 | DeepMimic [25], AMP [26], ASE [27], MoConVQ [37] | 배경. SemPhys는 직접 비교하지 않음 |
| 휴머노이드 추종과 전신 제어 | ASAP [14], ExBody2 [17], OmniH2O [13], HumanPlus [7] | 제어기 배경 |

**요점.** "생성 후 실행하고, 실행 결과에 따라 선택하거나 보상을 준다"는 계열(RobotMDM, Sample-Simulate-Select, PhysMoDPO, TeXedo)이 가장 가깝습니다. SemPhys의 차별점은 (1) 선택이 아니라 생성기의 투영 헤드를 갱신하고, (2) 전체 실행을 점수화하되 그래디언트는 창에서만 흐르게 하고, (3) 초기 생성기와 쌍을 이룬 앵커로 안정화하는 것입니다. 논문은 (2)에 대한 설명은 있으나, (1)과 (2)를 분리한 절제는 없습니다. "선호도 학습" 절제(48.54%)는 (1)과 같은 축이지만 구현 세부가 없습니다(`04` C10).

## 2. 구성 요소별 출처

| 구성 | 논문 출처 | 본 분석의 확인 |
|---|---|---|
| 텍스트 → 참조 생성, DiT 백본 | 논문 §3.1 (출처 표기 없음) | DiT 원 논문 인용 없음. 본문만으로는 확인 불가 |
| 매칭 거리와 R@k, MM | TMR [29], Guo et al. 2022 [8] | 평가 지표 설계의 출처. 매처가 GloVe 기반이라는 점은 §4.1에만 있음 |
| 미분 가능 세계모델과 짧은 롤아웃 | 식 9–14 (새 설계), 논문 [11, 15, 36]의 배경 | 설계는 새 것으로 보임 |
| 추종 제어기 | "SuperMimic 스타일 [21]" (§3.1, §4.1) | 공식 SuperMimic 체크포인트인지 여부 미확인 (`04` C-질문 6) |
| 데이터 | BONES-SEED [2] (2026-10-25 접근) | 데이터 자체는 이 분석에서 받지 않음 |
| 기준선 | 표 1 (각 논문) | 4개는 공식 체크포인트, 3개는 같은 데이터로 재학습 |

## 4. GitHub와 코드 자료 (확인 2026-11-21)

### 4.1 SemPhys 자체

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 상태 |
|---|---|---|---|---|
| 공식 코드 | **없음.** arXiv 본문, abs, HTML에 링크 없음. GitHub 검색(제목, 저자, arXiv 번호)에서 찾지 못함 | — | — | `repo/` 없음 |
| 프로젝트 페이지 | **없음** | — | — | — |
| 이름이 같은 저장소 | https://github.com/102637111/semphyspc (설명: "pointmamba变体") | Apache-2.0 | 별 0, 2026-11-04 | 이 논문과 무관해 보임. README 미확인 |
| 목록형 저장소 | awesome-humanoid-robot-learning (YanjieZe), Awesome-World-Model (LMD0311), Awesome-Humanoid-Robot-Learning (vergilshawn), robot-paper-daily (RainbowNebula) | — | — | arXiv 링크만 있음 |

### 4.2 기준선과 관련 저장소 (라이선스와 활동은 GitHub API)

| 방법 | 저장소 | 라이선스 | 별 | 마지막 push | 메모 |
|---|---|---|---|---|---|
| MDM [33] | https://github.com/GuyTevet/motion-diffusion-model | MIT | 4,150 | 2025-10-01 | 재학습 기준선 |
| MoMask [10] | https://github.com/EricGuo5513/momask-codes | MIT | 1,353 | 2024-09-13 | 재학습 기준선 |
| TextOp [35] | https://github.com/TeleHuman/TextOp | 감지되지 않음 (LICENSE 확인 필요) | 555 | 2026-11-18 | 공식 G1 체크포인트 사용 |
| PhysMoDPO [41] | https://github.com/Mael-zys/PhysMoDPO | 감지되지 않음 | 44 | 2026-10-26 | 실행 선호 계열. 논문 기준선이 아님 |
| MotionGPT3 [42], TeXedo [3], ECHO [18], OMG [16] | 이번에 찾지 못함 (검색어 2–3개로 시도) | — | — | — | 공식 체크포인트 사용 여부는 논문 주장뿐 |

- GitHub 검색으로 찾지 못한 저장소는 "없다"가 아니라 "찾지 못했다"입니다. 다음 단계로 저자 소속 기관의 GitHub 조직을 확인하면 됩니다.
- 이 표의 저장소는 모두 **읽기만 했고, 클론하거나 실행하지 않았습니다.**

## 5. 메모

- 이 논문은 같은 시기 발표된 SuperMimic(CoRL 2026)을 동시 연구(concurrent)로 인용합니다. 두 논문의 세계모델 기법이 가까워서, SuperMimic 쪽의 "경로 미분 대 정책 기울기" 비교 결과를 SemPhys의 핵심 가정(경로 미분이 생성기 학습에 적합하다)에 바로 대입해 볼 수 있습니다.

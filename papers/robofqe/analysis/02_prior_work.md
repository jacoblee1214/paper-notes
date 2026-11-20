# 02. 선행연구와 계보

번호 [n]은 논문의 참고문헌 번호입니다(`paper/paper.txt`). 각 논문을 따로 열어 검증하지는 않았습니다. 계보는 논문 본문의 인용과 저장소 코드를 기준으로 정리했습니다.

## 1. 계보

RoboFQE는 세 줄기가 만난 곳에 있습니다.

**(a) 오프라인 정책 평가(OPE).** 중요도 가중 계열(DR [31], 데이터 효율 OPE [32])은 행동 정책의 확률을 요구합니다. 로봇 데이터는 그 확률을 거의 기록하지 않습니다(§3.2). 정상 분포 보정 계열(DualDICE [33], 정규화 라그랑주 [34])은 밀도비를 최소최대로 추정하는데, 고차원 시각 관측에서 검증된 적이 없다고 논문은 지적합니다. 가치 기반 계열은 Q를 직접 회귀합니다. 고전적 FQE는 Ernst 외 [21]와 Le 외 [22]입니다. DOPE 벤치마크 [20]는 연속 제어에서 가치 기반이 가장 정확한 축에 든다고 보고했습니다. 논문은 이 결과에 기대어 FQE를 고릅니다.

**(b) 시각 OPE와 로봇 적용.** Irpan 외 [35]는 이미지 기반 파지에서 OPE를 적용했고, 분류 기반 순위 지표를 제안했습니다. Voloshin 외 [19]는 Atari와 격자 세계의 시각 OPE 벤치마크를 만들었습니다. Pavse & Hanna [36]는 상태-행동 유사도 표현을, Pavse 외 [37]는 양시뮬레이션(bisimulation) 기반 표현을 써서 시각 관측의 가치 학습을 안정화했지만, 평가는 상태 기반 과제에서 했습니다.

**(c) 로봇 정책 평가의 세 방향.** (i) 실세계 자동화와 분산 평가: AutoEval [8], RoboArena [9]. (ii) 시뮬레이터 재구성: SIMPLER [10](시스템 식별과 텍스처 베이킹), Polaris [25](3D 가우시안 스플랫), 불완전 시뮬레이터와 예측 기반 추론 [11, 26]. (iii) 영상 세계모델: WorldGym [12], Cosmos 기반 행동 조건 모델 [13], Gemini Robotics의 Veo 시뮬레이터 [14], Ctrl-World [15], 그리고 논문이 "동시 연구"로 든 WEAVER [29]와 SC3-Eval [30].

**(d) 보상과 성공 검출기.** LRM [38](대규모 로봇 보상 모델)과 Robometer [39](궤적 비교로 학습한 보상 모델)가 종료 판정에 쓰일 수 있다고 논문은 말합니다. RoboFQE는 LRM을 씁니다.

**이 논문의 위치.** 세계모델 평가와 시뮬레이터 평가는 새 궤적을 만들어야 합니다. RoboFQE는 기존 궤적에서 Q만 학습하므로 인프라 비용이 낮습니다. 대신 데이터 커버리지, 라벨 정확도, 외삽에 민감합니다(01 §4, §5).

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 논문에서의 역할 | 가까운 대안 |
|---|---|---|---|
| 가치 기반 OPE (FQE) | Ernst 2005 [21], Le 2019 [22] | 기본 추정기, 식 (4)–(6) | 중요도 가중 [31, 32], DualDICE [33] |
| FQE 선택 근거 | DOPE [20], Voloshin 2021 [19] | 가치 기반이 정확하다는 경험적 근거 | 세계모델 평가 [12, 13] |
| 사전학습 시각 표현 | DINOv3 [42] (DINOv2는 코드에 있음) | Q의 인코더, 부분 미세조정 | 처음부터 학습한 ResNet-18 (표 1 스크래치) |
| 종료 검출기 | LRM [38] (Qwen3-VL-8B 기반) | 롤아웃 라벨 | Robometer [39], 제로샷 판정 |
| 불일치 완화 | 없음 (지도 미세조정 또는 제로샷) | LIBERO 97.4%, 실로봇 83.3% | 정답 라벨 (코드의 `sim_rewards`) |
| 정책 | LIBERO ResNet-T BC, X-VLA [43], SmolVLA [6], π0.5 [44] | 평가 대상 | 실제 정책 |
| 벤치마크 | LIBERO [23], CALVIN [24], SO-101 실로봇 | 평가 환경 | 다른 벤치마크 없음 |
| 세계모델 기준선 | WorldGym [12], Cosmos 2.5-AC [13], Ctrl-World [15] | 비교 대상 | 논문 [14]의 Veo 기반 |
| 순위 지표 | MMRV, Pearson r (논문 인용: Li 외 [10], Quevedo 외 [12], Tseng 외 [13]) | 표 1–3, 6 | 분류 기반 순위 지표 [35] |

## 4. GitHub와 자료 (확인 2026-11-15)

| 자료 | 링크 | 라이선스 | 별 / 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (RoboFQE)** | https://github.com/prediction-action-lab/RoboFQE | MIT (`LICENSE`) | 별 0, 커밋 4, 최신 `a40a3bd` (2026-10-28) | `repo/`에 얕은 clone, **정적 읽기만**. LIBERO 실험만 공개 |
| 프로젝트 페이지 | https://yfdeng.com/robofqe/ | - | - | JS 앱. 본문 텍스트 없음. 논문, 코드, BibTeX 링크만 |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | 미확인 | 미확인 | 코드가 editable 설치를 요구. mujoco 3.1.6 고정 (`04` §4) |
| LRM 가중치 (USC-PSI-Lab) | https://huggingface.co/USC-PSI-Lab/LRM-models | 미확인 | - | 받지 않음. Qwen3-VL-8B 기반 (`fqe/lrm.py`) |
| LRM 참조 서버 | github.com/physical-superintelligence-lab/Large-Reward-Models | 미확인 | - | `fqe/lrm.py`가 프롬프트 출처로 인용. 확인하지 않음 |
| DINOv3 가중치 | facebook/dinov3-vits16-pretrain-lvd1689m (HF, gated) | DINOv3 라이선스 (미확인) | - | 받지 않음. 토큰 필요 |
| LIBERO 데모 | LIBERO 공식 스크립트 | (LIBERO) | 크기 미확인 | 받지 않음. Spatial만 해도 로더가 이미지 10 GB를 RAM에 올림 (`03` §1.5) |
| BC 체크포인트 | 저장소에 없음 | - | - | 학습 필요 (`04` D10) |
| 세계모델 기준선 | WorldGym, Cosmos 2.5-AC, Ctrl-World | - | - | Ctrl-World는 공식 코드(github.com/Robert-gyj/Ctrl-World, MIT)와 체크포인트(Hugging Face yjguo/Ctrl-World)가 공개되어 있음(2026-11-20 확인, 이전 판의 "미공개" 기재를 정정). WorldGym과 Cosmos 2.5-AC의 RoboFQE 실험용 설정은 미확인 |
| X-VLA, SmolVLA, π0.5 | arXiv 2510.10274 [43], [6], [44] | - | - | 미확인. 평가 정책 |

- 논문 프로젝트 페이지의 arXiv 링크 중 `2510.10125`는 본문에서 인용되지 않았습니다. 확인하지 않았습니다.

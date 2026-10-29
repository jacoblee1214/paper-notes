# 02. 선행연구와 계보

번호 [n]은 논문 참고문헌 목록의 순서(알파벳순, 본문 표에 쓰인 번호와 같음)입니다. arXiv ID는 논문 참고문헌에 적힌 값입니다. SAIL의 지표 함수, VLASH의 양자화 실행 루프, GR00T 저장소의 RoboCasa 과제 표는 코드를 직접 확인했습니다(§4). 나머지 논문은 따로 열어 검증하지 않았습니다.

## 1. 계보도

```
[action chunking 정책]
  ACT [36] → Diffusion·flow matching action head → GR00T N1.5 [5], RLDX-1 [18] (VLM + flow matching DiT)
        │  chunk를 시연과 같은 시간 해상도로 예측
        ▼
[시연보다 빠르게 실행하기]
  ├─ RL로 다듬기       SpeedTuning [34]                             — 전체가 고르게 빨라지는 경향
  ├─ 데이터를 가속    DemoSpeedup [13] arXiv:2506.05064             — proxy 정책의 행동 엔트로피로 구간별 다운샘플링
  ├─ 실행 시 조절     SAIL [2] arXiv:2506.11948                    — 운동 복잡도·그리퍼 사건으로 정해진 속도 사이 전환
  └─ 시간 양자화      Figure AI Helix 물류, VLASH [30] arXiv:2512.01031 — chunk를 고정 비율로 합침
        ▼
[모드를 고르는 라우팅]
  Switch Transformer [10] · MoH [15] · MoLe-VLA [35] (층 건너뛰기) · Mixture of Horizons [16] arXiv:2511.19433 (여러 horizon을 gate로 융합)
        ▼
  ★ PaceVLA: 시간 해상도와 horizon이 다른 decoder 중 하나를 상태마다 고름 (LVLM 라벨로 학습한 judge + 라우터)
```

**차별점 요약:** DemoSpeedup과 SAIL은 "어디서 빠르게 할지"를 행동 통계(엔트로피, 운동 복잡도, 그리퍼 사건)로 정합니다. 고정 양자화는 궤적 전체에 같은 비율을 씁니다. PaceVLA는 그 결정을 **장면을 보고 정책이 직접** 하게 하자는 것입니다. 다만 그 판단의 학습 신호는 사람(그리고 Claude)이 설계한 LVLM 질문지에서 옵니다(`01` §4.2-1).

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 기반 VLA | GR00T N1.5 [5], RLDX-1-IMG [18] arXiv:2605.03269 | VLM 고정, action expert·decoder·라우터·judge 학습 |
| flow matching | Lipman et al. [21] arXiv:2210.02747, DiT [28] | 식 1 |
| 시간 양자화 규칙 | VLASH [30]의 `action_quant_ratio` (코드 확인: 매 q번째 행동만 전송) | 절대 행동은 블록 끝값(식 5), delta는 SO(3) 합성(식 6) |
| 고정 양자화 기준선 | VLASH [30] | 기준 모델 출력에 q = 2 |
| 가속 기준선 | DemoSpeedup [13], SAIL [2] | SAIL은 재구현(delta 행동, action scheduling 끔) |
| 효율 지표 TPR | SAIL [2] `metric_fn.py`의 `throughput_regret_agg_fn` | 식 7. SAIL은 시뮬레이션 시간(초)에 분당 단위를 쓰고, PaceVLA는 행동 step 수와 ×10⁻³ 단위를 씀 |
| 효율 지표 ES | Dex-Net의 MPPH [23] | 식 8, 실패에 t_max를 매김 |
| 라우터 균형 손실 | Switch Transformer [10] | L_bal |
| 실시간 chunking | RTC [6] | ALLEX 컨베이어에서만 비동기 추론 |
| 벤치마크 | RoboCasa [26], LIBERO [22], DexJoCo [32] arXiv:2605.16257 | RoboCasa 24개 = PnP 8 + Others 16 (GR00T 저장소 표로 확인) |
| 라벨링 모델 | Claude Opus 5(기준 설계), Gemini 3.8 Flash(전체 라벨링) | 부록 C.5, E |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| **Chronos Policy** (`chronospolicy/`) | 둘 다 "정책이 시간을 어떻게 다루는가"의 문제입니다. Chronos는 관측과 행동의 **물리 시각**을 time-aware RoPE로 넣어 센서 주기·지연의 변화에 강하게 만듭니다. PaceVLA는 같은 시간 축에서 **명령 간격 자체**를 바꿉니다. PaceVLA의 양자화 모드는 행동 하나가 덮는 실제 시간을 2배로 바꾸는데, 정책은 그 차이를 decoder 번호로만 압니다. Chronos식으로 행동마다 실행 시각을 넣으면 decoder 여러 개 대신 연속적인 실행 간격을 표현할 수 있을 것입니다(내 해석). |
| **AW-0** (`agentic_wam/`) | AW-0은 코드·subgoal 계획 스트림을 행동과 함께 생성합니다. PaceVLA의 judge가 보는 것도 "지금 어떤 국면인가(도달, 쥐기, 운반, 놓기)"입니다. 계획 스트림이 이미 있는 정책이라면 별도의 LVLM 라벨 없이 국면 정보를 judge 입력으로 쓸 수 있습니다(내 해석). |
| **Hydra-0** (`hydra0/`) | 행동 표현을 바꾸는 연구입니다. Hydra-0은 공간 표현(이미지 평면의 점 궤적)을 바꾸고, PaceVLA는 시간 해상도를 바꿉니다. |
| **DexPolicy** (`dexpolicy/`) | 둘 다 "정밀한 국면과 여유 있는 국면은 다르게 다뤄야 한다"는 문제의식입니다. 다만 DexPolicy는 RL 학습 중의 탐색 잡음을 학습 step 함수로 줄이고, PaceVLA는 추론 시 실행 해상도를 상태 함수로 바꿉니다. |
| **Contact-Grounded Policy** (`contact_grounded_policy/`) | PaceVLA의 실패 분석(부록 D.3)은 "접촉 영역이 가려져서"라고 결론짓고 촉각 입력을 제안합니다. CGP처럼 촉각을 쓰는 정책에서는 접촉 시작 신호를 judge 입력으로 직접 쓸 수 있습니다. |

## 4. 관련 GitHub 저장소 (2026-10-29 `gh api` 확인)

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | ★ | 마지막 push | 확인 커밋 |
|---|---|---|---|---|---|---|
| PaceVLA/pacevla.github.io | 프로젝트 웹페이지(영상, 결과 표 이미지). **코드 없음** | 공식 | 표기 없음(페이지 하단에 CC BY-SA 4.0 문구) | 0 | 2026-10-28 | `86424ec` |
| iclr-pacevla/iclr-pacevla | 익명 심사용 웹페이지 사본(2026-09-24 생성). 현재 페이지와 저자 표기만 다름 | 공식으로 보임 | 표기 없음 | 0 | 2026-09-26 | `4ef81f5` |
| NVIDIA/Isaac-GR00T | 기반 VLA GR00T N1.5 | 구성 요소 공식 | Apache-2.0 | 8,280 | 2026-10-28 | `c56f541` (`examples/robocasa/README.md`) |
| RLWRLD/RLDX-1 | 기반 VLA RLDX-1 | 구성 요소 공식 | Apache-2.0 | 355 | 2026-09-09 | `cf67c31` (메타데이터만) |
| mit-han-lab/vlash | 고정 양자화 기준선 [30] | 공식으로 보임 | Apache-2.0 | 514 | 2026-04-22 | `22cbabf` (`vlash/run.py`) |
| lingxiao-guo/DemoSpeedup | 기준선 DemoSpeedup | 공식으로 보임 | MIT | 30 | 2025-09-10 | `34bd43a` (메타데이터만) |
| nadunRanawaka1/SAIL | 기준선 SAIL, TPR 정의 | 공식(프로젝트 페이지 링크) | MIT | 11 | 2026-07-19 | `1440c16` (`metric_fn.py`) |
| Timsty1/MixtureOfHorizons | 가장 가까운 관련 연구(ICML 2026) | 공식으로 보임 | 표기 없음 | 68 | 2026-05-04 | `5da3500` (README만) |
| Physical-Intelligence/real-time-chunking-kinetix | RTC | 공식 | MIT | 631 | 2025-12-08 | `9296f31` (메타데이터만) |
| DaivdYuan/SpeedTuning | RL 가속 관련 연구 | 공식으로 보임 | MIT | 17 | 2026-08-11 | `b4a4cfc` (메타데이터만) |
| RoyZry98/MoLe-VLA-Pytorch | MoLe-VLA | 확인 못 함 | MIT | 70 | 2025-07-31 | `8ae45f8` (메타데이터만) |
| robocasa/robocasa | 벤치마크 | 공식 | NOASSERTION | 1,848 | 2026-10-28 | `dc73a8c` (메타데이터만) |
| Lifelong-Robot-Learning/LIBERO | 벤치마크 | 공식 | MIT | 2,488 | 2025-03-15 | `8f1084e` (메타데이터만) |
| brave-eai/dexjoco | 벤치마크 DexJoCo | 공식으로 보임(이름·시기 일치) | MIT | 239 | 2026-10-15 | `8d23b0f` (메타데이터만) |

- `gh search repos PaceVLA` 결과는 위 두 웹페이지 저장소뿐이었습니다.
- 받은 파일은 `refs/`에 있습니다(gitignore). `sail_metric_fn.py`, `gr00t_robocasa_README.md`, `vlash_README.md`, `vlash_run.py`입니다.
- GR00T 저장소의 RoboCasa 평가 예시는 `--max-episode-steps 720 --n-action-steps 8`입니다. 논문은 t_max = 1500을 씁니다(`03` §3).

## 5. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **AWE: Automatic Waypoint Extraction** (Shi, Sharma, Zhao, Finn, 2023) | 시연을 "선형 보간으로 복원 가능한 최소 waypoint"로 압축합니다. 시간 해상도를 구간마다 다르게 두는 비균일 양자화의 데이터 쪽 버전입니다. PaceVLA의 LVLM 라벨 대신 재구성 오차 기준으로 "합쳐도 되는 구간"을 정하는 대안이 됩니다. | 내 사전 지식. 서지 확인 필요 |
| **Options / macro-action** (Sutton et al., 1999) | 여러 step을 하나의 결정으로 묶는 시간 추상화의 고전 틀입니다. PaceVLA의 모드는 길이가 정해진 option입니다. | 내 사전 지식 |
| **사건 기반 제어·적응 샘플링** (제어 이론) | 오차가 커질 때만 명령을 갱신하는 고전 기법입니다. judge 대신 추적 오차나 접촉 사건으로 세밀 모드를 켜는 규칙 기반 대조군이 될 수 있습니다. | 내 해석 |
| **Bidirectional Decoding (BID)** | chunk를 샘플링할 때 이전 chunk와의 일관성과 미래 예측을 함께 봐서 chunk 길이의 trade-off를 다룹니다. | 내 사전 지식. 서지 확인 필요 |

**한계:** 5절의 서지 정보는 내 사전 지식에 근거합니다. 블로그나 git에 올리기 전에 원문을 확인해야 합니다.

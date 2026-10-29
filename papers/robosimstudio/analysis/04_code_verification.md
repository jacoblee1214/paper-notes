# 04. 코드 검증: 논문 ↔ 구현 대조, 수치 재계산, 구현 가능성

- **대상:** `robosimstudio/robosimstudio` @ `81f57c9` (2026-10-28, HEAD, shallow clone). 논문 시점 상태는 첫 커밋 `73739ab`(2026-10-14, v2와 같은 날)의 파일 트리로 확인했습니다. 저장소에는 커밋이 12개뿐이고, 첫 커밋 하나에 대부분이 들어 있습니다.
- **범위:** 코드·문서·테스트 읽기, PyPI 메타데이터 조회(macOS arm64 wheel 여부), CPU 재계산만 했습니다. 설치·실행은 하지 않았습니다.

## 1. 모듈 매핑

| 논문 | 구현 위치 | 일치 여부 |
|---|---|---|
| 물체는 "무엇인지", 장면은 "어느 solver인지" 선언 (Sec. 3) | `robosimstudio/specs/physics.py`(ArrangementSpec: 영역별 backend, coupler, 충돌 정책), `specs/arrangement.py` | ✅ 데이터로 정의된 배치(arrangement)를 이름으로 고릅니다 |
| 기본 solver: 관절 MuJoCo-Warp, 천·케이블 VBD, 연성체 외부 엔진 | `specs/arrangement.py: GENERIC` — articulated `mujoco`, soft `superdex`(substeps 2), 입자 재료 `mpm`, 그 밖 `vbd`, coupler `proxy` | ✅ 논문 이후 MPM(알갱이·액체·반죽)이 추가됐습니다 |
| 외부 엔진과 자세·접촉 wrench 교환 | `engines/cosim.py`(CosimRig: 자세 읽기 → 외부 엔진 substep → 반력 수확 → Newton에 기록), `engines/feedback.py`(WrenchAccumulator) | ✅ |
| 인스펙터: 단위·범위, 실행 중 변경 vs 재빌드 표시 | `specs/physics.py`의 ParamSchema("단위, 범위, 변경 비용 tier"), `robosimstudio_studio/inspector.py`, `tuning/` | ✅ |
| 에이전트 반복 인터페이스 | `robosimstudio_real2sim_agent/`: 저장할 때마다 `scene.py`를 다시 빌드하는 live 페이지 + Unix socket CLI(`./ctl`), `compare.py`(사진 대비 크기·위치 수치), `checks.py`(take ↔ 영상 timeline, gate: 빌드·유한성·카메라·가라앉음·표류), `frames.py`(영상에서 시각이 찍힌 정지 화면) | ⚠️ D5 |
| 힘 피드백 원격 조종: 속도 의존 deadband, 멈춤·지연 시 해제, 측정 열림 사용 | `robots/teleop/force_feedback.py`(`deadband`, `velocity_deadband`, stale 샘플이면 disarm, "measured jaw" 사용), 리더별 100 Hz 스레드 | ✅ 논문 서술과 일치합니다. 리더 하드웨어는 D7 |
| 로봇: Panda, UR5e, 단팔·양팔 YAM | `robots/embodiments/`: panda, yam_bimanual, yam_i2rt 등 | ⚠️ D6 (UR5e 없음) |
| IK PyRoKi, 계획 cuRobo | `pyproject.toml`: pyroki를 `95afccc`에 고정, cuRobo는 `curobo` extra(cu12) | ✅ |
| privileged 상태와 관측 분리 | `robosimstudio_policy_agent/sim_tools.py`("in-process 모드에서만" 등록), `examples/07_policies/02_without_privilege.py` | ✅ 구조적으로 분리되어 있습니다 |
| GaP 정책 개선 | `robosimstudio_policy_agent/`(GaP 런타임 facade, 도구), GaP 런타임 `KeplerC/graph-as-policy@09e45f8` | ⚠️ D2 (닦기 그래프 없음) |
| 스펀지 닦기 twin, Preston 마모 | `examples/08_demos/wipe_the_bar/`(`bar.py`, `routine.py`), `engines/superdex/bridge.py`(접촉 patch별 압력) | ⚠️ D3, D4 |
| 18개 과제 (Fig. 5) | 공개 저장소에는 데모 3개(wipe, cable, towel)와 재료별 채점 예제 11개 | ❌ D1 |

## 2. 불일치와 논문에 적히지 않은 구현 요소

| # | 내용 | 영향 |
|---|---|---|
| D1 | **Fig. 5의 18개 과제 정의가 공개 저장소에 없습니다.** 과제는 `tasks/` 디렉터리("root")에서 읽는데, 저장소에 그 디렉터리가 없습니다. 테스트는 `vabar/assets/vab_tool_sort`를 "컬렉션 없이 설치하면 없음"이라고 부릅니다(`tests/test_panda_vendored.py`). 공개된 것은 양팔 YAM 데모 3개(`08_demos`)와 재료별 채점 예제(`06_tasks`)입니다. 프로젝트 페이지도 "More Real2Sim Examples Coming Soon"이라고 씁니다. | 18개 과제를 다시 만들 수 없습니다. Sec. 4.1의 결과는 확인할 수 없습니다. |
| D2 | **GaP 닦기 정책, 30개 배치, 강성 변형 스윕, 실제 로봇 실행 코드가 없습니다.** 공개된 policy 예제는 pick-and-place 그래프뿐입니다. | Fig. 7e–h를 재현할 수 없습니다. |
| D3 | **공개 twin은 스펀지를 손에 고정(latch)합니다.** 잡기가 안정되면 패드 근처 폼 노드 106개를 손에 고정하고 **Open**에서 풉니다. 마찰만으로는 세 take에서 폼이 69·226·218 mm 미끄러졌고, 닦인 비율은 0.761·0.724·0.687였습니다. 30회 더 바꿔 시도했지만(마찰, 그립 12 N, 누름 1.5–4 N, 기울기, 속도, 볼록 분해) 미끄럼을 10 mm 아래로 줄인 경우는 없었습니다(README "Why a latch"). 막대도 블록에 용접(weld)되어 있습니다. | 논문은 latch를 언급하지 않습니다. 같은 저장소의 에이전트 작업 규칙(`examples/live/AGENTS.md`, 2026-10-22 추가)은 "latch, weld, attachment 금지, 손가락 접촉으로만 움직일 것"을 완성 조건으로 둡니다. 논문 시점 데모가 이 규칙을 지키지 않습니다. |
| D4 | **공개 twin의 스펀지는 영상 속 스펀지보다 4배 단단합니다.** `bar.py`는 "영상의 부드러운 부엌 폼(20 kPa)은 8 N 집게에서 패드 사이로 흘러나가 들리지 않아" 80 kPa 폼으로 바꿨다고 적었습니다. | 영상과 "맞춘" twin이 물성에서는 영상과 다릅니다. Fig. 7e의 6·20 kPa 변형은 이 집게 조건에서 잡을 수 없는 범위입니다(`03` §3). |
| D5 | **에이전트용 "시공간 피드백"의 실체는 시각 비교 수치와 gate입니다.** `compare.py`는 사진 대비 크기 비율과 픽셀 위치 차이를, `checks.py`는 take의 물체 이동과 영상 timeline 대비, 빌드·유한성·가라앉음·표류 검사를 줍니다. 논문이 나열한 "접촉 사건, 잡기·미끄러짐" 텍스트 피드백을 만드는 전용 모듈은 찾지 못했습니다. | 에이전트가 접촉 문제를 풀려면 `./ctl eval`로 시뮬레이터 상태를 직접 읽어야 합니다(Python 실행). 기능은 있지만 논문 서술보다 원시적입니다. |
| D6 | **UR5e가 공개 저장소에 없습니다.** 레지스트리는 "콘텐츠로 배포되는 로봇은 벤치마크 계층을 import해야 등록된다"고 안내합니다. 공개된 것은 Panda, YAM 계열, Robotiq 2F-85 그리퍼입니다. | Fig. 5에서 UR 계열 팔이 보이는 과제는 공개 자산으로 만들 수 없습니다. |
| D7 | **힘 피드백 리더 암은 GELLO가 아니라 YAM 리더입니다.** Feetech STS3215 서보, CH340 USB 어댑터, `lerobot` 버스 드라이버, udev 규칙과 보정 파일을 씁니다. 공개 예제의 힘 피드백 장면은 패널과 상자이고, 공구 걸기(Fig. 6)가 아닙니다. | 논문의 GELLO 인용은 개념 출처로 읽어야 합니다. 하드웨어 목록은 저장소 README가 정확합니다. |
| D8 | **CPU 실행은 논문 이후에 추가됐습니다**(2026-10-18, 커밋 `a2fe723`). CPU에서는 MuJoCo C가 팔·강체를, SuperDex가 천·가방·로프·연성체를 맡습니다. 종이·테이프·붓·MPM·물은 거부됩니다. 로프는 단방향(케이블이 닿은 것을 밀지 않음)이고, SDF·hydroelastic 접촉과 다중 world도 쓸 수 없습니다(`examples/install.md`). | 논문의 VBD 기반 장면(케이블, 필름, 수건, 열전사)은 CPU에서 다른 물리로 대체되거나 거부됩니다. |
| D9 | **렌더러가 플랫폼에 따라 다릅니다.** 경로 추적 렌더러 OVRTX(1.8 GB)는 RT core가 있는 NVIDIA GPU와 Linux·Windows에서만 씁니다. 그 밖에는 Newton의 tiled 렌더러가 **collider만** 그립니다. | 에이전트가 사진과 렌더를 비교하는 루프(`compare_still`)는 RT GPU가 없으면 질감 없는 collider 그림으로 비교하게 됩니다. A100·H100·B200도 RT core가 없어 같습니다. |
| D10 | 공식 테스트 기록(README "Measured")은 RTX PRO 6000 Blackwell에서 했습니다. 닦기 데모 README는 "superdex extra와 GPU가 필요하다"고 쓰지만, 설치 문서는 SuperDex 스펀지가 CPU에서 돈다고 씁니다. | 문서끼리 다릅니다. 이 맥에서 닦기 twin이 도는지는 실행해 봐야 알 수 있습니다. |
| D11 | 논문 표지(v2)의 소속 번호는 NVIDIA 1, Berkeley 2이고, 프로젝트 페이지는 Berkeley 1, NVIDIA 2입니다. arXiv 제목에는 "for Human and Agent"가 있고 PDF 제목에는 없습니다. | 사소한 표기 차이입니다. |

## 3. 수치 재계산 (`analysis/table_check.py`, `analysis/identifiability_toy.py`)

| # | 대상 (출처) | 재계산 | 판정 |
|---|---|---|---|
| C1 | Table 1 | RSS 10/10 ✓, 다른 프레임워크 ✓ 2–4개. RSS만 ✓인 5행은 모두 이 논문의 워크플로 | 기능 목록에 가까움 |
| C2 | GaP 곡선 0·29·71·100% (Fig. 1, 7f) | k/n 반올림으로 나오는 n은 7, 14, 17, 21, 24 …이고 **30은 아님** | 30개 배치의 성공률이 아님. Fig. 1 y축 이름 "completion rate"는 오해 소지 |
| C3 | 에이전트 구축 시간 (Fig. 7c) | 첫 닦기까지 42분(전체 56분의 75%). 56분은 한 에이전트만 보고됨 | 두 번째 에이전트 시간 없음 |
| C4 | latch 효과 (저장소 README) | latch 0.958 vs 마찰만 0.724, 차이 23.4%p, Welch p = 0.008(n = 3) | 공개 twin의 닦기 성능은 latch가 만듦 |
| C5 | Preston 주석 "4 N, 30 mm/s면 셀 하나에 약 2초" (`bar.py`) | 면 평균 압력 1,067 Pa → 2.34 s. 한 스트로크 마모 = k_p·p·L = 0.96(속도 무관) | ✅ 주석과 측정 첫 스트로크(0.91)와 맞음 |
| C6 | 강성과 잡기 (`bar.py` FOAM 주석) | 80 kPa에서 변형률 9% → 유효 패드 면적 12 cm². 선형 예측 6 kPa 120%, 20 kPa 36% | Fig. 7e 범위의 절반은 잡을 수 없는 강성 |
| C7 | v1 ↔ v2 | 13쪽 → 14쪽, 참고문헌 51 → 58개, 단어열 유사도 0.87. Fig. 1 라벨, 관련 연구(MAESTRO, ASPIRE, ENPIRE, DexAgent, RPG, Push-T), Real2Sim 문단(frontier 모델의 시각 재구성 한계) 추가 | 방법·결과 수치는 그대로 |
| V1 | 영상만으로 E 식별 (toy) | 1080p·0.5 px에서 E > 104 kPa면 상대 SE > 50%. 80 vs 100 kPa는 0.36σ. 힘 관측 SE는 0.4 kPa로 일정 | `03` §2 |

## 4. 구현 가능성 판단

### 4.1 공식 코드를 그대로 쓰는 경우

| 항목 | 요구 사항 | 비고 |
|---|---|---|
| OS / Python | Linux x86-64 + Python 3.12 (개발·테스트 플랫폼). "macOS와 Windows는 테스트하지 않음" | 이 맥에는 Python 3.12.7이 있고 `uv`는 없습니다 |
| GPU | 선택. NVIDIA GPU가 있으면 Warp CUDA, MuJoCo-Warp, VBD, MPM, CUDA graph를 씁니다. 없으면 CPU 경로(D8) | 경로 추적 렌더러는 RT core GPU + Linux·Windows만 |
| 주요 의존성 | newton[sim] 1.7, warp 1.18, mujoco 3.15, viser, coacd 1.0.11, pyroki(git, jax), imageio[ffmpeg]. 선택: superdex(cpu), genesis(+torch), curobo(cu12), gap(torch, cuRobo, transformers), lerobot-feetech | PyPI 조회 결과 warp·coacd·superdex는 macOS arm64 wheel이 있고, newton·mujoco-warp·viser는 순수 Python입니다 |
| 용량 | 저장소 checkout 약 210 MB(로봇 자산 포함). rtx 그룹 1.8 GB(Linux·Windows만) | |
| 코딩 에이전트 | Claude Code나 Codex(유료 구독 또는 API). 영상·사진이 공급자 서버로 갑니다 | 논문의 "약 1시간"은 이 비용과 사람 감독 시간을 포함하지 않습니다 |
| 하드웨어 (선택) | 힘 피드백: Feetech STS3215 YAM 리더 암 + CH340 어댑터. 실제 로봇: 양팔 YAM | |

**판단:** 도구 자체는 **설치해서 쓸 수 있는 수준으로 공개**되어 있습니다(Apache-2.0, 설치 문서, 예제, 테스트). 논문의 결과(18개 과제, GaP 닦기, 실제 로봇)는 핵심 자산이 비공개라 **재현할 수 없습니다**(D1, D2, D6).

### 4.2 이 맥(Apple M4 Pro, CUDA 없음)에서

공식 문서의 CPU 매트릭스(`examples/install.md`)와 Fig. 3의 solver 표시를 합쳐 18개 과제를 추정했습니다. 과제 정의가 비공개라 재료 종류는 이름에서 짐작한 것입니다.

| 과제 (Fig. 5) | 추정 재료 | 이 맥(CPU)에서 |
|---|---|---|
| 공구 정리, 주사기 폐기, 바구니 담기, 공구 걸기, 접시 꽂기, 서랍 열기, 상자 뒤집기 | 강체·관절 | 가능(MuJoCo C). 단 UR5e 과제는 자산 없음 |
| 기어 맞물림, 커넥터 삽입 | 강체 + 정밀 메쉬 접촉 | 어려움: CPU에는 SDF·hydroelastic 접촉이 없음 |
| 케이블 배선, 도관 통과 | 로프(논문은 VBD) | 다른 물리로 가능: SuperDex 로프, 단방향 |
| 수건 접기, 잎 누르기 | 천·쉘(논문은 VBD) | 다른 물리로 가능: SuperDex 천 |
| 필름 벗기기, 박스 테이프 자르기 | 테이프·절단(VBD) | 테이프는 거부, 절단은 느리게 가능(1–2 s/step) |
| 스펀지 닦기 | FEM(SuperDex) | 가능해 보임(macOS arm64 wheel 있음). 데모 README는 GPU 필요라고 씀(D10) |
| 띠 연마, 열전사 | 접촉 + 표면 법칙, 필름(VBD) | 띠 연마는 가능해 보임, 열전사는 불확실 |

- **할 수 있는 것:** `examples/`의 강체·관절 장면, Viser studio, `./ctl` 기반 에이전트 루프(Unix socket은 macOS에서 동작), CPU 데모 기록. 설치에는 승인이 필요하고, 실제 동작은 확인하지 않았습니다.
- **할 수 없는 것:** CUDA 경로 전체(MuJoCo-Warp, VBD, MPM, cuRobo 계획), 경로 추적 렌더러, 다중 world, GaP 런타임(torch + cuRobo, cuRobo가 CUDA 전용이라 불가로 추정). Apple GPU(MPS)는 이 스택이 쓰지 않습니다. 저장소가 고르는 장치는 `cpu`와 `cuda`뿐입니다(`robosimstudio/machine.py`).
- **속도:** 공식 문서 기준 CPU 한 스레드에서 VBD 수건(1,922 입자)은 실시간의 0.003배, 절단은 GPU 7.6 ms 대비 step당 1–2 s입니다. 맥에서는 "보고 만지는" 용도로는 쓸 수 있지만, 데이터 생성이나 정책 개선 루프에는 비현실적입니다.

### 4.3 아이디어만 가져와 재구현하는 경우

| 구성 요소 | 난이도 | 메모 |
|---|---|---|
| "물성·solver를 편집 가능한 항목으로" 장면 기술 | 중간 | 이미 쓰는 시뮬레이터(MuJoCo, Isaac Lab) 위에 인스펙터와 파라미터 schema(단위·범위·재빌드 여부)를 얹는 일 |
| 에이전트 루프(저장하면 다시 빌드, 사진 비교 수치, gate) | 낮음–중간 | `robosimstudio_real2sim_agent`의 구조(live 페이지, socket, compare, gate, log.jsonl)가 짧고 독립적이라 옮기기 쉽습니다 |
| 힘 피드백 원격 조종 | 중간 | 리더 암 하드웨어가 필요합니다. deadband·stale 처리 같은 안전 로직은 `force_feedback.py`에서 가져올 수 있습니다 |
| 다중 물리 co-sim(강체 + FEM) | 높음 | Newton과 SuperDex를 직접 쓰는 것이 현실적입니다. 직접 결합을 짜는 것은 권하지 않습니다 |

**요약:** RSS는 논문이라기보다 **바로 쓸 수 있는 오픈소스 도구**입니다. 다만 논문의 성과를 판단하려면 비공개 자산이 필요합니다. 이 맥에서는 강체 장면과 에이전트 루프를 시험해 볼 수 있지만, 변형체와 정책 개선은 NVIDIA GPU 서버가 필요합니다.

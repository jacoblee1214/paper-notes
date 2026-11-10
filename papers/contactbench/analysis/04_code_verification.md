# ContactBench 1.0: 논문 ↔ 공식 코드 대조 (정적 읽기)

## 0. 대상과 방법

- 저장소: github.com/Michaelwyx/contactbench-sim (MIT). 얕은 클론 `repo/contactbench-sim/`(gitignore). 커밋 `ccd864501275112e369a71c91880adf2da28b536`(2026-11-04 커밋, 2026-11-05 push, 제목 "ContactBench 1.0 simulation").
- **읽은 것**: README, docs/tasks.md(과제 표, "How a run is judged", "Tasks not included"), docs/variations.md(랜덤 시작), docs/data.md(수집과 에피소드 형식), docs/installation.md(요구 사항), docs/tactile.md 일부, `pyproject.toml`과 `packages/force-bench/pyproject.toml`(의존 고정), `qa.py`, `appendix_tasks/common.py`(판정 헬퍼), 과제 모듈 `plug_in_charger.py`, `pick_up_object_on_platform.py`, `screw_nut_onto_screw.py`, `turn_doorknob_open_door_flat.py`, `retrieve_utensil_from_bin.py`(일부), `_allen_key.py`(일부), `hang_wrench.py`, `hang_ruler.py`(시간 예산).
- **하지 않은 것**: 코드 실행, `pip install`, 자산·데이터·가중치 다운로드, GPU 사용. docs가 "`force-bench list`는 GPU가 필요 없는 순수 파이썬"이라고 쓰지만 실행하지 않았습니다.
- 표기: 논문 쪽은 절·표·그림 번호, 코드 쪽은 파일명과 줄 번호 또는 docs 절로 씁니다.

## 1. 과제 정의 대조

**논문 36개 ↔ 코드 범위**(docs/tasks.md)

- 기본 과제 31개 중 **22개에 스크립트 정책과 자동 판정**이 있고(73행), 과제 2(수저 통)는 **장면만**(판정 없음) 있습니다.
- 제외된 기본 과제 8개: 1(열쇠고리에서 열쇠 하나), 3(나사 통에서 한 개), 7·8(여러 물체 모으기), 10(통에 넣기), 11·12(손안 재배치), 21(열쇠 꽂기). 제외 이유는 docs에 과제별로 적혀 있습니다(예: 11·12는 "패드 마찰과 순응이 보정되지 않았다").
- 장기 과제 5개(부록 C, Table C2)는 **구현되지 않았습니다**. 몇몇 단계는 기본 과제로 존재합니다.

**정책 평가 6과제 ↔ 코드 행**(Table II 기준)

| 논문 과제 | 코드 행(예) | 자동 판정 | 비고 |
|---|---|---|---|
| Thread nut onto screw | `screw_nut_onto_screw`, `thread_nut_onto_screw_from_table` | 있음 | |
| Tighten screw with Allen key | `tighten_screw_with_allen_key_from_table` | 있음 | 래퍼 31줄, 본체는 `_allen_key.py` |
| Plug in charger | `plug_in_charger`, `plug_in_charger_block1`…`4`, `…_spring_0.85mm` | 있음 | |
| Turn doorknob to open door | `turn_doorknob_open_door_flat`, `_unlock40`, `_unlock80`, `_knob1_55mm`(직경 55 mm) | 있음 | 회전 40–80°, 스프링 5/7.5/10 mm 변형 |
| Retrieve single utensil from bin | `retrieve_utensil_from_bin` | **없음(장면만)** | docs/tasks.md 과제 2: 손가락 공간 때문에 스크립트 정책 불가 |
| Pick up thin card | `pick_up_object_on_platform` (카드 행) | 있음 | |

→ 논문의 6과제 정책 평가 중 **5개만 시뮬에서 채점할 수 있습니다**(D2).

## 2. 불일치와 일치 항목 (D-items)

판정 열: **불일치**(논문과 코드가 다름), **추가**(코드에만 있음), **일치**, **확인 불가**(코드에 근거 없음, 원자료 미공개).

| ID | 주제 | 논문 | 코드 | 판정 | 근거 |
|---|---|---|---|---|---|
| D1 | 과제 범위 | 36개(31 + 5) | 기본 22개 채점 + 과제 2 장면, 장기 과제 0개 | 불일치 | docs/tasks.md, README |
| D2 | 정책 평가 과제 | 6개 중 실물 시연·평가 | 6개 중 5개 채점 가능. 수저 통은 장면만 | 불일치 | docs/tasks.md 과제 2 |
| D3 | 로봇 | 실물 UR5e + WSG50, 손가락 3D 프린트 + GelSight Mini 또는 Fin-Ray | 시뮬 Franka FR3 + Franka Hand, Fin-Ray 어댑터(FR3용), 시뮬 GelSight | 불일치(시뮬 대 실물) | docs/installation.md, docs/finray.md, Sec. V-A |
| D4 | 성공 기준 | 텍스트(예: 문 5° 이상, 충전기 완전 삽입) | 텍스트 기준 + **논문에 없는 수치 기준** (아래 표) | 추가 | 과제 모듈, `common.py` |
| D5 | 힘 지표 | >40 N 비율(손목 힘 최댓값), 측정 위치·방법 미기재 | 손목 힘 40 N 판정 없음. 40 N은 **원격조작 가드의 위험 임계값**(robot→object, 경고 15 N). 접촉 해제는 0.05 N 미만 | 불일치(정의 불명) | docs/teleoperation.md 표, `common.py` `CONTACT_FREE_N = 0.05` |
| D6 | 물체 초기 조건 | 0.15 m × 0.15 m 균일, 고정 시드 | `--seed N`으로 같은 사각형 안 균일 추출, 물체 겹침 회피 | 일치 | docs/variations.md, `common.py` `sample_start` |
| D7 | 로봇 초기 자세 | 시연 UMI 초기 자세에서 가우시안 추출 | 로봇 시작 자세 랜덤화 **없음**(`appendix_tasks`에서 로봇 시작 랜덤 코드를 찾지 못함). 시드는 물체만 | 불일치 | grep 결과, docs/variations.md("물체 위치 및 yaw") |
| D8 | 정책 학습과 평가 | DP, RDP, π0.5, FTP-1을 과제당 학습·30회 평가 | "Policy learning and evaluation pipelines are not part of this release." 스크립트 정책 수집과 원격조작 기록만 있음 | **불일치(핵심)** | README 37행, docs/data.md |
| D9 | 롤아웃 시간 | 1분(실물) | 과제마다 다름: `hang_wrench` 40 s, `hang_ruler` 50 s, 나사돌리개 120 s, Allen key 140 s, 렌치 150 s | 불일치(목적이 다를 수 있음) | 과제 모듈의 `T_BUDGET` |
| D10 | 판정 | 과제 성공 기준 적용, 판정 주체 미기재 | 자동 판정(`appendix_qa.json`), 기준 항목별 pass, 물리 무결성 검사 포함 | 추가(시뮬은 자동) | docs/tasks.md "How a run is judged" |
| D11 | 결정성 | 미기재 | GPU 접촉 순서가 결정적이지 않음. 같은 시작에서도 힘이 조금 다름. 판정은 마진으로 흡수 | 추가(재현 한계) | docs/data.md, docs/installation.md, docs/tasks.md |
| D12 | 자산 힘 캘리브레이션 | Fig. 5: 인쇄본 대 실물 F/T 힘, 10회 반복 | `kit/mechanisms/outlet.py`: "28 kN/m fitted to measured insertion force (UR arm, wrist F/T, 10 trials)". docs/assets.md에 측정 힘 프로파일과 그것을 재현하는 파라미터가 있다고 적혀 있음(수치는 읽지 않음) | 일치(상수), **원자료 확인 불가** | 소스 주석, docs/assets.md |
| D13 | 시뮬 엔진 | Newton + MuJoCo Warp(버전 없음) | `newton==1.5.0`, `warp-lang==1.16.0`, `mujoco-warp==3.11.0`, `mujoco==3.11.0`으로 고정 | 일치(버전 추가) | `packages/force-bench/pyproject.toml` |
| D14 | 촉각 시뮬 | hydroelastic 압력장 + Taxim 렌더링, 선택 모듈 | `hydrogel` 패키지(압력, 렌더, 센서 모듈), GS Mini 보정표는 HydroShear(MIT) 파생. `TB_GELSIGHT=1` 표시 과제만 지원 | 일치(범위는 과제 일부) | THIRD_PARTY_NOTICES, docs/tactile.md |
| D15 | 자산 수 | 40개 이상 | 자산 페이지 기준 49개 키트 자산 | 일치 | README |
| D16 | 사람 시간 | 사람 시연 시간(Table III) | 사람 시연 없음. 스크립트 정책은 "기준 시연이지 기준선이 아님" | 확인 불가(시뮬에 대응 없음) | docs/tasks.md |
| D17 | 학습 설정 | 부록 B(DP 1×RTX 5090 등) | 학습 설정 없음 | 불일치(없음) | 저장소 전체 |
| D18 | 라이선스 | arXiv 비독점(논문), 자산 공개 예정 | 코드·자산 MIT, GS Mini 외장 CAD 제외(GPL-3.0) | 추가 | LICENSE, THIRD_PARTY_NOTICES |
| D19 | 공개 일정 | "will be open-sourced" | 2026-11-05 push, 별 9개 | 일치 | GitHub API |
| D20 | 과제 목록 | 부록 C의 과제 이름 | `docs/tasks.md`의 과제 표가 같은 이름과 번호를 씀 | 일치 | docs/tasks.md |

**D4 세부: 논문 텍스트에 없는 수치 기준(코드)**

| 과제 | 코드의 추가 검사 (임계값) |
|---|---|
| 문손잡이 | 잠금 해제 전 문 열림 < 1°, 마지막 시점 문 열림 ≥ 5°, 마지막 1초 손목 이동 < 3 mm, 마지막 1초 문 각도 범위 < 1° (`turn_doorknob_open_door_flat.py`) |
| 충전기 | 밀기 시험에서 콘센트 반력 ≥ 8 N(`PUSH_F`), 추가 밀기 거리 ≤ 0.3 mm(`PUSH_MOVE_MAX`), 꽂은 뒤 정착 간격 ≤ 0.5 mm(`SEAT_GAP_MAX`), 놓은 뒤 정지 ≤ 0.5 mm(`STAY_MAX`) (`plug_in_charger.py`) |
| 카드 집기 | 들어 올림 ≥ 30 mm(`LIFT_MIN`), 1.5 s 유지(`HOLD_S`) 동안 양쪽 패드 힘 > 0.5 N인 프레임 ≥ 95%, 미끄럼 < 2 mm (`pick_up_object_on_platform.py`) |
| 너트 나사 | 최종 너트 중심 높이 `ENGAGED_Z`, 7회 스트로크 동안 평균 패드 힘 > 1 N, 마지막 z 변화 < 0.3 mm (`screw_nut_onto_screw.py`) |
| 모든 "접촉 해제" | 로봇-물체 접촉 합 < 0.05 N(`CONTACT_FREE_N`) |

→ 논문 텍스트는 "stably grasped and lifted off the platform"처럼 질적으로 씁니다. 코드는 그것을 수치 규칙으로 바꿉니다. **이 임계값들은 논문에 없습니다.** 다른 연구자가 정책을 채점할 때 같은 결과를 내려면 이 값을 공유해야 합니다. 임계값을 바꾸면 Table III 같은 성공률은 달라질 수 있습니다. 정책 평가 재현에서 가장 중요한 설정 항목입니다.

## 3. "접촉이 많다"는 정의의 코드 쪽 대응

- 코드는 접촉을 **존재와 해제**로 채점합니다. 해제는 0.05 N 미만, 패드 힘은 1 N 이상이나 0.5 N 이상, 반력은 8 N 이상입니다.
- 접촉의 양이나 "얼마나 접촉에 의존하는가"를 재는 지표는 없습니다. 과제 목록의 기술 라벨(예: `extrinsic`, `insertion`)은 문서용 표시이고 채점 로직과 연결되지 않습니다.
- 따라서 "접촉이 많은 과제"의 조작적 정의는 논문과 코드 모두에 없습니다. 이 점은 `01` §8과 같습니다.

## 4. 구현 가능성 판정 (이 맥)

**이 맥의 조건**: Apple M4 Pro, 통합 메모리 약 24 GiB, MPS, CUDA 없음, torch 미설치(메모리 기록). 설치는 하지 않았습니다.

**시뮬레이터 요구 (docs/installation.md)**

- Linux x86_64(Ubuntu 22.04에서 시험), **NVIDIA GPU**와 CUDA 12 드라이버(시험 장치 RTX 5090 32 GB), Python 3.10 이상(시험 3.12).
- 과제 프로세스당 GPU 메모리 1–2 GB(GelSight는 더 많음). Warp 커널 첫 컴파일에 수 분 걸립니다.
- 과제 실행 시간: 삽입·문·다이얼 2–6분, 손잡이 4–12분, 컵 6–20분, 공구 집기·사용·복귀는 30–60분(RTX 5090 기준, docs/tasks.md).
- Apple GPU(MPS)는 CUDA가 아니므로 이 저장소의 Warp 경로로는 쓸 수 없습니다(문서가 지원하지 않음). Warp의 CPU 백엔드로 돌릴 가능성은 문서에 없고 시도하지 않았습니다(미확인).

**실물 요구 (논문 Sec. V-A, Table B1, B2)**

- UR5e, WSG50, RealSense D435i(손목), GelSight Mini 여러 개(손가락 3D 프린트 부품), Fin-Ray 손가락, HTC VIVE 추적기와 OpenVR, 3D 프린터(PLA, 논문은 Bambu Lab P1S와 Prusa XL), M3 나사, 24×24 in 벨크로 보드, 렌더 오버레이용 PyBullet.
- 시연: 과제당 100회 × 6과제 = 약 600회(UMI 수집). 학습: DP와 RDP 각 RTX 5090 1장, π0.5 GH200 2–4장, FTP-1 GH200 2장(Table B1).

**판정**

| 항목 | 판정 | 이유 |
|---|---|---|
| 공식 시뮬 채점(force-bench) 실행 | **불가** | CUDA GPU와 Linux가 필요. 이 맥에는 둘 다 없음 |
| 정책 평가(Table III) 재현 | **불가** | 실물 장비, GPU 학습, 시연 데이터, 그리고 평가·학습 코드가 공개되지 않음(D8) |
| 실물 과제 재현 | **불가** | 하드웨어 7종 이상, 3D 프린트 자산, 사람 시연 |
| 시뮬 모델 읽기·사양 확인 | 가능 | 문서와 과제 모듈은 읽기만 하면 됨(이 분석에서 수행) |
| `force-bench list` 실행 | 조건부 가능 | docs는 "GPU 불필요, 순수 파이썬"이라고 씀. 설치 승인 후 CPU에서 실행해 행 수 확인 가능(미실행) |
| 아이디어 수준 재구현(DP 소형 학습) | 조건부 | DP 공개 구현(MIT, 2024)은 MPS에서 돌 수 있음. 그러나 학습 데이터가 필요하고, 데이터는 시뮬(GPU 필요)이나 실물(없음)에서만 나옴 |

**노력 추정(추정이며 검증하지 않음)**: 시뮬 채점 재현은 Linux GPU 장비에서 한 주 정도(설치와 Warp 컴파일 포함). 정책 평가 재현은 시연 수집과 학습을 포함해 수 개월과 GPU 여러 장이 필요합니다. 실물 재현은 하드웨어 조달과 자산 인쇄까지 포함하면 더 깁니다.

**결론(한 줄)**: 이 맥에서 ContactBench 1.0을 재현하는 것은 **불가능**합니다. 할 수 있는 것은 표 재계산, 통계 점검, toy, 과제 사양 읽기입니다. 정책 수준의 주장을 검증하려면 Linux GPU 장비와 실물, 그리고 저자의 평가 코드(D8)가 필요합니다.

## 5. 권리와 라이선스

| 대상 | 근거 | 판단 |
|---|---|---|
| 논문 PDF와 그림 | arXiv 초록 페이지의 라이선스 링크: `nonexclusive-distrib/1.0`. PDF 본문 전체(pdftotext)에서 "©", "copyright", "license" 문구를 찾지 못함 | **CC BY 아님.** 그림·표 이미지는 개인용 리포트와 PPT에만 넣음. 게시본에는 넣지 않음 |
| 프로젝트 페이지 | HTML에 라이선스 문구 없음. 숨은 HTML 주석 안의 `[TODO]` 표시는 화면에 보이지 않음 | 라이선스 판단 근거 없음. 페이지 내용 인용은 짧게 |
| 수치(Table III) | 사실 자료. 재계산은 저작권 보호 대상이 아님 | 게시본에 재계산값과 출처를 씀 |
| 시뮬 코드와 자산 | MIT(LICENSE). THIRD_PARTY_NOTICES: HydroShear 보정 MIT, GelSight GS Mini 외장 CAD 제외(GPL-3.0) | MIT 조건(저작권과 허가 문구 유지)을 지키면 사용 가능. 이 분석은 코드를 복사하지 않음 |
| 정책 구현 | DP MIT(2024-12-24), openpi Apache-2.0, FTP-1 Apache-2.0, RDP는 GitHub 자동 판정 NOASSERTION(LICENSE 파일 미확인) | RDP는 라이선스를 확인하기 전까지 재배포하지 않음 |

**저자 정보**: 논문의 이메일 주소와 소속 연락처는 게시본과 요청문에 넣지 않았습니다. 소속은 대학 이름만 씁니다(인용에 필요한 경우).

## 6. 실행 안 한 것과 미확인 항목

- 시뮬레이터, 테스트, 예제를 실행하지 않았습니다. D11(결정성)과 힘 수치는 확인하지 못했습니다.
- 자산 STL과 키트의 `provenance.json`은 읽지 않았습니다.
- `packages/force-bench/tests`의 물리 계약 테스트를 읽지 않았습니다.
- 채택되어 있는 `repo/contactbench-sim` 커밋 이력 전체는 확인하지 않았습니다(얕은 클론). "2026-11-04 단일 커밋"으로 보이지만 확인하지 못했습니다.

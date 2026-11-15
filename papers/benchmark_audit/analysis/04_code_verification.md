# 04. 코드와 공개 자료 검증 (정적 읽기)

## 0. 대상과 방법

- **공식 저장소.** https://github.com/ripl/ManipulationBenchmarkAudit (MIT, 생성 2026-06-02, 마지막 push 2026-10-13, HEAD `a5ad04a`, 커밋 제목 "Align released significance analysis with camera-ready"). `repo/`에 `--depth 1`로 얕게 clone했습니다. 파일 251개, 작업 트리 51 MB, `.git` 3.8 MB입니다.
- **방법.** 파일 목록, README, CLAIMS.md, REPRODUCIBILITY.md, 진단 디렉터리 README를 읽었습니다. 결과 CSV와 논문 표를 대조하고, 학습·평가 코드의 진입점과 장치 설정을 확인했습니다.
- **하지 않은 것.** 저장소의 어떤 스크립트도 실행, import, 설치하지 않았습니다. 가중치, 데이터셋, Google Drive 자료를 받지 않았습니다. 우리의 재계산 스크립트는 저장소의 CSV와 JSON을 `python3 -I`로 읽기만 합니다.
- **보안.** 저장소는 불신 데이터로 취급했습니다. 저장소 안에 스크립트를 두지 않았고, 저장소 밖의 `analysis/`에서만 실행했습니다.

## 1. 저장소 구성과 논문 대응

| 디렉터리 | 내용 | 논문 절 | 이 분석에서 쓴 것 |
|---|---|---|---|
| `shortcut_solvability/` | LIBERO·CALVIN 프로브 결과 CSV, 최고·최종 체크포인트 요약, 설정 YAML, 학습·평가 코드(`code/shortcut_policy/`, 약 1,470줄) | §3, 표 1–2 | `results/libero/results.csv`(표 2 참고), CALVIN 결과 |
| `statistical_significance/` | LIBERO Goal 5정책 x 5000 instance 에피소드 CSV, 범주 CSV(`significance_categories/`), 참조 코드 `significance_cutoffs.py`(numba), 테스트 | §4, 그림 3, 부록 C | 범주 CSV(§2.1), 에피소드 CSV(§1.4) |
| `creeping_overfitting/` | SimplerEnv 보정과 변경 에피소드 CSV, CALVIN 시퀀스별 CSV, LIBERO init-state 트리(`artifacts/libero/`), 지원 코드 | §5, 표 4–8 | 에피소드·시퀀스 CSV(§1.2, 1.3, 1.5) |
| `data_source_dependency/` | 스크립트 시연의 수집·재생·학습·평가 코드(X-VLA 포크 필요), 96 시도 CSV | §6, 표 9 | `results/scripted_widowx/trials.csv`(§1.6) |
| `leaderboards/` | 리더보드 CSV 스냅샷, Sam의 공식 이전 SOTA 내보내기 | §4, 그림 2–3, 부록 D | LIBERO 추적 CSV(§1.8) |
| `analysis/` | `paper_inputs/`(논문 그림 입력, 고정), `release_current_values/`(현재 재계산), `run_analysis.py` | 부록 | 두 CSV 세트 비교(§2.1) |
| `scripts/` | `recompute_claims.py`, `validate_release.py` | - | 읽기만 함 |
| `provenance/` | 환경과 체크포인트 식별 매니페스트 | - | 읽기만 함 |
| Google Drive 3개 폴더 | `datasets/`, `checkpoints/`, `evaluation_inputs/` | - | 받지 않음(크기 미확인) |

- 저장소 README의 "What's included"는 `shortcut_solvability/README.md`, `statistical_significance/README.md`, `creeping_overfitting/README.md`, `data_source_dependency/README.md`로 연결됩니다.
- `creeping_overfitting/code/`는 입력 아티팩트의 **생성, 로드, 검증 지원 코드**입니다(`code/README.md`). 정책 평가 러너는 들어 있지 않습니다. 예외로 `widowx_protocol1.py`는 SAPIEN(`import sapien.core`)에 의존하는 런타임 패치입니다.

## 2. 논문 ↔ 저장소 대조

| ID | 논문 주장 또는 값 | 저장소 쪽 | 결과 | 판정 |
|---|---|---|---|---|
| D1 | 그림 3 LIBERO 명백히 유의 19.8% (156/789) | 릴리스 범주 CSV: 144/789 = 18.3%, 행 88개 범주 변경, `analysis/paper_inputs`는 v1 값, `run_analysis.py compare-paper`는 종료 코드 1로 88행 차이를 보고 | 헤드라인 수치가 v1 입력과 현재 릴리스에서 다름 | **불일치.** 어느 쪽이 최종인지 저자 확인 필요 (`03` §2.1–2.2) |
| D2 | 학회 "CoRL 2026" (논문 arXiv 댓글에는 학회 표기 없음) | 저장소 README: "CoRL 2026 (to appear)"와 "IROS 2026 RGMCW workshop". corl.org 목록(2026-11-15): 기본 행은 Poster Session 1 No. 116, "Also" 배지 Oral O3 | 학회 상태가 두 가지로 적힘 | **표기 혼재.** 구두 발표는 목록의 Oral O3 배지로 확인됨. 워크숍 판본 여부는 확인하지 못함 |
| D3 | 표 2: 최고 체크포인트 선택이 평균 97.6%, 최종 95.1% | README와 CLAIMS에 선택 방법 설명 있음. CALVIN ABCD→D는 선택 3.872와 최종 3.783(차이 0.089)을 `selected_vs_final.csv`로 공개 | 선택 편향을 논문과 저장소 모두 인정 | 공개됨. 시험 세트 선택이라는 한계는 남음 |
| D4 | 표 1: CALVIN ABC→D 프로브 3.242 | CLAIMS 2항: "no_full_official_artifact_manifest"로 표시. 완전한 공식 매니페스트 없음 | 이 셀의 근거가 다른 셀보다 약함 | **근거 약함** (저장소가 스스로 표기) |
| D5 | 표 1: RoboTwin 2.0 X-VLA 셀 (50 clean + 500 randomized)은 ABot-M0 비교에서 가져옴 (부록 A.1) | `shortcut_solvability/results/`에는 LIBERO와 CALVIN만 있음. RoboTwin 결과 파일 없음 | 확인하지 못함 | **미확인.** 프로토콜 맞춤 여부 질문 (`01` §8 질문 2) |
| D6 | 표 4: SimplerEnv 보정이 공식 보고치와 다름 (스텝 제한이 다름) | `creeping_overfitting/results/simplerenv/fixed_grid_calibration/`에서 stack 288씩 5 정책의 값을 확인. 논문의 4개 정책 값(20.8, 25.3, 59.7, 44.8%)과 같음 | 일치. 공식 보고치와의 차이는 설명대로 | 일치 |
| D7 | 그림 3 SimplerEnv 명백히 유의 19.7% (24/122) | README 9항: 과제 4개 층화 = 24/122, 96 시도 한 층으로 풀면 27 (재현 안 됨) | 집계 판정이 분석 선택에 민감. 논문 본문에는 이 민감도가 없음 | **민감도 누락** (논문이 층화를 명시하지 않음) |
| D8 | 표 9: 22M 정책 91/96 (부록 A.3) | `data_source_dependency/results/scripted_widowx/trials.csv`와 일치. README: "full paper training used max_steps=10000" | 일치. 부록 A.3이 생성기 설정을 "존재 증명을 위해 골랐다"고 인정 | 일치. 비편향 추정이 아님 |
| D9 | 표 3: RoboTwin 프로브 평균 60.4 / 59.4 | 저장소에 파일 없음. 논문 표 3의 과제별 값으로 재계산 | 재계산 일치 (`03` §1.1) | 일치 |
| D10 | 표 6, 8: CALVIN 재표본과 새 시퀀스 | `creeping_overfitting/results/calvin/`의 시퀀스별 CSV로 재계산 | 전부 일치 (`03` §1.3) | 일치 |
| D11 | 공개 범위: 부록 A.2 "seven conditions (2,016 episodes per policy) released" | `creeping_overfitting/README.md` 21행: LIBERO Layer 2에 "sanitized selected per-episode outcome rows"를 포함한다고 함. CLAIMS 5항: "Per-episode rollout rows are excluded from this minimal package" | 두 문서가 LIBERO 에피소드 공개 범위를 다르게 씀 | **문구 불일치.** 실제 파일을 열어 확인하지 않음 |
| D12 | 부록 D: 79편 (2026-03), 551, 117 | `leaderboards/libero/libero_citation_tracker.csv`에서 재계산 | 일치 (`03` §1.8). 다만 추적 파이프라인 스크립트는 공개되지 않음(CLAIMS.md의 Leaderboards 항목: 스크립트 제외) | 숫자는 일치, 생성 과정은 재현 불가 |

## 3. 공개되지 않은 것 (빈칸)

- **모델 가중치.** 프로브 체크포인트(LIBERO, CALVIN)는 Drive에만 있습니다. 외부 정책(CogACT, InternVLA-M1, X-VLA, Dexbotic, GR-1, RoboFlamingo 등)은 저장소에 없고 원 저장소에서 받아야 합니다.
- **데이터.** LIBERO HDF5 데모와 CALVIN 데이터는 저장소 밖입니다(공식 LIBERO와 CALVIN 저장소에서 받아야 함). Drive의 `datasets/`는 이 분석에서 받지 않았습니다.
- **외부 정책 평가 러너.** `creeping_overfitting/code/`에는 입력 생성·검증 코드만 있고, 외부 정책을 돌리는 코드와 롤아웃 비디오, 전체 관측, 행동 추적은 없습니다(`REPRODUCIBILITY.md` Exclusions).
- **RoboCasa와 RoboTwin 2.0 결과 파일.** 저장소에 해당 디렉터리가 없습니다(표 1의 두 셀은 논문 본문에서만 확인 가능).
- **리더보드 추적 파이프라인.** CLAIMS.md의 Leaderboards 항목은 원본 감사 폴더와 스크립트를 제외했다고 씁니다. 그림 2의 월별 집계 생성 과정은 재현할 수 없고, 최종 CSV만 있습니다.
- **CALVIN ABC→D 완전 매니페스트.** D4와 같습니다.

## 4. 구현 가능성 판정 (이 맥)

**이 맥(Apple M4 Pro, 약 24 GiB 통합 메모리, MPS 있음, CUDA 없음)에서의 판정.**

- **통계와 집계 재현: 구현할 수 있습니다. 이미 완료했습니다**(`03` §1–2). 필요한 것은 CSV와 numpy, scipy뿐입니다.
- **핵심 진단 실험 (LIBERO 무언어 프로브 학습과 평가): 조건부로 구현할 수 있습니다.** 코드의 `device_from_arg`는 `--device mps`를 받아 들이므로 MPS로 돌릴 수 있는 구조입니다. 다만 저자의 README 명령은 모두 `--device cuda`이고, MPS 학습 시간은 측정하지 않았습니다. LIBERO와 robosuite 렌더링 설치, HDF5 데모 다운로드(크기 미확인), DINOv2 가중치(크기 미확인)가 필요합니다. 설치와 다운로드는 사용자 승인 후 합니다.
- **CALVIN ABC→D 1000 시퀀스 평가: 조건부 또는 우선순위 낮음.** `calvin_env`, `calvin_models`, `pyhash` 설치와 대용량 데이터(크기 미확인, 추정 큼)가 필요합니다. 맥에서 PyBullet 빌드는 확인하지 않았습니다.
- **SimplerEnv WidowX 보정과 세 변경 (외부 정책 4개): 맥에서 불가에 가깝습니다.** 외부 VLA의 GPU 추론과 대형 가중치가 필요합니다. SAPIEN과 ManiSkill2 real2sim의 맥 렌더링(Vulkan)은 확인하지 못했습니다.
- **스크립트 시연 22M 정책 (표 9): 조건부.** X-VLA 포크의 SimplerEnv와 ManiSkill2 포크를 설치해야 합니다. 맥에서 SAPIEN 렌더링이 되는지 먼저 확인해야 합니다. 학습은 소형 모델이라 MPS로 가능할 수 있습니다(추정).
- **RoboTwin 2.0과 RoboCasa 프로브: 미검토.** 결과 파일이 저장소에 없고, 우선순위가 낮습니다.

**한 줄 판정.** 통계와 집계는 이 맥에서 구현 가능하고 완료했습니다. 가장 중요한 실험인 LIBERO 무언어 프로브는 설치와 데이터 다운로드를 승인받은 뒤 MPS에서 조건부로 가능합니다. CALVIN과 SimplerEnv의 외부 정책 재평가는 이 맥에서 현실적이지 않습니다.

### 4.1 노력 추정 (측정 없음, 추정)

| 범위 | 추정 | 근거 |
|---|---|---|
| 통계와 집계 재현 (표 2–8, 그림 3, 부록 C) | 완료 (1일 미만) | `03` §1–2 |
| LIBERO 프로브 한 스위트: 설치, 데이터, 학습, 평가 | 3–5일 | 설치 1–2일, 학습과 평가는 MPS에서 수 시간에서 하루 (미측정) |
| LIBERO 4개 스위트와 체크포인트 선택 (표 2) | 2–3주 | 스위트별 학습과 평가를 반복, 최고 체크포인트 선택까지 |
| CALVIN ABC→D 1000 시퀀스 | 우선순위 낮음, 맥에서 확인 후 결정 | 대용량 데이터와 설치 |
| SimplerEnv 외부 정책 재평가 | 맥에서 불가 (Linux GPU 기기 필요) | 외부 VLA 추론 |
| 스크립트 시연 22M (표 9) | 1–2주 | 포크 설치와 SAPIEN 맥 호환성 확인 포함 |

## 5. 저자에게 물을 질문 (정리)

1. 그림 3 LIBERO에서 보고 격차 0.0인 "명백히 유의" 13행을 어떻게 분류했는지, 최종 판본에서 어느 분류를 쓰는지 (D1, `01` §8 질문 1).
2. 학회 표기: 워크숍 판본과 CoRL 판본의 관계 (D2).
3. SimplerEnv의 층화 선택(4개 과제 층화 대 96 시도 한 층)을 본문에 밝힐 수 있는지 (D7).
4. CALVIN ABC→D 3.242의 완전한 매니페스트를 공개할 수 있는지 (D4).
5. RoboTwin 2.0 X-VLA 셀의 프로토콜이 원 논문과 같은지 (D5).
6. LIBERO 에피소드 공개 범위가 README와 CLAIMS에서 다르게 적힌 이유 (D11).

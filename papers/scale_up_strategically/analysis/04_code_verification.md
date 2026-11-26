# 04. 공식 코드와 논문 대조

대상: [yqi19/CoRL2026-Scale-Up-Strategically](https://github.com/yqi19/CoRL2026-Scale-Up-Strategically), 커밋 `b5b32418bd070b2ce55f3fcde21d742306331ce0`(2026-10-12 마지막 커밋, 커밋 4개). 얕은 clone을 **정적으로만** 읽었고 실행, import, 설치는 하지 않았습니다. 이 노트에서 코드를 옮겨 적은 부분은 `analysis/recompute.py`, `analysis/toy_seen_fraction.py`의 지지집합 리스트뿐입니다(두 파일의 설명에 저장소와 커밋이 적혀 있습니다).

## 1. 저장소 구성

| 폴더 | 내용 | 논문과의 대응 |
|---|---|---|
| `maniskill/` | ManiSkill 3 포크, 환경 `VerbObjectColor-v1`과 모션 플래너 솔버, 자산 | §3, 부록 A, C.3 |
| `data_collection/conflict_experiment/` | FDR용 대각선 충돌 데이터 수집(300시연) | §4.1, 부록 B |
| `data_collection/collection_strategy/` | V(`stair`), L(`Lrandom`), Random 지지집합과 수집 | §5, 부록 C.1–C.3 |
| `finetune/gr00t`, `finetune/pi05` | 미세조정 레시피(상위 저장소 커밋 고정 패치) | 부록 C.3 |
| `evaluation/` | 롤아웃 클라이언트(GR00T는 zmq, π0/π0.5는 OpenPI 웹소켓), 격자 평가, VLM 판정기, FDR/FDH 계산 | §4, §6.1, 부록 B.2, B.3 |

코드에 없는 것(README가 직접 밝힌 것 포함): 실로봇 코드와 데이터, Complete 전략("not implemented"), OpenVLA-oft·X-VLA 평가기, Genie-Envisioner 평가기(비공개 추론 패키지 의존으로 제외했다고 적음), π0.5의 Verb-Color-Object·All-Factor 클라이언트(OpenPI 포크로 안내), 사람 판정 주석. 체크포인트와 데이터는 프로젝트 페이지에서 링크한다고 적혀 있습니다(이 분석에서는 받지 않음).

## 2. 논문과 일치하는 항목

| 번호 | 항목 | 코드 위치 | 판정 |
|---|---|---|---|
| M1 | FDR = (N_f1 − N_f2)/(N_f1 + N_f2), 합이 0이면 NA | `evaluation/vlm_judge/fdr_fdh.py` `fdr()` | 일치(식 1의 ε 대신 NA 처리) |
| M2 | Copeland: 쌍마다 FDR > τ이면 f1 +1, FDR < −τ이면 f2 +1, 동점 0, τ = 0.05 | `copeland()` | 일치(식 3. FDR 부호 대칭이라 f_i 기준 합과 같음) |
| M3 | "neither"와 "error"를 분모에서 제외 | `counts_from_vlm()`은 f1, f2 라벨만 셈 | 일치(부록 B.2). 단 neither·error 건수를 따로 내지 않음 (D7) |
| M4 | 훈련 데이터 300시연, 훈련 대각선 지시 수 | README, `collect_convert.sh` 환경변수 `NUM_DEMOS=300` | 일치(부록 B.1) |
| M5 | 어휘와 6,480 지시 | `README.md` 표, `training_vocab.py` | 일치(표 4) |
| M6 | V(stair) f = 18: 첫 행 + 주대각선 + 부대각선 | `stair_18_pairs()` | 일치(부록 C.1 서술) |
| M7 | GR00T-N1.7, π0.5/π0 미세조정 대상, 시연은 모션 플래너 스크립트 | `finetune/*`, `data_collection` | 일치(부록 C.3) |
| M8 | 평가 시드 3개 (42, 40, 41), 같은 체크포인트 | `run_pairwise_grid.sh` | 일치(그림 4 캡션) |

## 3. 차이와 확인 필요 항목 (D)

### D1. 여섯 정책 중 코드로 확인되는 것은 세 개

README의 "Not included"와 평가 폴더 구성을 보면 평가 러너는 GR00T-N1.7과 π0/π0.5뿐입니다. 표 1의 OpenVLA-oft, X-VLA, Genie-Envisioner 열(전체 60칸 중 30칸)과 이에 근거한 FDH 세 행은 이 저장소로 다시 만들 수 없습니다. 특히 FDH가 다른 모델인 OpenVLA-oft가 포함되지 않습니다. 중요도: 중(검증 가능 범위 한정).

### D2. 인자 변화량 f는 Hamming 거리의 합이 아니라 설정 이름입니다

논문 §6과 부록 C.2는 f를 연속 구성 사이의 Hamming 거리 합으로 정의하고, 전략마다 f가 다르면 Random 시연을 더해 같게 맞춘다고 씁니다. 코드에는 Hamming 거리 계산도, f를 맞추기 위한 추가 Random 시연 로직도 없습니다(저장소 전체에서 "hamming", "equaliz"를 찾지 못함). README는 `FACTOR_COUNT`를 "격자의 크기"(쌍별 6/12/18, Verb-Color-Object 15/20/30, All-Factor 50)로 정의하며 격자 크기는 변수 선택과 훈련 칸 목록으로 결정됩니다. 따라서 "같은 f"의 의미는 논문(이동 비용)과 코드(격자 설정)에서 다릅니다. 실로봇 코드가 없어 4×4 f = 12의 구현은 확인할 수 없습니다. 중요도: 높음(비교 단위).

### D3. 같은 f에서 훈련 칸 수가 전략마다 다르고 평가는 전체 격자입니다

README와 `lib/pairwise_factor_patterns.py`로 센 훈련 칸 수(쌍별):

| f | V(`stair`) | L(`Lrandom`) | Random |
|---|---|---|---|
| 6 | 5 | 4 | 3 |
| 12 | 10 | 9 | 6 |
| 18 | 15 | 14 | 9 |

평가는 f와 무관하게 격자 전체를 돕니다(`groot_grid_eval.py`의 `cells = [(c, x) for c in f1vals for x in f2vals]`, 칸마다 ceil(200/칸 수)회). 6×6이면 훈련 칸 비율이 V 41.7 %, L 38.9 %, Random 25.0 %(f = 18)입니다(`03` V8). 논문의 성공률이 훈련 칸을 포함한 전체 평균인지, 미훈련 칸만의 평균인지는 본문 §6.1에 명시되지 않았고, 그림 8의 칸별 성공(훈련 칸 표시 포함)은 실로봇에서 전체를 센다는 것을 보여 줍니다. 코드의 README 표는 "evaluates on all cells, including held-out ones"라 적습니다. 중요도: 높음(이득의 해석).

### D4. Random 기준선의 구성

논문(§5, 부록 C.1): 전체 하위공간에서 조합을 균일 표집해 예산 n까지. 코드: f마다 **고정된 작은 지지집합**을 먼저 뽑고(쌍별 3/6/9칸, 전체의 절반 격자를 쓰는 변형: f = 6이면 2×2 격자에서 3칸, f = 12이면 4×4에서 6칸, f = 18이면 6×6에서 9칸) 그 지지집합에서 중복 허용으로 `NUM_DEMOS`개를 뽑습니다(README: "`random` draws uniformly with replacement from its support"). 전략 V와 L의 구조 전략은 "재섞은 전체 통과"를 순환합니다. 즉 Random의 시연 하나하나가 중복된 칸에서 나올 수 있고, 시연 밀도가 칸 9개에 집중됩니다. 작은 f에서 Random의 격자(2×2, 4×4)와 V의 격자(3×3 인덱스, 4×4)가 같은 영역이 아닙니다. 중요도: 중.

### D5. L 기준선의 구성

논문(§5): "각 인자를 한 값에 고정하고 다른 인자를 균일 표집". 부록 C.1: 한 행과 한 열의 "L" 모양. 코드의 `Lrandom`: f = 6은 2×2 정사각형 4칸(L 모양이 아님), f = 12는 첫 행 + 첫 열 + 내부 2칸, f = 18은 첫 행 + 첫 열 + **무작위 내부 3칸**(시드에 따라 지지집합이 달라짐). All-Factor는 다섯 단일 인자 광선 + 무작위 4칸. 논문의 L은 순수 L 모양이지만 코드에서는 무작위 칸이 섞입니다. 중요도: 중.

### D6. All-Factor의 지지집합 크기

README의 수치(코드로 계산): `stairvss`(V) 42칸(f = 50, n = 400), `Lrandom` 29칸(f = 50), `random` 12칸(f = 12). `random`은 `--factor-count`가 12(또는 20, 40의 하위 격자)여야만 하고, 논문 그림 4 panel 11, 12의 "f = 50"과 맞는 Random 설정은 코드에서 f = 12의 12칸입니다. 시연 예산은 V n = 400, Random n = 800으로 비교하므로 칸당 시연 수가 V ≈ 9.5, Random ≈ 67입니다. 6,480개 지시 중 12칸에 800개를 집중한 Random은 전체 공간의 약 0.2 %만 봅니다. 논문 본문 "Random: 전체 공간에서 무작위 조합"과의 관계를 부록에서 구체적으로 알 수 없습니다. 중요도: 중.

### D7. VLM 판정 프롬프트와 파싱

코드(`vlm_eval.py`)와 부록 B.2의 프롬프트 사이:

- 부록은 응답 필드를 4개(ROBOT_ACTION, OBJECT_TOUCHED 또는 ACTION_TYPE, FACTOR_FOLLOWED, REASONING)로 보이고 파싱을 "다섯 개 KEY: value 줄"이라 적습니다. 코드의 프롬프트는 CONFIDENCE 필드가 있는 6줄 형식입니다.
- 프롬프트에 들어가는 **값 사전**이 논문 어휘와 다릅니다. SIZES = [smallest, second smallest, middle-sized, second largest, largest, biggest], SPATIALS = [on the left, on the right, in the middle, in front, at the back, **in the center**]. 논문 어휘(small, large, smaller, larger, smallest, largest / left, right, middle, front, behind)와 값의 목록과 개수(spatial은 6개)가 다릅니다. 판정기가 크기 인자를 어떻게 설명받는지(예: size 쌍에서 객체 A를 "smallest"로 서술)가 훈련 지시의 크기 단어와 일치하지 않는 부분이 있을 수 있어 확인이 필요합니다. 판정기는 영상과 이 설명만 보므로 쌍에 따라 설명이 정확하지 않으면 라벨에 영향이 있을 수 있습니다.
- 영상은 롤아웃의 `video/ep000.mp4`만 봅니다(첫 에피소드, 손목 영상은 쓰지 않음).
- 호출이 4번 실패하면 라벨이 "error"로 저장되고, 이것도 FDR 분모에서 빠집니다. error와 neither의 건수는 `_summary.txt`에는 남지만 논문에는 보고되지 않았습니다(표 1에서 분모 400을 못 맞추는 칸이 34개: `03` V2). 중요도: 중(재현 가능성).

### D8. 충돌 평가의 표집 방식

논문 §4.1: "OOD 분포에서 400롤아웃을 균일 추출". 코드(`run_conflict.sh`): OOD 쌍 (i, j)를 `rng.choice(all_pairs)`로 **200번 복원 추출**하고 쌍마다 장면 두 개(인자마다 하나)를 돌려 400롤아웃, 에피소드는 쌍·장면당 1개, **시드 42 하나**입니다. 쌍 후보는 6×6이면 i ≠ j인 30쌍, spatial이 들어가면 20쌍이며, **size가 들어간 쌍(verb_size, size_object, color_size)은 (small, large), (smaller, larger), (smallest, largest) 안의 교환 6쌍뿐**입니다. 그 결과 size 쌍은 같은 OOD 칸이 평균 33번씩 반복되고, 시드가 하나라 FDR의 표본 변동(`03` V9)은 롤아웃 수준만 해당합니다. 논문은 size 쌍의 OOD 범위가 좁다는 것을 본문이나 부록에서 말하지 않습니다(코드 README에는 있음). 중요도: 중.

### D9. 평가 프로토콜 세부가 본문에 없는 값

| 항목 | 충돌(FDR) | 격자 평가 | All-Factor |
|---|---|---|---|
| 최대 스텝 | GR00T 300, π0 150 | 환경 기본(120), GR00T 클라이언트 150 상한 | 500 |
| 재계획(청크당 실행 행동) | GR00T 5, π0 10 | 10 | 5 |
| 시뮬레이터 | GR00T gpu, π0 cpu | 같음 | gpu |
| 난이도 | 환경 기본 | `TASK_DIFFICULTY` 1.5 ("HARD") | 환경 기본 |
| 방해물 | 경쟁 인자 값을 가진 물체 1개 | 같은 색, 탐침 인자만 다른 방해물 | 70 % 에피소드는 없음 |

논문 부록 C.3은 "각 모델의 기본 학습 설정으로 수렴까지"라고만 적고, 코드의 GR00T 미세조정은 `max_steps 10000`, 전역 배치 32, 학습률 1e-4, GPU 4장이며 π0.5는 기본 스텝 수에 GPU 8장입니다. 즉 "수렴까지"는 고정 스텝 수 학습입니다. 정책마다 최대 스텝과 재계획이 달라서 FDR을 모델 사이에서 비교할 때(표 1) 평가 조건이 같지 않습니다. 중요도: 중.

### D10. 평가 에피소드 수

README 표: 쌍별 격자는 "목표 200, 칸당 ceil(200/칸 수)회"이므로 6×6은 216(= 36칸 × 6회), 6×5는 210(= 30칸 × 7회)으로 목표를 넘는 수입니다. 논문은 시드당 "200 롤아웃"입니다. All-Factor는 200칸 1회(일치). 차이가 작지만 보고된 롤아웃 수의 근거를 바로 확인하게 해 줍니다. 중요도: 낮음.

### D11. π0.5 검증 범위

공개 격자 러너는 GR00T 여섯 설정과 π0의 color_size, color_spatial 두 설정뿐입니다(`evaluation/README.md`). 논문 그림 4에서 π0.5로 보고된 panel 1–4, 9, 11은 이 저장소의 러너로는 평가할 수 없습니다(OpenPI 포크 안내). 훈련(`finetune/pi05`)은 π0.5와 π0 공통입니다. 중요도: 낮음(범위).

### D12. 환경 기반 FDR

`fdr_fdh.py --env-dir`는 시뮬레이터가 알려 주는 이중 축 성공(`overall_<f1>_success`, `overall_<f2>_success`)으로 같은 FDR을 계산합니다. 논문은 VLM 판정만 보고합니다. 두 방식의 일치 여부는 논문에 없습니다. 시뮬레이션에서는 어느 물체를 집었는지 환경이 알 수 있으므로 VLM 판정 일치율(표 5)을 환경 값으로도 확인할 수 있는 구조입니다. 중요도: 정보(대안 존재).

### D13. 라이선스와 데이터

저장소 최상위에 라이선스 파일이 없습니다(GitHub 라이선스 표기 없음). `maniskill/` 폴더에는 Apache-2.0 LICENSE와 서드파티 LICENSE가 있습니다. 데이터셋(All-Factor ManiSkill, Hugging Face)은 카드에 라이선스 표기가 없고 약 106 GB(API 메타데이터)입니다. 중요도: 정보.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**논문 실험은 이 맥에서 재현할 수 없습니다. 표 재계산은 완료했고, 축소판(장난감 정책으로 FDR/FDH 설계와 수집 전략 비교)은 구현할 수 있습니다.**

| 구성 요소 | 이 맥에서 | 이유와 노력(추정, 측정 없음) |
|---|---|---|
| 표 1, 2, 3, 5, 6, 7 재계산, 통계 | 가능, 완료 | `analysis/recompute.py`, 수 초 |
| `fdr_fdh.py`(FDR, Copeland 계산) | 가능 | 순수 Python. 입력(판정 jsonl)이 있어야 함. 입력 생성은 롤아웃과 API 호출이 필요 |
| ManiSkill 시연 수집(모션 플래너) | 불가(확인 전) | README가 모션 플래닝에 Linux(`mplib`)를 요구. 환경은 SAPIEN 기반이며 macOS 지원은 확인하지 못함 |
| GR00T-N1.7(3B), π0.5 미세조정 | 불가 | 4–8 GPU 전제, 10,000스텝 |
| VLM 판정(Gemini-2.5-Flash) | 조건부 | API 키와 롤아웃 영상 4,000개 필요. 이 분석에서는 호출하지 않음 |
| 실로봇(UR5, GELLO) | 불가 | 하드웨어, 코드 없음 |
| 축소판: 작은 정책으로 상관 훈련 → FDR/FDH → V, L, Random | 가능 | torch(MPS), 합성 격자 과제 또는 소형 환경. 3–7일(추정). 합성 판(훈련 칸 비율 분리 포함)의 최소판은 이 분석에서 수십 초 규모로 만들어 둠(`analysis/toy_seen_fraction.py`) |

공식 코드는 이 맥에서 실행하지 않았고 다른 환경에서의 실행 가능성은 확인하지 않았습니다.

## 5. 재실행용 정보

- 이 폴더의 스크립트는 `repo/`를 읽지 않습니다. 코드에서 옮긴 지지집합 리스트는 스크립트 안에 있습니다.
- 정적 읽기를 다시 하려면: `git clone https://github.com/yqi19/CoRL2026-Scale-Up-Strategically repo && git -C repo checkout b5b32418bd070b2ce55f3fcde21d742306331ce0`.

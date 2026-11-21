# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 대상: [S-sirigiri/FAKTUAL](https://github.com/S-sirigiri/FAKTUAL), 커밋 `7aa0e6e3a97b065b0460fa355a71280a8d94a194` (2026-11-09), 얕은 복제 `repo/`.
- 방법: **정적 읽기만.** 코드를 실행, import, 설치하지 않았습니다. 패키지도 설치하지 않았습니다. 복제는 `git clone --depth 1`로 받았습니다. 저장소 내용은 신뢰하지 않는 데이터로 다뤘습니다.
- 읽은 파일: `README.md`, `kernel_diversity_lib/diversity_metrics.py`(253줄), `maximizer.py`(427줄), `kernels.py`(818줄, 주요 클래스), `embeddings.py`(95줄), `dataset_reducer.py`(인자와 실행 분기), `HDF5_handler.py`(시간 축과 임베딩 부분), `utils/kern_utils.py`, `requirements.txt`, 저장소 파일 목록.
- 읽지 않은 것: KSig 포크(RFSF 구현 본체), sigkernel 포크, `hflerobot_dataset_handler.py`, `convex_combination.py`, `robomimic/`과 `baselines/`의 학습 코드.

## 1. 설치 요구사항과 라이선스

| 항목 | 내용 | 확인 |
|---|---|---|
| Python | 3.10 이상 | README |
| GPU | `cupy-cuda12x` 필요. README는 "CUDA 환경이 필요하며 Torch 서명 솔버를 CPU에 두어도 전체 파이프라인은 CPU 전용이 되지 않는다"고 적음 | README, requirements.txt |
| 최상위 import | `diversity_metrics.py`, `kernels.py`, `kern_utils.py`가 파일 맨 위에서 `import cupy as cp`를 함 | 정적 확인 |
| 포크 의존 | `sigkernel`과 `KSig`를 S-sirigiri 포크에서 git으로 설치 | requirements.txt |
| 서브모듈 | `lerobot/` (huggingface/lerobot). 복제에서 비어 있음 | .gitmodules |
| 가중치 | CLIP(`openai/clip-vit-base-patch32`)과 DINOv2(`facebook/dinov2-base`)를 처음 사용할 때 내려받음 | embeddings.py |
| 라이선스 | **루트에 LICENSE 없음.** GitHub API `license: null`. 코드 재배포 권한 없음 | 정적 확인, API |
| 벤더링 라이선스 | robomimic (MIT), baselines/cupid (MIT), baselines/demonstration-information (MIT) | 각 LICENSE 파일 |

판정: 이 맥(CUDA 없음)에서는 공식 라이브러리를 **그대로 import하지 못합니다.** 최상위에서 CuPy를 가져오기 때문입니다. 이것은 정적 판단이며 실행으로 확인하지 않았습니다.

## 2. 논문 ↔ 코드 대조 (D 항목)

판정 기호: 일치 (맞음), 차이 (다름), 미확인 (확인 못 함), 누락 (논문이 말하지만 코드에 없음).

| 번호 | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| D1 | 이미지 임베딩 | "ViT [67] 임베딩"(§4, 부록 B.3), 모델명 없음 | 기본 CLIP ViT-B/32(`--embedding clip`, 512차원, L2 정규화). DINOv2-base(`--embedding dinov2`, CLS 토큰, 768차원). 선택적 무작위 선형 투영(`--embedding-dim`) | **차이 (명시 안 됨).** 논문의 "ViT"는 CLIP 기본 설정과 일치할 수 있지만 확정 불가 |
| D2 | 커널 대역폭 | 본문과 부록에 대역폭 규칙이 없음 | CLI 기본값: `compute_bandwidth(data) = median(data) / log(n) + 1e-3` (`kern_utils.py`). `data`는 (n, T, d) 배열의 **모든 원소 값**. 쌍별 거리가 아님. 쌍별 거리 버전 `__compute_bandwidth`는 구현되어 있지만 호출되지 않음(밑줄 두 개 private, 정적 확인) | **차이 (논문 미기재) + 주의.** 원소 값의 중앙값은 부호가 섞인 상태 데이터에서 0 근처가 될 수 있고, 그러면 대역폭이 작아져 커널이 단위행렬에 가까워짐. 실제 영향은 미측정 |
| D3 | 혼합 목적 큐레이션 | 엔트로피 최대 mp개 후 행렬식 최대 m(1−p)개, p ∈ [0.5, 0.9] 권장 (§5, §7) | `--maximizer mixed-objective`: `--split-percentage` 기본 **30**(= p 0.3), `--secondary-metric` 기본 **shannon**(행렬식 아님). 두 단계 모두 `BlackBoxMaximizer.greedy_local_search(max_iters=0)`(입력 순서 탐욕). 논문의 stochastic greedy가 아님. README: "논문의 혼합 선택 방법을 재현한 것으로 보지 말 것" | **차이 (기본값).** `--split-percentage 50–90 --secondary-metric det`로 주면 논문 구조에 가까워지지만, 1단계와 2단계의 탐욕 방식은 여전히 다름 |
| D4 | von Neumann 엔트로피 | H_vN = −Tr[(K/n) ln(K/n)] (정의 4) | `VonNeumannEntropy._von_neumann`은 `self._shannon(K)`를 그대로 호출. 행렬 로그를 계산하지 않음 | **일치 (수학적으로 동치, Lemma 1).** 다만 구현이 같은 함수이므로 두 값이 항상 같음. "von Neumann 결과가 Shannon과 같다"는 실험적 확인이 아님 |
| D5 | 정규화 K_ii = 1 | 정의 3: K_ii = 1을 가정하고 고유값 합이 1 | `_shannon`은 `K/n`만 하고 대각 정규화를 하지 않음. torch 시그니처 커널 경로(`SignatureKernelTorch`)는 `k(x,x)=1`로 정규화함(kernels.py 201–215). **기본 RFSF 경로(`RandomFourierSignatureFeaturesKernel`)와 `KernelMatrix` 래퍼는 정규화하지 않고 KSig 출력을 그대로 반환**(kernels.py 96–111, 601–633) | **미확인 (중요).** KSig 특징 맵의 노름이 1인지 확인해야 함. 아니면 엔트로피가 정의한 분포가 아니며 Vendi 값의 절대 크기가 달라짐. 순위는 달라지지 않을 수 있음 |
| D6 | stochastic greedy | 알고리즘 2: r = ⌈(n/m) log(1/ε)⌉, 양의 이득 조건 없이 e*를 추가 | `stochastic_greedy`: r 계산 동일, ε 기본 0.01. 양의 이득 검사는 주석 처리되어 항상 추가 | **일치.** 알고리즘 2와 같음 |
| D7 | 탐욕 기본 방식 | "greedy 전략" (§5) | CLI 기본은 lazy greedy(정확). `--stochastic-greedy`를 켜야 확률적 방식 | **미확인.** 논문 주 결과가 어느 방식인지 본문에 없음 |
| D8 | RFSF 하이퍼파라미터 | 주 실험 설정은 본문에 없음 (부록 H.1만 레벨 절제) | CLI 기본: `n_levels=5`, `n_components=100`, `method=TRP`. 무작위 시드는 `random_state=0`으로 **고정**(`--random-state`와 무관) | **미확인 (논문 미기재).** 시드 고정은 논문에 없음 |
| D9 | 시간 좌표 추가 | 부록 B.4: 시그니처의 재매개변수화 불변성 때문에 시간 좌표를 붙이는 것이 "유용할 수 있음" | 코드에는 시간 채널을 추가하는 부분이 없음 (정적 검색). KSig 내부 처리는 미확인 | **차이 (선택 사항).** 논문도 필수라고 하지 않음 |
| D10 | 길이 다른 궤적 | 시그니처는 길이가 달라도 비교 가능 (§2) | `HDF5_handler.py`: 마지막 시점을 반복해 패딩. 반복 점은 증분이 0이라 시그니처 적분에는 영향이 없다고 볼 수 있으나, RFSF 특징 맵에 반복 점이 미치는 영향은 미확인 | **미확인** |
| D11 | 시간 간격과 마지막 프레임 | 본문에 없음 | `--time-stride` 기본 1, 마지막 시점 포함(`--include-last` 기본 참) | 논문에 없는 설정 |
| D12 | 영상 임베딩 시점 | §4: 각 시점마다 임베딩. 부록 B.3: 첫 프레임 I(0)을 RBF 모달리티에 사용 | HDF5 경로는 **각 시점** 이미지를 임베딩(`HDF5_handler.py` 136–143). 첫 프레임만 쓰는 RBF 결합 경로는 `ConvexKernelMatrix`에 있으나 미확인 | §4 **일치**, B.3 **미확인** |
| D13 | 대역폭 휴리스틱 (결합 커널) | 식 (1) 등 대역폭 `h`가 있지만 값 규칙 없음 | `ConvexKernelMatrix.fit`: 같은 공식(중앙값 / log n + 1e-3) 모달리티별 적용 (kernels.py 750–755) | D2와 같은 문제 |
| D14 | 기준선 CUPID, DemInf, DemoSCORE | 기준선 설정은 Agia et al. [22]를 따름 | `baselines/cupid/`(CUPID, DemoSCORE 평가 스크립트 포함), `baselines/demonstration-information/`(DemInf) 벤더링 | **일치 (존재 확인).** 학습과 평가 재현은 하지 않음 |
| D15 | MetaWorld, SmolVLA, 실로봇 설정 | 부록 G.1.3과 G.2.3: "완전한 구현 세부 사항과 설정 파일은 공식 저장소에 있다". G.3.3(실로봇)은 표 4의 설정을 따른다고만 함 | 저장소에 `smolvla`, `metaworld`, `meta-world` 문자열이 없음. 실로봇 코드와 데이터도 없음 | **누락.** 논문 주장과 다름 |
| D16 | RoboMimic 데이터 | 부록 G.1.2: PH 200, MH 300 | 저장소에 데이터 없음. 어댑터는 `mask/train` 경로를 읽음 | 데이터는 외부 |
| D17 | 출력 방식 | 선택된 부분집합 사용 | 출력 HDF5는 원본을 복사하고 `mask/train`만 바꿈(README). 학습 코드가 마스크를 따라야 함 | 구현 방식 차이(실질 동일) |

- 정적 확인으로 **가장 중요한 두 항목은 D5(정규화)와 D2(대역폭)** 입니다. 둘 다 논문이 말하지 않는 세부이고, 엔트로피 값의 절대 크기와 큐레이션 결과의 선택 집합에 직접 영향을 줍니다.
- 논문의 수치 재현은 코드만으로는 안 됩니다. 정책 학습 설정(D15)이 빠져 있어 표 5, 6의 성공률을 재현할 수 없습니다.

## 3. 구현 세부 (본문 밖)

- 평가 경로: `dataset_reducer.py`는 `--get-score`(점수만 계산)와 `--maximizer`에 따라 선택을 수행합니다. `random` 선택은 `--random-state`를 쓰지만 RFSF 시드는 따로 고정입니다.
- `--k`는 0 이상 전체 데모 수 이하여야 합니다(코드 검사 있음).
- LeRobot 어댑터는 선택 결과를 HDF5가 아니라 JSON의 **정렬된 에피소드 ID 위치**로 씁니다(README). 위치가 `episode_index`와 같으려면 ID가 0부터 연속이어야 합니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)

**판정: 부분 구현 가능 (조건부). 논문 수치 재현은 불가.** 시간 추정은 측정하지 않았습니다.

| 구성 요소 | 이 맥에서 | 필요 자원 | 노력 (추정, 측정 없음) |
|---|---|---|---|
| 공식 라이브러리 실행 (`dataset_reducer.py`) | **불가** (CuPy와 CUDA 12 필요) | NVIDIA GPU | — |
| 시그니처 커널 엔트로피를 numpy로 독립 구현 (상태, 행동만, 절단 레벨 2–3, 정확한 PDE 또는 작은 레벨 전개) | 가능. n = 200, T를 10–20배 솎으면 CPU 수 분–수십 분으로 추정 | 손으로 짠 numpy 코드 | 2–4일 (추정) |
| 같은 구현에 RFSF(무작위 푸리에 특징) 추가 | 가능 (KSig 본체를 읽어야 함) | 포크 소스 정적 읽기 | 3–5일 (추정). KSig 본체를 읽기 전에는 불확실 |
| 영상 임베딩 (CLIP ViT-B/32) 추출 | 조건부. MPS에서 가능할 것이나 측정 안 함. 가중치 다운로드 필요 | HF 가중치 다운로드 (승인 필요) | 반나절 (추정) |
| RoboMimic 데이터로 큐레이션 선택 집합 비교 | 조건부. 데이터 다운로드 필요 | 데이터 수 GB 이상 가능 (미확인) | 1–2일 (추정, 다운로드 제외) |
| 표 6 재현 (BC-RNN 또는 확산 정책 학습, 시뮬 평가, 20 체크포인트 × 50 롤아웃 × 3 시드) | **불가** (학습 GPU와 시뮬 롤아웃 필요) | GPU 수십 시간 이상 (추정) | — |
| 표 5 재현 (SmolVLA 미세조정) | **불가** (GPU, 설정 파일 없음) | GPU, 설정 누락 (D15) | — |
| 그림 6 재현 (실로봇 4 과제) | **불가** | 실로봇, 데이터 미공개 | — |
| 엔트로피와 성공률 상관 (그림 4) 재현 | 불가 (정책 학습 필요) | — | — |

- 이 맥의 메모리(24 GiB)는 시그니처 커널의 n² 행렬(n = 300이면 90,000 항목)에는 충분합니다. 병목은 PDE 계산 시간과 정책 학습입니다.
- 공식 코드를 쓰려면 CUDA 머신이 필요합니다. 맥에서는 독립 구현만 가능하고, 이 경우 D2, D5의 확인이 먼저입니다.

## 5. 재실행

정적 분석 전용이므로 재실행 명령은 없습니다. 복제는 다음과 같이 했습니다.

```bash
git clone --depth 1 https://github.com/S-sirigiri/FAKTUAL diversity_metric/repo
```

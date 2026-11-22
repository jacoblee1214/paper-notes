# 04. 논문 ↔ 공식 코드 정적 대조와 구현 가능성

대상: 공식 저장소 AgibotTech/EWMBench (커밋 `3a5531c`, 2025-06-13, `--depth 1` clone, `repo/`). **정적으로만 읽었습니다.** 실행, import, 설치를 하지 않았고 `submodel/dinov2`(git submodule)와 가중치, 데이터는 받지 않았습니다. 저장소 내용은 신뢰하지 않는 데이터로 취급했습니다.

## 1. 대조 범위

| 파일 | 읽은 내용 |
|---|---|
| `EWMBench/__init__.py` | 평가 흐름 `evaluate()`, CSV 병합 `merge_all_metrics_to_csv()`, 평균 행 |
| `EWMBench/trajectory_consistency.py` | HSD, DYN, NDTW, 보간, 정규화 상수, 무효 궤적 처리 |
| `EWMBench/scene_consistency.py` | DINOv2 장면 지표 |
| `EWMBench/diversity.py` | CLIP 다양성 |
| `EWMBench/semantics.py` | BLEU, CLIPScore 설정 |
| `EWMBench/caption.py` | 캡션 프롬프트, Qwen2.5-VL 호출 |
| `EWMBench/utils.py` | 전처리 transform, 모델 경로 |
| `processing/detection_tracking.py`, `processing/video_resize.py` | 궤적 추출, 해상도 |
| `evaluate.py`, `config.yaml`, `dino_config.yaml`, `requirements.txt`, `README.md` | 실행 방법과 설정 |

읽지 않은 것: `EWMBench/basic_metrics.py`(PSNR, SSIM, 논문 밖), `EWMBench/distributed.py`(분산 실행 보조), `submodel/qwen_vl_utils/vision_process.py`(표준 Qwen 도구의 변형).

## 2. 대조표

판정: **일치**, **차이**(논문에 없거나 다르게 적힘), **빈칸**(코드가 없음), **확인 필요**.

| 번호 | 논문 | 코드 | 판정 |
|---|---|---|---|
| D1 | §3.3: 3개 궤적 중 **HSD 기준**으로 최고를 고른다. §4: "best-of-three" (기준 미기재) | 최고 선택 로직 없음. `merge_all_metrics_to_csv`가 모든 trial 행을 저장하고 `MEAN` 행은 전체 평균. 저장소에 'best' 검색 결과 없음 | **차이.** 표 2가 최고 선택 후 값인지, 전체 평균인지 코드로 확인할 수 없음 |
| D2 | §3.4: 단계 설명(Events)을 GT와 **CLIP 점수**로 비교 | `semantics.py`의 `compute_semantics`: CLIPScore 설정의 `key`가 `"General"`(전역 캡션). BLEU도 General | **차이.** CLIP은 단계 설명이 아니라 전역 설명으로 계산 |
| D3 | 표 2: 축별 평균과 Overall, 0–1 정규화 | 세부 지표의 trial별 값과 평균만 저장. Motion Avg, Semantics Avg, Overall, 정규화 코드 없음 | **빈칸.** 표 2는 저장소만으로 재현되지 않음 |
| D4 | HSD, NDTW, DYN은 "0–1로 정규화" (§4.1). 상수 없음 | `eval_traj`: `hsd_max=24.979`, `dyn_max=22.519`, `ndtw_max=50.202`로 나누고 상한 초과는 1.0 | **차이.** 임의 상수가 점수 척도를 정함. 논문에 없음. 민감도 분석 없음 |
| D5 | 무효 궤적 처리 언급 없음 | `one_traj_interpo_fill`: 결측 비율 > 0.80이면 무효로 보고 HSD, DYN, NDTW를 모두 0 | **차이.** 논문에 없음 |
| D6 | 부록 A.1: 각 손의 궤적 볼록 껍질에서 **가장 넓은 손**을 고른다 (예측 궤적 기준처럼 읽힘) | `HAUSDORFF`, `DYNAMICS`, `NDTW`가 모두 **GT 궤적**의 `farthest_distance`(볼록 껍질 회전 캘리퍼)로 손 인덱스를 고름 | **차이(해석).** 손 선택이 GT 기준. 예측 궤적이 다른 손을 움직여도 GT 손이 평가됨 |
| D7 | §3.4 "Logical Error Punishment": 오류에 벌점 | `Overall_Constraints`(참이면 1)의 평균. 키가 없으면 경고 후 건너뜀 | **차이.** 벌점이 아니라 보상 방향. 건너뛴 항목은 분모에서 빠짐. Logics 분모 90(`03` §2) |
| D8 | 표 2 CLIP 값 0.83–0.89. 척도 변환 언급 없음 | `CLIPScore` 원값을 그대로 저장(코드에 /100 없음). torchmetrics CLIPScore의 척도는 설치하지 않아 확인하지 못함 | **확인 필요.** 척도 변환 규칙이 기재되지 않음 |
| D9 | §3.3, 부록 A.3.1: 연속 프레임과 첫 프레임의 패치 코사인 유사도 평균 | `scene_consistency`: `(sim_pre + sim_fir)/2`, 음수는 `max(0, ·)`로 자름, 입력은 정사각 패딩 후 518 리사이즈 | **일치**(주 방식). 클리핑과 패딩, 518 크기는 논문에 없음 |
| D10 | §3.3: 3개 후보 CLIP 특징의 1 − 유사도 | `compute_cost_matrix`: 같은 에피소드의 trial 쌍 평균 (1 − cos), 프레임 평균 특징, ViT-B/32 | **일치** |
| D11 | 식 (3) HSD = 1/d_symH | `hs = 1/hi`(대칭 Hausdorff는 두 방향 최댓값). 그 뒤 D4 상수로 나눔 | **일치**(식), 상수는 D4 |
| D12 | 식 (4) NDTW = 1/NDTW | `di = di/len(path)` 후 `1/di`. 경로 길이 정규화는 논문에 없음 | **차이.** 식에 없는 정규화 |
| D13 | 식 (5)–(7): α = 0.007, β = 0.003, VR과 AR 진폭 보정 | `score = 0.007*vel_score + 0.003*acc_score`, VR, AR 계산 ε = 1e-8 | **일치**(식과 상수) |
| D14 | 부록 A.1: 프레임 640×480, 30 FPS로 정규화 | `video_resize.py`: 640×480 cubic 리사이즈만. FPS 재샘플 코드 없음 | **차이.** 30 FPS 정규화 코드 없음 |
| D15 | 부록 A.1: YOLO-World 미세조정, 1,451 프레임, 100 epoch. 부록 A.3.1: DINOv2 20,000 반복 | 가중치만 HF에 공개. 학습 스크립트와 데이터 없음. `submodel/dinov2`는 upstream submodule | **빈칸.** 미세조정 재현 불가 |
| D16 | §4.2 사람 평가, §4.3 궤적 실험(그림 6C), 그림 6B VBench 비교, 정성 사례 | 해당 코드 없음 | **빈칸** |
| D17 | 논문 밖 | `psnr`, `ssim` 차원(커밋 2025-06-11 "add psnr and ssim"). README에는 없음 | 논문에 없는 추가 기능 |
| D18 | 논문에 장치 언급 없음 | `evaluate.py`가 `torch.device("cuda")` 고정. `caption.py`는 `device_map="cuda:0"`과 `.to('cuda')`. `detection_tracking.py`는 `.to('cuda:0')`. 장면과 다양성은 CPU 대체 있음 | **차이(구현 조건).** CUDA 고정 경로가 있음 (§4) |
| D19 | §3.2, 부록 A.1: 궤적 추출은 YOLO-World 검출 후 BoT-SORT 추적 | `detection_tracking.py`: `model.track(conf=0.8)`, 프레임당 클래스별 최고 신뢰도 상자 1개(왼손 cls 0, 오른손 cls 1), 없으면 (-1, -1), 좌표 ÷ (640, 480) | **차이.** 임계값 0.8과 "클래스당 하나" 규칙은 논문에 없음 |
| D20 | (D19와 관련) 추적 ID와 상자의 대응 | `zip(filtered_boxes, track_ids, …)`에서 `track_ids`는 **모든** 상자의 ID 목록. `boxes.id`가 없으면 `track_ids = []`라 그 프레임의 상자가 버려짐 | **확인 필요(정적 판독).** 궤적 값에 영향이 있는지는 실행하지 않아 미확인. 추적 ID가 없는 프레임이 많으면 궤적이 (-1, -1)로 채워짐 |
| D21 | §3.1: 궤적 평가는 6D 자세 궤적 옵션 | 평가는 영상에서 얻은 **2D 상 궤적**. 6D 자세는 쓰지 않음 | **차이.** 용어가 같아 혼동될 수 있음 |
| D22 | 부록 A.3.2 식 HSD, NDTW, DYN 설명과 본문 §3.3 "DYN은 Wasserstein" | `scipy.stats.wasserstein_distance`, 가중치 1/(N−1), 1/(N−2) | **일치** |

부차 관찰: `trim_trajectory`와 `calculate_means_and_variances`는 정의만 있고 호출되지 않습니다(죽은 코드). `scene_consistency.py`의 `(len(images)-1)`은 프레임 수가 1일 때 0으로 나눌 수 있습니다.

## 3. 불일치와 빈칸

### 3.1 평가 결과를 재현할 수 없는 이유

1. 표 2의 집계 식과 정규화(D3, D4)가 코드에 없습니다. 같은 CSV에서 Overall을 다시 만들려면 저자의 집계 규칙이 필요합니다.
2. 최고 후보 선택(D1)이 코드에 없으므로, CSV의 `MEAN` 행이 논문 표 값과 같은지 알 수 없습니다.
3. 캡션 기반 점수는 Qwen2.5-VL-7B 출력에 의존하고, 모델 버전과 디코딩 설정(샘플링 여부)이 적혀 있지 않습니다. `caption.py`는 `max_new_tokens=2048`, 기본 생성 설정을 씁니다.

### 3.2 라이선스 차이 (저장소와 논문)

- arXiv 논문: CC BY 4.0.
- 저장소 README: "All the data and code within this repo are under CC BY-NC-SA 4.0". 저장소에 LICENSE 파일은 없습니다.
- HF 데이터와 가중치: CC BY-NC-SA 4.0.

논문의 그림은 CC BY 4.0으로 쓸 수 있지만, 코드와 데이터는 비상업 조건입니다. 우리 공개본은 저장소 코드를 복사하지 않습니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 조건부. 궤적 지표의 CPU 재구현은 가능하지만, 논문 표 2의 전체 재현은 이 맥에서 불가능합니다.** 이유는 세 가지입니다.

### 4.1 막히는 곳

| 구성 | 필요 | 이 맥에서 | 근거 |
|---|---|---|---|
| 캡션 Qwen2.5-VL-7B (Logics, CLIP, BLEU 입력) | GPU 메모리, 2,100 영상 + GT 100 영상의 생성 | 가중치 약 15 GB(추정, bf16). 코드가 `cuda` 고정. 영상당 최대 2,048 토큰 생성이라 수 시간 이상 예상 (MPS 처리량은 측정하지 않음) | D18 |
| 장면 DINOv2 미세조정 가중치 | 가중치, `submodel/dinov2` | 가중치 파일 크기 미확인. 추론은 ViT-B/14 규모라 MPS 가능성 있음. 코드 경로는 `cuda`로 연결 | D15, D18 |
| 궤적 검출 YOLO-World | ultralytics 환경, `cuda:0` | 설치 필요. 코드가 `cuda:0` 고정 | D18, D19 |
| 다양성 CLIP ViT-B/32 | 가중치 | 가능(작음). 코드는 CPU 대체 있음 | D10 |
| 궤적 지표 (HSD, NDTW, DYN) | 궤적 `traj.npy`만 있으면 됨. numpy, scipy, fastdtw | **가능**. CPU 계산 | D4, D5, D12 |
| 사람 평가와 VBench 비교 | 주석 작업과 VBench 실행 | 불가(사람 평가 원자료 없음) | D16 |

### 4.2 세부 판단

- **재현의 중심이 되는 두 부분**: (a) 궤적 지표는 CPU에서 재구현할 수 있음. 입력은 저장소가 제공하는 샘플 데이터(`generated_samples.tar`, `gt_dataset.tar`)의 궤적 파일이 필요하고, 크기는 HF 메타데이터 기준 전체 약 270 MB입니다(받지 않음). (b) 캡션 기반 점수는 CUDA 전용 경로를 MPS나 CPU로 바꾼 뒤에야 돌릴 수 있는데, 7B 모델은 이 맥에서 느릴 가능성이 큽니다.
- **상수와 집계를 알 수 없는 문제**: D3, D4, D1은 저자에게 물어야 합니다. 코드만으로는 표 2를 재현할 수 없습니다.
- **학습 재현**: DINOv2와 YOLO-World 미세조정은 데이터와 레시피가 없어 불가능합니다. 가중치만 씁니다.

### 4.3 가능한 축소판 (조건부, 노력 추정)

1. **궤적 지표 CPU 재구현과 상수 민감도 (노력 1–2일):** 샘플 궤적을 받고(승인 필요), `trajectory_consistency.py`의 식을 그대로 옮겨 HSD, DYN, NDTW를 계산합니다. 상수 24.979 등을 ±20% 바꿔 순위가 바뀌는지 봅니다. numpy, scipy, fastdtw 설치 여부는 확인하지 않았습니다.
2. **캡션 기반 점수 MPS 포팅 (노력 1주 이상, 승인 필요):** `cuda` 문자열 제거, Qwen2.5-VL-7B 로딩과 생성 시간을 샘플 몇 개로 측정. 시간이 크면 중단합니다.
3. **전체 재현:** 이 맥의 자원으로는 불가능합니다. CUDA 기기가 있어도 저장소 requirements(torch 2.4.0, mmcv 2.2.0, mmdet 3.3.0, xformers 0.0.27) 설치와 데이터 준비에 1–2일이 듭니다.

## 5. 저자에게 물을 질문

1. 표 2의 최고 후보 선택은 어떤 기준으로 했습니까? HSD 기준입니까, 아니면 다른 기준입니까? 그 값은 코드의 `MEAN` 행과 같습니까? (D1)
2. CLIP 점수는 단계 설명(Events)과 계산했습니까, 전역 설명(General)과 계산했습니까? 코드는 General을 씁니다. (D2)
3. Overall, Motion Avg, Semantics Avg의 식과 0–1 정규화 방식은 무엇입니까? 상수 24.979, 22.519, 50.202는 어떻게 정했습니까? (D3, D4)
4. Logics 분모는 90입니까, 100입니까? 키가 없는 항목은 어떻게 처리했습니까? (D7)
5. 그림 6B의 EWMBench 점수 5.49, 4.75, 4.50, 4.05는 표 2의 어떤 값과 대응합니까? (`03` §3.2)
6. 사람 평가의 샘플 수, 주석자 수, 3위 점수, 일치도는 얼마입니까?
7. 손 선택 기준은 예측 궤적입니까, GT 궤적입니까? (D6)
8. 궤적 검출의 신뢰도 0.8과 클래스당 상자 하나 규칙은 논문에 적지 않은 설정입니까? (D19)
9. CLIP 척도(0.83–0.89)는 torchmetrics 원값의 어떤 변환입니까? (D8)
10. 30 FPS 정규화(부록 A.1)는 어느 코드에서 수행합니까? (D14)

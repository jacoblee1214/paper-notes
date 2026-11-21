# 04. 코드 대조: 논문 ↔ 공식 코드 (정적 읽기)

*대상:* ACT 논문 (arXiv:2304.13705v1, RSS 2023). 공식 코드 두 저장소를 얕게 복제해 정적으로만 읽었습니다. 실행, import, 설치는 하지 않았습니다.

| 저장소 | 커밋 | 마지막 push | 라이선스 | 역할 | 비고 |
|---|---|---|---|---|---|
| `tonyzhaozh/act` | `742c753` (2024-01-28) | 2024-07-23 | MIT | ACT 정책, DETR 기반 CVAE, 시뮬레이션 평가 | 얕은 복제 (정적 읽기, 저장소에 넣지 않음) |
| `tonyzhaozh/aloha` | `06369f0` (2024-04-19) | 2024-04-19 | MIT | 실로봇 수집·제어 (Interbotix ROS, 4대 카메라) | 얕은 복제 (정적 읽기, 저장소에 넣지 않음) |

코드 줄 번호는 위 커밋 기준입니다. "논문 §"은 논문 본문의 절 번호입니다.

## 1. 요약

- 핵심 구조(CVAE 인코더는 이미지 없이 qpos와 행동 청크만 받음, 디코더는 4대 카메라 ResNet18 특징 + qpos + z를 트랜스포머로 처리, 추론 때 z = 0, L1 재구성 + β·KL, 시간 앙상블 가중치 exp(−k·i))은 논문과 일치합니다.
- 논문 Table III의 하이퍼파라미터(lr 1e-5, batch 8, enc 4 / dec 7, ff 3200, hidden 512, heads 8, chunk 100, β 10, dropout 0.1)도 README 명령과 기본값에서 일치합니다.
- 차이는 대부분 **논문에 적히지 않은 세부 설정**입니다: 입력·행동 정규화, ImageNet 초기화, 시간 앙상블의 k 값(0.01), AdamW와 weight decay, z 차원, 학습 에폭 수. 이 중 k는 실험 결과를 좌우할 수 있는 값입니다.
- 논문과 코드가 다른 것은 질의(query) 임베딩 하나입니다: 논문은 고정 사인 임베딩이라고 하지만 코드는 학습 임베딩을 씁니다 (D1).
- **Mac 이식 장벽:** 공식 코드에는 `.cuda()`가 `imitate_episodes.py`에 8곳, `detr/main.py`에 2곳 (합 10곳) 하드코딩되어 있습니다. MPS로 돌리려면 장치 인자를 바꾸는 수정이 필요합니다. 라이브러리 문제는 아닙니다.

## 2. 대조 표

| ID | 항목 | 논문 | 공식 코드 (위치) | 판정 |
|---|---|---|---|---|
| D1 | 디코더 질의 임베딩 | 부록 C(논문 §C "The queries are fixed sinusoidal embeddings") | `nn.Embedding(num_queries, hidden_dim)`을 학습 파라미터로 씀 (`detr/models/detr_vae.py:54`, `transformer.py:56`의 `query_embed.weight`) | **불일치**: 코드는 학습 질의. 영향은 측정 안 됨 |
| D2 | 백본 초기화 | 본문에 ACT 백본의 사전학습 여부 없음 (VINN에는 "pretrained ResNet"이라 적음, 부록 설명) | torchvision `resnet18(pretrained=is_main_process())`로 ImageNet 가중치 사용, BatchNorm은 Frozen (`detr/models/backbone.py:93–94`). 백본 lr = 1e-5 (`imitate_episodes.py:51`) | **논문 누락**: "from scratch"(§V-C) 표현과 대비됨. 가중치 다운로드는 실행하지 않아 확인 못 함 |
| D3 | 입력·행동 정규화 | 언급 없음 | 행동과 qpos를 데이터셋 평균·표준편차로 z-점수화, 표준편차는 1e-2로 하한 (`utils.py:73–74`, `95–102`). 이미지는 /255 | **논문 누락** (표준 관행이지만 재현에 필요) |
| D4 | 시간 앙상블 감쇠 k | 식 w_i = exp(−m·i)만 있고 **m 값은 없음** (§IV-B, 표 III에도 없음) | `k = 0.01`, `exp(-k * arange)` (`imitate_episodes.py:255–256`). 추론에서만 적용, 매 시점 질의 (`query_frequency = 1`, `:192–193`) | **논문 누락 (핵심 값)**: 가장 오래된 청크에 가중치를 많이 줌 (w_0이 가장 큼) |
| D5 | 옵티마이저 | 알고리즘에 "ADAM" (§IV) | `torch.optim.AdamW`, weight decay 1e-4 기본값 (`detr/main.py:17`, `:87–88`). 논문 표 III에는 없음 | **세부 차이**: Adam인지 AdamW인지, weight decay 값이 논문에 없음 |
| D6 | z 차원 | 언급 없음 | `latent_dim = 32` (`detr_vae.py:67`) | **논문 누락** |
| D7 | 학습 에폭 | 표 III에 없음, 실로봇 "약 5시간 (RTX 2080 Ti)"만 적음 | README 예시 `--num_epochs 2000` (sim, `act/README.md:76–78`). 실로봇 에폭은 없음 | **논문 누락** |
| D8 | 시뮬레이션 카메라 | 실로봇 관측은 4대 (§IV-C, 그림 10) | 시뮬 설정은 `'camera_names': ['top']` 1대 (`constants.py:10,17,24,31`) | 논문은 시뮬 카메라 수를 명시하지 않음 (코드와 충돌은 없음) |
| D9 | 그리퍼 정규화 | "지지대 기구로 그리퍼 연속 제어" (§III) | 리더 그리퍼 관절을 열림·닫힘 값으로 [0, 1] 정규화 (`aloha/aloha_scripts/constants.py:38`, `real_env.py:148–149`). 팔로워에는 역변환 | **논문 누락** (행동 14차원 중 그리퍼 2개의 의미를 바꿈) |
| D10 | 평가 시 샘플링 | z = 0 결정적 추론 (§IV-C) | `torch.zeros([bs, latent_dim])` (`detr_vae.py:113`) | **일치** |
| D11 | CVAE 인코더 입력 | 이미지 없이 qpos와 행동 청크 (§IV-C) | `[CLS]`, qpos 투영, 행동 투영만 인코더에 넣음 (`detr_vae.py:90–95`) | **일치** |
| D12 | 재구성 손실 | L1 (§IV-C) | `F.l1_loss(..., reduction='none')`에 패딩 마스크 적용 (`policy.py:30–31`) | **일치** |
| D13 | KL 가중치 β | β = 10 (표 III) | `--kl_weight 10` (`act/README.md:76`), `loss = l1 + kl·kl_weight` (`policy.py:34`) | **일치** |
| D14 | 청크 크기 k | 100 (표 III) | `--chunk_size 100` (README), `num_queries = chunk_size` | **일치** |
| D15 | 트랜스포머 크기 | enc 4 · dec 7 · ff 3200 · hidden 512 · heads 8 · dropout 0.1 (표 III) | `imitate_episodes.py:54–56`, README의 `--hidden_dim 512 --dim_feedforward 3200`, `detr/main.py:43` (dropout 0.1) | **일치** |
| D16 | 학습률·배치 | lr 1e-5, batch 8 (표 III) | `--lr 1e-5 --batch_size 8` (README) | **일치** |
| D17 | 이미지 해상도·특징 맵 | 480×640, ResNet18 → 15×20×512, 1200×512 입력 (§IV-C) | 480/32 = 15, 640/32 = 20과 맞음. 카메라 4대 (`aloha/aloha_scripts/constants.py:9`) | **일치** |
| D18 | 제어 주기 | 50 Hz (§III) | `DT = 0.02` (`aloha/aloha_scripts/constants.py:14`) | **일치** |
| D19 | 행동 표현 | 절대 목표 관절각 14차원, delta는 성능 저하 (§IV-C) | 절대 관절각 (`real_env.py:145–149`), delta 옵션 없음 | **일치** |
| D20 | 로봇 모델 | 리더 WidowX-250, 팔로워 ViperX-300 (§III) | 리더 `wx250s`, 팔로워 `vx300s` (`real_env.py:41`, `record_episodes.py:71`) | **일치** |
| D21 | 실로봇 카메라 이름 | top·front·왼쪽·오른쪽 손목 (부록 B) | `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist` (`constants.py:9`) | **일치** (이름만 다름) |
| D22 | 배포 루프 | 실로봇 평가 (§V) | `eval_bc`가 `real_robot`이면 `aloha_scripts.real_env.make_real_env` 호출 (`imitate_episodes.py:181–184`) | 저장소에 포함. 하드웨어 필요 |
| D23 | 예시 과제 설정 | 논문 과제 6개 (§V-A) | `aloha_wear_shoe`(에피소드 1000 스텝, `constants.py:4–11`). 논문 과제 이름과 다름 | 참고: 공개 저장소 설정은 논문의 설정과 같지 않음 (에피소드 길이 400–700 스텝과 다름) |

## 3. 재현 관점에서 중요한 차이

1. **D4 (시간 앙상블 k):** 논문에 m 값이 없으므로 논문만 보고 재현하면 k를 맞출 수 없습니다. 코드의 0.01은 청크 100개 범위에서 가중치를 1에서 약 0.37까지만 줄입니다(exp(−0.01·99) ≈ 0.37). 즉 거의 균등한 평균에 가깝고, 논문 식의 설명("m이 작을수록 새 관측을 빨리 반영")과 같은 방향입니다. 그림 8(b)의 +3.3%p도 이 값에 의존하지만, 그림 수치는 텍스트에만 있어서 확인하지 못했습니다.
2. **D2와 D3 (초기화와 정규화):** 정책의 출발점과 손실 스케일을 바꿉니다. 실험 비교(예: 제거 실험)에서 논문과 코드가 같은 조건인지 확인하려면 둘 다 기록해야 합니다.
3. **D1 (질의 임베딩):** 논문 부록 C의 설명과 다릅니다. 코드를 따르면 학습 질의이고, 논문 문장을 따르면 고정 사인입니다. 본문 실험 결과의 해석에는 큰 영향이 없을 것으로 보지만 측정하지는 않았습니다.
4. **D7 (에폭 수):** README 예시의 2000 에폭은 시뮬레이션용입니다. 실로봇 에폭은 논문과 코드 어디에도 없습니다. 실로봇 재현 비용을 계산할 때 이 값이 필요합니다.

## 4. 구현 가능성 (코드 기준)

- **CUDA 하드코딩:** `imitate_episodes.py`(147, 170, 219, 221, 242, 258, 318, 332)와 `detr/main.py`(78, 101)에 `.cuda()`가 있습니다. 이 맥(Apple M4 Pro, MPS, CUDA 없음)에서 돌리려면 장치를 `mps`로 바꾸는 수정이 필요하고, 그 뒤 동작은 확인해야 합니다. 추정 노력 **0.5–1일** (수정 자체, 측정 전).
- **시뮬레이션 평가:** MuJoCo 기반 `sim_env.py`와 장면 XML이 저장소에 있습니다. MuJoCo 설치가 필요하고, 설치 전에 설치 목록을 먼저 승인받아야 합니다.
- **실로봇:** ROS 1 Noetic, Interbotix 패키지, Dynamixel 모터, 웹캠 4대가 필요합니다. 이 맥에서는 불가능합니다.
- **학습 비용:** 논문 기준 RTX 2080 Ti에서 약 5시간, 모델 약 80M 파라미터 (§IV-C, 본문 수치. 이 세션에서 파라미터 수는 세지 않음). 시뮬 데이터 생성과 학습은 이 분석에서 실행하지 않았습니다.

## 5. 검증하지 않은 항목

- 코드를 실행하지 않아서 D2(ImageNet 가중치 다운로드 여부), D5(실제 옵티마이저 설정 적용 여부)는 읽기로만 판단했습니다.
- 80M 파라미터 수, 시간 앙상블의 실제 효과(그림 8b), 질의 임베딩 차이의 효과는 확인하지 못했습니다.
- 논문 v1 PDF와 저장소 HEAD(2024년 중반 이후 커밋이 아님)의 대응은 확인했지만, 논문 저자가 다른 버전의 코드를 쓴 가능성은 배제하지 못했습니다.

# 04. 코드 검증: 논문 ↔ 구현 대조, 수치 재계산, 구현 가능성

- **대상:** `zhangganlin/BA-T` @ `406d791d21d7ae60ad58b0d2a8892735709ac83f` (2026-10-21, "ba-t codebase init", shallow clone). HF `zhangganlin/BA-T` 메타데이터(커밋 `ac53e30`, 2026-10-22)는 API로만 확인했고 가중치는 받지 않았습니다.
- **범위:** torch 설치, 가중치 다운로드, 실행 없이 코드 읽기와 CPU 재계산만 했습니다. 저장소는 **추론 전용**입니다. 학습 코드, 평가 스크립트, 평가 쌍 목록이 없어서 표 1–6을 결과 파일로 재계산할 수는 없습니다.

## 1. 모듈 매핑

| 논문 | 구현 위치 | 일치 여부 |
|---|---|---|
| ViT 인코더, ViSTA 사전학습 (부록 A) | `bat_model.py: BAT` — ViT-L 24층, 1024차원, 16 heads, 패치 16, RoPE2D(base 100) | ✅ 해상도는 논문에 없고 코드는 224×224 (D7) |
| 토큰 차원 D = 768 | `dec_embed_dim=768`, `appearance_embed: Linear(1024→768)` | ✅ |
| 학습되는 초기 카메라 토큰 (§3.2) | `init_pose_token` (1×1×768), 간선마다 복제 | ✅ |
| 층 하나를 K번 공유 반복 (식 4) | `decode()`의 `for k in range(num_refine_iter): self.ba_layer(...)` | ✅ 기본 K는 다름 (D6) |
| 카메라 조건 변환 π_c, adaLN (§3.3, 부록 A) | `ConditionModulationBlock` (ModLN = LN + SiLU + Linear(768→1536)이 shift·scale 생성, 내부에 self-attn + MLP) | ✅ |
| soft 대응 매칭 8 heads (식 5, 부록 A) | `TokenCorrRes(num_heads=8)`: q·k attention으로 상대 시점 v의 가중합 | ✅ 좌표계 방향은 D2 |
| 성질 f = 선형 투영, 잔차 = 차이의 MLP (식 6–7) | `TokenCorrRes`: 두 시점에 같은 `projv`, `LN(v_a) − LN(attn·v_b)` → Linear-GELU-Linear | ⚠️ 확장비 1 (D4), 부호·방향 (D2) |
| 카메라 갱신: 카메라 토큰 → 잔차 N개 cross-attention, 12 heads + MLP (식 8) | `camera_attn = CrossAttnLayer(12 heads, use_mlp=True)` | ✅ 순서 (D1), 덧셈 아님 (D5) |
| geometry 갱신: 시점 간 cross-attention, 12 heads + MLP (식 9) | `cross_attn_layers[0]` (이웃을 `_transform`으로 자기 공간에 옮긴 뒤) | ✅ 이전 카메라 사용 (D1) |
| 시점 내 self-attention 3개, 각각 MLP (부록 A) | `attn`, `attn2`, `attn3` 직렬, **MLP 없음** | ❌ (D3) |
| 역방향 카메라 = 작은 MLP (부록 C) | `camera_inverse = Mlp(768→3072→768)`. 포즈 헤드도 `pose_ji = head(camera_inverse(t))` | ✅ 두 방향 포즈를 따로 예측해 근사적으로만 역행렬 관계 (README도 명시) |
| 헤드: point map(시점별 좌표계) + 상대 포즈 | `LinearPts3d`(DUSt3R 선형 헤드, 3+신뢰도), `PoseHead_small`(768→512×3, 회전 9D·이동 3·신뢰도 sigmoid) | ✅ |
| 반복마다 디코딩 (식 11의 전제) | `decode(all_iters=True)`가 반복마다 헤드 실행 | ✅ 손실 코드는 없음 (D12) |
| 다시점: 간선마다 카메라 토큰, 시점마다 이웃 전체에 cross-attention (부록 G, 알고리즘 2) | `forward(gs, poss, ts, graph)`: `others[v]`에 이웃 토큰을 변환해 이어 붙임, 잔차·카메라는 간선별 배치 | ✅ 그래프 생성·PGO는 코드에만 (D11) |
| 디코더 38M (표 1, 5) | `BundleAdjustmentLayer` 38.40M (`param_check.py`) | ✅ 코드 기준. 부록 A 서술로는 52.6–56.1M (D3, D4) |

## 2. 불일치와 논문에 적히지 않은 구현 요소

| # | 내용 | 영향 |
|---|---|---|
| D1 | **한 층 안의 갱신 순서가 논문과 반대입니다.** 논문(§3.3 "we update the camera token first", 식 8–9, 알고리즘 1)은 c^(k+1) = c^(k) + CrossAttn(c^(k), r(g^(k), c^(k)))를 먼저 하고, 갱신된 c^(k+1)로 geometry를 고칩니다. 코드(`ba_blocks.py: BundleAdjustmentLayer.forward`)는 먼저 `ts_inv = camera_inverse(ts)`를 **이전 카메라 토큰**으로 만들고, 이웃 시점을 그것으로 변환해 geometry를 갱신합니다. 그다음 **새 geometry**와 같은 `ts_inv`로 잔차를 만들어 카메라를 갱신합니다. 반복을 펼치면 논문은 C G C G …, 코드는 G C G C …입니다. 코드는 첫 geometry 갱신을 초기 토큰(F.2에 따르면 거의 항등 변환)으로 하고, 마지막 카메라 갱신 뒤에는 geometry 갱신이 없습니다. | 논문의 "Schur 순서를 따른다"는 핵심 비유가 공개 구현과 맞지 않습니다. 공개 체크포인트를 쓰려면 코드 순서를 따라야 합니다. 고전 BA toy에서는 순서가 고정점을 바꾸지 않지만 수렴 속도는 크게 바뀝니다(`03` §3). 학습 모델에서의 영향은 측정되지 않았습니다. |
| D2 | **잔차를 계산하는 좌표계가 반대입니다.** 논문(식 5–7, Fig. 4 "Residuals are computed in the latent space of view b", 알고리즘 1)은 g_a를 c_{a→b}로 b 공간에 옮겨 질의로 쓰고 g_b와 매칭합니다. 코드는 `token_corr_res(gs_new[i], _transform(gs_new[j], ts_inv[e]))`로 **변환하지 않은 g_a가 질의**, `camera_inverse`로 a 공간에 옮긴 g_b가 key·value입니다. 잔차도 f(a) − f(매칭된 b)라 부호가 논문(f(b^q) − f(π(a)))과 반대입니다. | 학습되는 함수라 부호는 의미가 없고, 방향도 대칭이라 성능 차이는 작을 것으로 봅니다. 하지만 Fig. 4의 해석("b 공간 잔차")과 실제 연산이 다르고, 카메라 토큰 자체가 아니라 `camera_inverse` 출력이 잔차에 들어갑니다. |
| D3 | **시점 내 self-attention 3개에 MLP가 없고, 층 사이 잔차 연결도 없습니다.** 부록 A는 "three intra-view self-attention layers (each with 12 heads and a 2-layer MLP with expansion ratio 4)"라고 씁니다. 코드는 `g_self = attn3(LN(attn2(LN(attn(LN(g_update))))))`로 attention만 직렬로 잇고, 마지막에 `geo_norm(g_update + g_self + g)`로 한 번만 더합니다. | MLP가 있었다면 층이 52.6M입니다. "38M"은 코드 구조로만 성립합니다(`03` §2). 부록 A대로 재구현하면 체크포인트를 읽을 수 없고 크기 비교도 달라집니다. |
| D4 | **잔차 MLP의 확장비가 4가 아니라 1입니다.** 부록 A "2-layer MLP with expansion ratio 4". 코드 `TokenCorrRes.proj = Linear(768,768) – GELU – Linear(768,768)`. | D3과 합치면 부록 A 서술의 층은 56.1M입니다. 주석 처리된 코드(`torch.cat([va, va_in_b, (va - va_in_b).abs()])`)를 보면 잔차 형태를 여러 번 바꿔 본 흔적이 있습니다. |
| D5 | **갱신이 순수 덧셈이 아닙니다.** 식 4는 T^(k+1) = T^(k) + ΔT입니다. 코드는 g^(k+1) = LN(g + g_update + g_self), c^(k+1) = LN(c + Δc)로 **덧셈 뒤 LayerNorm**(post-LN)을 합니다. 또 `CrossAttnLayer`에서 MLP 주변 잔차 연결이 주석 처리되어(`# x = x + self.drop_path(self.mlp(...))`), 출력이 `MLP(LN(x + CrossAttn))` 그 자체입니다. | 토큰 노름이 매 반복 정규화됩니다. README에 따르면 공개 가중치는 4–24회 반복으로 학습되었는데, 반복을 늘려도 발산하지 않는 데 이 정규화가 기여할 것으로 봅니다(추정). 논문에는 없는 설계입니다. |
| D6 | **기본 반복 수와 기본 체크포인트가 논문과 다릅니다.** 논문은 학습 K = 5, 평가 K = 4입니다. 코드 `decode()`, `forward()`, `demo.py`의 기본값은 `num_refine_iter=8`입니다. 기본 체크포인트 `bat.pth`는 README에 따르면 "논문 모델을 더 많은 반복(4–24)과 다시점 그래프(2–8개 시점)로 추가 미세조정"한 것이고, 논문 수치는 `bat_paper.pth`입니다. | README 기본 설정으로 돌린 결과를 논문 표와 비교하면 안 됩니다. 논문 재현에는 `--ckpt pretrains/bat_paper.pth --num_refine_iter 4`가 필요합니다. |
| D7 | 입력 해상도가 논문에 없습니다. 코드는 224×224, 패치 16이라 시점당 토큰 196개이고, 전처리는 중심 크롭 + Lanczos 축소 + 중심 크롭입니다(`data.py`). 주점은 이미지 중심으로 가정합니다. | 표 5의 "same input images with the same resolution"이 모든 비교 모델을 224로 돌렸다는 뜻인지 알 수 없습니다. VGGT(518) 등은 기본 해상도가 더 높습니다. |
| D8 | 인코더 출력에 `enc_norm`을 적용하지 않습니다(코드 주석 "this is how the model was trained"). `enc_norm`은 체크포인트 호환을 위해 남겨 둔 미사용 모듈입니다. | 재구현할 때 흔히 넣는 최종 LN을 넣으면 공개 가중치와 맞지 않습니다. |
| D9 | 카메라 갱신 cross-attention에서 카메라 토큰에 RoPE 위치 (−1, −1)을 줍니다(`cam_pos = -torch.ones(...)`). 잔차 토큰은 view a의 패치 위치를 씁니다. | 논문에 없는 세부입니다. 카메라 토큰이 모든 패치에서 "격자 밖 한 점"으로 보이게 하는 장치입니다. |
| D10 | 알려진 포즈(회전 벡터, 이동, 초점거리, Fourier 특징)를 카메라 토큰으로 바꾸는 `CameraTokenEncoder`(0.43M)가 체크포인트에 들어 있지만 추론에는 쓰지 않습니다. | 논문에 언급이 없습니다. 학습 중 알려진 포즈를 넣는 변형(포즈 조건 복원)을 시도했을 가능성이 있습니다(추정). |
| D11 | **다시점 경로가 논문보다 많습니다.** 코드는 ① 인코더 특징의 상호 최근접 패치 비율로 view graph를 자동 생성(`graph.py`: 유사도 ≥ 0.3, 이웃 최대 7개, 최대 신장 트리 추가)하고, ② 간선별 포즈 신뢰도로 간선을 고른 뒤 신뢰도 제곱 가중 강건 손실로 **pose graph 최적화**(`pose_graph.py`)를 합니다. 논문 부록 G와 표 6은 완전 그래프와 알고리즘 2뿐이고 PGO가 없습니다. | README의 다시점 결과(8개 시점 0.4 s, 64개 3 s, RTX 4070)는 논문 표 6과 다른 파이프라인입니다. 반대로, 로봇 응용에서 쓸 다시점 경로는 코드 쪽이 더 완성되어 있습니다. |
| D12 | **학습 코드가 없습니다.** 손실(식 10)의 λ 값, 사이클 일관성 항, 반복 감독(식 11), 데이터 로더와 샘플링, 4-view 미세조정 설정이 모두 미공개입니다. λ 값은 논문에도 없습니다. 체크리스트 Q5 답은 [No] "code will be released upon acceptance"이고, 공개된 것은 추론 코드입니다. | 학습 재현과 ablation 재현은 불가능합니다. 재구현하려면 손실 가중치를 새로 찾아야 합니다. |
| D13 | 실행 환경: import 시점에 `torch.backends.cuda.matmul.allow_tf32 = True`(공개 모델이 TF32로 학습됨). xformers는 CUDA 텐서에서만 쓰고 CPU에서는 torch SDPA로 대체합니다. curope(CUDA RoPE 커널)가 없으면 순수 PyTorch RoPE로 대체하고 경고를 출력합니다. | CPU·MPS 경로가 코드상 열려 있습니다(§4). TF32 설정은 CUDA에만 영향이 있습니다. |

## 3. 수치 재계산 (`analysis/param_check.py`, `analysis/table_check.py`, `analysis/ba_toy.py`)

| # | 주장 (출처) | 재계산 | 판정 |
|---|---|---|---|
| C1 | 표 3 사다리 비율 (§4.3): 25/15/27%, 76/54/75%, 68/46% | 25.3/14.8/26.9%, 76.0/53.5/75.0%, 68.4/46.5% | ✅ 9개 모두 일치. 지표 선택에 따라 32–80%로 달라짐(`03` §6) |
| C2 | Corr. 감소 ViSTA† 27%, BA-T 44% (§4.2, 프로젝트 페이지) | 27.4%, 43.5% | ✅ |
| C3 | DUSt3R 디코더의 16% (초록, 프로젝트 페이지 "38M vs 227M") | 38 / 227 = 16.7% | ✅ (반올림하면 17%) |
| C4 | 대형 multi-view 모델의 5–6% (v2 §1) | VGGT 6.3%, DA3 5.0%, **MapAnything 22.2%** | ⚠️ MapAnything 제외 시만 |
| C5 | 최대 6배 큰 2-view 모델 (v2 §1) | 227 / 38 = 5.97배 | ✅ |
| C6 | DROID-SLAM 대비 약 5배 빠르고 메모리 약 45% 절감 (부록 D) | 5.2배, 44.4% | ✅ |
| C7 | MapAnything 메모리의 39% (부록 G) | 1.35 / 3.42 = 39.5% | ✅ |
| C8 | 표 4와 표 2의 같은 설정 (7Scenes, K = 4) | 회전 0.68 = 0.683, 이동 0.42 = 0.422, δ1.25 0.98 ≈ 0.976, **Rel. 0.20 ≠ 0.048** | ❌ Rel.만 4.2배 차이 |
| C9 | BA-T 최대 메모리 | 표 5 1.32 GB, 표 4·6 1.35 GB | ⚠️ 사소한 불일치 |
| C10 | ViSTA† 3층 = 38M (부록 B) | DecoderBlock(768) 9.45M/층, 12층 113.4M(ViSTA 113M과 일치), **3층 28.4M**, 4층 37.8M | ⚠️ 산술 불일치 |
| C11 | 최대 메모리가 가장 낮음 (부록 E) | 전체 344.4M = fp32 1.283 GiB. 1.32 GiB면 활성값 약 0.04 GiB | ✅ 다만 비교의 실체는 전체 파라미터 수 |
| C12 | 표 6 "comparable accuracy" | VGGT가 이동·회전·Rel.·δ1.25 네 지표 모두 좋음, BA-T는 MapAnything보다 네 지표 모두 좋음 | ⚠️ VGGT 기준으로는 너그러운 표현 |
| C13 | 2-view 모델 중 포즈 최고 (표 1) | 7Scenes 세 지표 최고. BundleFusion은 @5 동률(DUSt3R, MASt3R와 0.16), @10·@20 최고. geometry는 ViSTA가 Chamf.·Comp.에서 더 좋음 | ✅ (geometry는 "competitive") |
| C14 | 반복 비용 (표 5) | 반복당 3.22 ms, 기저 18.26 ms. K = 8 외삽 44 ms (README RTX 4070 K = 8 약 50 ms와 같은 자릿수) | 참고 |
| C15 | "이득은 반복이 아니라 구조" (§4.3) | BA-T k=1이 이미 ViSTA† k=4보다 회전 +0.159, 이동 +0.171. k=4 우위의 83% / 63%가 k=1에 있음 | ✅ 표 2로도 지지됨 |
| C16 | 가중치 파일 크기 | 344.4M × 4 B = 1.378 GB = HF 파일 크기 | ✅ 파라미터 셈 교차 검증 |
| V1 | Schur 보완(식 3) = 전체 정규방정식 해 | 최대 차이 1.4×10⁻¹⁴, C 비대각 0 | ✅ (`03` §1) |
| V2 | 카메라 먼저 / geometry 먼저가 같은 해로 가는가 | 200회 뒤 joint 해와 회전 차이 ≤ 1.7×10⁻⁴° / ≤ 0.015° | ✅ 고정점 같음, 속도는 다름(`03` §3) |
| V3 | 포즈가 geometry보다 잘 좋아진다 (F.2) | 고전 BA에서 N 16배 → 회전 오차 5.1배 감소, 깊이 오차 1.9배 감소 | ✅ 같은 방향(`03` §4) |

## 4. 구현 가능성 판단

### 4.1 공개 체크포인트로 추론 (논문 표 일부 근사)

| 항목 | 요구 사항 | 이 맥 (Apple M4 Pro, 통합 메모리 24 GiB, MPS, CUDA 없음) |
|---|---|---|
| 소프트웨어 | Python 3.11, torch 2.5.1(테스트 버전), numpy, pillow. xformers·curope·open3d는 선택 | torch CPU/MPS 휠 설치가 필요합니다(설치 승인 필요). xformers와 curope는 CUDA 전용이지만 코드가 순수 PyTorch로 대체합니다(D13) |
| 가중치 | `bat_paper.pth`(논문 수치) 또는 `bat.pth`, 각 1.378 GB | 다운로드 승인 필요. 메모리는 가중치 1.28 GiB + 활성값 소량이라 충분합니다 |
| 실행 경로 | `demo.py --device` 기본값이 CUDA 없으면 `cpu`. README "The model also runs on the CPU (slowly)" | CPU는 공식 지원입니다. MPS는 `--device mps`로 가능해 보이지만 미검증입니다(RoPE 대체 경로와 SDPA의 MPS 동작 확인 필요) |
| 속도 | K = 8에서 약 450 GFLOP/쌍(추정, `03` §5) | CPU 쌍당 약 0.5–2초로 추정(미측정) |
| 평가 데이터 | 7Scenes, BundleFusion(공개) | 받을 수 있습니다. 다만 **평가 쌍 목록과 평가 스크립트가 없어** 표 1–2를 정확히 재현할 수는 없습니다. 쌍 샘플링과 지표 정의(특히 Rel., C8 불일치)를 직접 정해야 하므로 근사 재현만 가능합니다 |
| 라이선스 | 가중치 CC BY-NC-SA 4.0, Naver 파생 코드 CC BY-NC-SA 4.0, 나머지 Apache-2.0 | 연구·비상업 용도만 가능합니다 |

**판단:** **추론은 이 맥에서 가능합니다.** CPU 경로가 공식으로 열려 있고, 가중치 1.4 GB와 메모리 2 GB 미만이면 됩니다. 필요한 것은 torch 설치와 가중치 다운로드 승인입니다. 지금은 재현 보류 상태라 실행하지 않았습니다. 논문 표 재현은 평가 자료가 없어 **근사만** 가능합니다.

### 4.2 학습 재현

| 항목 | 요구 사항 | 비고 |
|---|---|---|
| 코드 | 학습 코드 없음 (D12) | 손실 가중치 λ, 데이터 로더, 반복 감독을 직접 구현해야 합니다 |
| 계산 | H100 8장 × 2일 ≈ 384 H100-h, 다시점 미세조정 12시간 추가 | CUDA 필수. 우리 규칙상 GPU 재현은 하지 않습니다 |
| 데이터 | ScanNet, ScanNet++, ARKitScenes, CO3D, ASE, Replica (표 7, 모두 비상업, 일부 신청 필요) | 수 TB 규모로 알려져 있습니다(내 사전 지식, 미확인) |
| 인코더 초기화 | ViSTA 사전학습 ViT | ViSTA-SLAM 저장소에서 받을 수 있는지는 확인하지 않았습니다 |

**판단:** **학습 재현은 불가능합니다.** 코드, 계산, 데이터가 모두 막혀 있고, 이 맥에서는 계산만으로도 불가능합니다.

### 4.3 핵심 아이디어를 다시 구현하는 경우

| 구성 요소 | 난이도 | 메모 |
|---|---|---|
| BA-T 층 (조건 변환, 매칭·잔차, 카메라·geometry 갱신) | 낮음 | `ba_blocks.py`가 약 350줄이고 DUSt3R 블록에만 의존합니다. 체크포인트와 맞추려면 논문이 아니라 **코드의 순서와 구조**(D1–D5, D8, D9)를 따라야 합니다 |
| 반복 감독 학습 루프 | 낮음 | 반복마다 헤드를 돌리고 α^{K−k}로 가중합 |
| 손실 (point map, 포즈 + 사이클, 기하 일관성) | 중간 | DUSt3R·ViSTA 손실을 참고할 수 있지만 λ 값은 새로 찾아야 합니다 |
| 작은 규모 검증 (합성 장면, 저해상도, 작은 인코더) | 중간 | CPU·MPS에서 "명시적 잔차 vs 직접 cross-attention"(표 3 (b) vs (h))만 비교하는 실험은 가능합니다. 대형 사전학습 인코더 없이도 의미 있는지는 미지수입니다 |
| 논문 규모 학습 | 높음 | §4.2와 같음 |

**요약:** 이 논문은 **쓰기는 쉽고 다시 만들기는 어렵습니다.** 공개 체크포인트로 두 장 또는 여러 장의 상대 포즈와 point map을 얻는 것은 이 맥 CPU에서 바로 할 수 있습니다(설치·다운로드 승인 후). 반면 학습 재현은 불가능하고, 논문 서술만으로 재구현하면 공개 모델과 다른 구조(D1–D5)가 됩니다. 아이디어를 가져가려면 공개 코드의 `BundleAdjustmentLayer`를 기준으로 삼는 것이 맞습니다.

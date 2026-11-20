# 04. 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `tsinghua-fib-lab/WorldArena`, 커밋 `2da2ae253b` (2026-07-22). 로컬 위치 `repo/` (shallow clone).
- 방법: **정적 읽기만 했다.** 코드를 실행하거나 import하거나 패키지를 설치하지 않았다. 외부 가중치, 데이터, 리더보드도 받지 않았다. `grep`과 `sed`로 해당 함수만 읽었다.
- 확인 범위: 16개 지표 중 게이트와 정규화가 걸린 지표, 집계, 정책 평가 상관, VLM 판정 프롬프트. 나머지 지표(예: MUSIQ, LAION, DINO 특징)의 세부 구현은 파일명과 호출만 확인했다.
- 1.0 코드만 본다. 2.0 저장소(`WorldArena2/WorldArena-2.0`, MIT)는 범위 밖이다.

## 1. 지표별 대조

| 지표 (논문 식) | 코드 위치 | 판정 | 비고 |
|---|---|---|---|
| Image Quality (식 1, MUSIQ) | `WorldArena/imaging_quality.py` | 파일 확인 | 세부 미확인 |
| Aesthetic (식 2, LAION) | `WorldArena/aesthetic_quality.py` | 파일 확인 | 세부 미확인 |
| JEPA Similarity (식 3–4, α = 40) | `run_evaluation_JEPA.sh` → `JEDi/` | **미확인** | MMD 구현은 외부 패키지 `videojedi` 휠에 있고, 저장소에 `alpha`/`mmd` 문자열이 없다 (D8) |
| Dynamic Degree (식 5, 6/256·min(H,W), 시그모이드) | `WorldArena/dynamic_degree.py` | **일치 (α는 논문에 값 없음)** | `thres = 6.0·min(H,W)/256`, `α = 5.0` 기본값 (D3) |
| Flow Score (식 6, RAFT 평균 크기) | `WorldArena/flow_score.py` | 일치 | 쌍마다 RAFT 광류 크기 평균 후 평균 |
| Motion Smoothness (식 7, SSIM·ln(1+diff)) | `WorldArena/motion_smoothness_metrics.py` | **일치** | `ssim_val * np.log1p(raw_diff)` 평균, 논문 식 (7)과 같다 |
| Subject Consistency (식 8–9) | `WorldArena/subject_consistency.py` | **차이** | 게이트 규칙이 다르다 (D1) |
| Background Consistency (식 10–11) | `WorldArena/background_consistency.py` | **차이** | D1과 같은 규칙 |
| Photometric (식 12–13) | `WorldArena/flow_aepe_metrics.py` | **역수는 일치, 게이트는 차이** | `score = 1/score` 후 D1 규칙 |
| Trajectory Accuracy (식 14–16, NDTW) | `WorldArena/trajectory_accuracy.py` | **차이** | 단일 궤적 DTW가 아니라 `max_distance_index` 한 점의 DTW를 쓰고, 경로 길이로 나눈다 (D4) |
| Depth Accuracy (식 17, median 정렬) | `WorldArena/depth_accuracy.py` | 일치 (방향) | `scale = median_gt / median_gen`을 생성 깊이에 곱한다. 논문 식의 분수 방향은 PDF 텍스트가 깨져 확정 못함 |
| Interaction, Perspectivity, Instruction (A.10, VLM) | `video_quality/VLM_judge.py` | **차이 (키 이름)** | 프롬프트 본문 키 표기가 논문과 다르다 (D7) |
| Semantic Alignment (식 18) | `WorldArena/semantic_alignment.py` | 파일 확인 | 세부 미확인 |
| Action Following (식 19, 1−cos) | `WorldArena/action_following.py` | **일치** | 쌍별 `1 − cos`를 평균. 정규화는 없다 (논문도 "이미 정규화"라고 씀) |
| 정규화 경계 (식 20–21, Table 6) | `WorldArena/__init__.py` `_EMPIRICAL_BOUNDS` | **값 일치** | 다섯 쌍의 min·max가 Table 6과 모두 같다. 유래는 논문 본문과 다르다 (D2) |
| EWMScore 평균 | **저장소에 없음** | 미확인 | `aggregate_results.py`는 CSV만 만든다 (D5) |
| 정책 평가 Pearson (§4.2.2) | `embodied_task/policy_eval_release_bundle/calculate_policy_pearson_r.py` | **일치 (계산식)** | 단순 Pearson. 기본 시뮬레이터 점수 5개가 그림 4 x축과 같다 (D9) |
| 정책 성공 판정 (부록 B) | `embodied_task/policy_eval_release_bundle/vlm_policy_evaluator.py` | **일치 (의도)** | 팔 선택, 완료 상태, 행동 의도 세 기준이 같다. 판정 VLM의 기본값은 `Qwen/Qwen3-VL-32B-Instruct`다 (D9) |
| 데이터 분할 (2000/500) | `README.md` | 일치 | 과제당 40개 학습, 10개 시험. 50 과제면 2000/500과 같다 |

## 2. 차이 목록

**D1. 정적 영상 게이트 규칙 (영향 큼).** 논문은 Subject, Background, Photometric에 `min(1, Sdyn/γ)`를 곱한다 (식 9, 11, 13). γ 값은 없다. 코드는 세 파일 모두 같은 상수를 쓴다.

```
if dynamic_score <= 0.1213:
    sim_per_images = sim_per_images * dynamic_score
```

(`subject_consistency.py` 77행, `background_consistency.py` 66행, `flow_aepe_metrics.py` 204행.) 차이는 두 가지다. (a) 임계 0.1213은 논문에 없다. (b) 곱하는 값이 `Sdyn/γ`가 아니라 `Sdyn`이고, 임계 바깥에서는 1이다. 그래서 **임계에서 점수가 불연속**이 된다 (`03` §6.1). 식 (9)가 "연속적인 페널티"라는 논문의 설명과 어긋난다. 영향은 Subject, Background, Photometric 세 지표와 EWMScore 전체에 걸린다.

**D2. 정규화 경계의 유래.** 논문 A.17은 경계를 "8개 모델 영상의 99/1 백분위"에서 구했다고 쓴다. 평가 대상은 14개 모델이다. 코드는 Table 6 값을 **상수로 박아 둔다**. 백분위 계산 코드와 8개 모델 목록은 저장소에 없다. 값이 같으므로 표 값 재현에는 문제가 없지만, 새 모델을 넣었을 때 경계가 어떻게 바뀌는지는 알 수 없다. 또 `flow_aepe_metrics.py` 193–196행에는 쓰지 않는 상수 `empirical_max = 3.4536`과 주석 처리된 클리핑이 있다. 현재 Table 6의 Flow 상한(8.9414)과 다르다. 죽은 코드이므로 결과에는 영향이 없다.

**D3. 동적 정도 파라미터.** 식 (5)의 τ는 PDF 텍스트에서 `6/256 · min(H,W)`로 읽힌다(분수가 깨져 있었다). 코드의 `thres = 6.0·min(H,W)/256`과 맞는다. 시그모이드 기울기 α는 논문에 값이 없고 코드는 5.0을 쓴다. 논문에 값이 없어서 재현 시 기본값이 달라질 수 있다.

**D4. 궤적 정확도 (NDTW).** 논문 식 (15)는 한 궤적의 DTW를 `(1/|R|)·min_π √Σ‖r−p‖²`로 쓴다. 코드(`trajectory_accuracy.py` 186–200행)는 `max_distance_index`로 고른 **한 표본점**의 궤적에만 `fastdtw`를 쓰고, 경로 길이로 나눈 뒤 `1/di`를 쓴다. 표본점 선택(`rotating_calipers`, `farthest_distance`)과 점 개수 `k`는 논문에 없다. 점수 체계(1/거리)는 같다.

**D5. EWMScore 집계가 저장소에 없다.** 논문 §3.4의 "16개 정규화 지표의 산술 평균 ×100"을 재현할 코드가 없다. `aggregate_results.py`의 `COLUMN_ORDER`는 16개 지표 열을 정의하지만 평균은 하지 않는다. 표 2·3 값으로 EWMScore를 재현한 것은 `03` §1에서 했다(최대 차이 0.02).

**D6. 표 4와 5의 TesserAct 행.** 두 표 모두 TesserAct의 과제 1은 1%, 과제 2는 35%다. 저장소에는 표 값이 없다. 같은 값이 2.0 프로젝트 페이지의 RoboTwin 행에도 있다. 복사 오류인지 실제 값인지 이 저장소로는 확정할 수 없다. 확인하려면 저자에게 원자료를 요청해야 한다.

**D7. VLM 판정 JSON 키.** 논문 부록 A.10 프롬프트의 출력 형식은 `'Interaction Quality'`, `'Perspectivity'`, `'Instruction Following'`(공백)으로 적었다. 같은 프롬프트의 예시 JSON은 `"Interaction_Quality"`, `"Instruction_Following"`(밑줄)을 쓴다. 코드(`VLM_judge.py` 96, 123–141, 206행)는 밑줄 키만 쓴다. 코드가 맞다. 그러나 논문 프롬프트를 그대로 옮기면 파싱이 깨질 수 있다. 또 `num_frames=16`(기본값)이지만 논문은 "샘플 프레임 수"를 적지 않았다.

**D8. JEPA Similarity 구현 위치.** 저장소의 `video_quality/JEDi/`에는 MMD 커널이나 α 값이 없다. `dist/`의 휠과 tarball(`videojedi-0.1.3`)에 구현이 있을 수 있으나, 이 작업에서는 풀지 않았다(외부 패키지를 풀어 보는 것도 신뢰하지 않는 코드를 여는 일이다). **미확인.**

**D9. 정책 평가 판정기와 시뮬레이터 점수.** 논문은 정책 판정 VLM의 크기를 적지 않았다. 저장소 기본값은 Qwen3-VL-32B다. 그림 4의 x축 값(28.6, 34.58, 37.78, 43.52, 46.8)은 `DEFAULT_SIMULATOR_SCORES`와 같다. 세계 모델 쪽 점수(y축)를 만드는 결과 파일은 저장소에 없다. 그림에서 읽은 근사로 r을 재현했다(`03` §4).

**D10. 논문 내부 표기 (코드와 무관, 참고).** 본문 §3.1.2는 움직임 지표를 "세 개"라고 쓰면서 네 개를 나열하고, Trajectory를 Physics 표에 둔다. 그림 11의 Action Following 예시 값(12.01, 0.47)은 표 3의 범위(0.008–0.142)와 맞지 않는다. 사례 그림의 단위가 다른 것으로 보인다.

## 3. 본문 주장과 코드 (요약)

- 정규화 경계는 표 6과 코드가 같다 → 수치 재현 가능.
- 정적 영상 페널티는 본문 식과 코드가 다르다 → 같은 이름의 지표라도 값이 달라질 수 있다.
- EWMScore 평균과 사람 평가 상관은 이 저장소로 재현할 수 없다.
- 정책 평가 상관은 계산식과 시뮬레이터 점수가 맞지만, 세계 모델 쪽 결과가 없어 r 자체는 재현할 수 없다.

## 4. 구현 가능성 판정

**판정: 조건부 (분석과 재계산, 토이까지 이 맥에서 완료. 지표 전체 재구현과 평가 재현은 GPU와 로봇 시뮬레이터가 필요하다.)**

이 맥은 Apple M4 Pro, 통합 메모리 약 24 GiB, MPS 사용, CUDA 없음이다 (메모리 `mac-apple-gpu`).

| 필요 항목 | 이 맥 | 근거 |
|---|---|---|
| 재계산 (표 2·3 → EWMScore, 통계, 상관) | **가능, 완료** | CPU, numpy, scipy (`03`) |
| 게이트 규칙과 임계 민감도 toy | **가능, 완료** | CPU (`toy_dynamic_gate.py`) |
| 정규화 경계와 Depth·Motion 식 재구현 (영상 몇 개) | **가능 (조건부)** | 식은 CPU로 짤 수 있다. RAFT, Depth-Anything은 MPS나 CPU로 느리게 돈다. 가중치 다운로드 승인이 필요하다 |
| MUSIQ, LAION, DINO, CLIP 지표 (영상 수백 개) | 조건부 | 모델 수백 MB–수 GB. MPS에서 동작은 하지만 시간은 미측정 |
| Qwen3-VL-8B 판정 | 조건부 | 파라미터 8B, bf16 약 16 GB로 추정 (실측 아님). 32B(기본값)는 메모리에 안 들어간다 |
| 14개 세계 모델 영상 생성 | **불가** | 각 모델이 수 B 파라미터 이상이고 CUDA 전제. Veo 3.1·Wan 2.6은 API 필요 |
| π0.5 정책 학습 (데이터 엔진, 정책 평가자 5개) | **불가** | GPU 학습 |
| RoboTwin 2.0 시뮬레이터 롤아웃 (행동 계획자) | **불가 (확인 필요)** | 시뮬레이터 설치 환경을 확인하지 않았다. Linux GPU 렌더링 전제일 가능성이 높다 |
| 사람 평가 재현 | **불가** | 원자료가 공개되지 않았다 |

**노력 추정 (거친 추정, 측정 아님):** 지표 파이프라인을 CUDA 없이 맞추는 데 1–2주, 전체 재현(세계 모델 14개 생성, π0.5 학습, 시뮬레이터 평가)은 GPU 자원에 따라 수 주에서 수 개월이다. 이 맥에서 할 수 있는 것은 "재계산과 지표 일부 재구현"까지다.

## 5. 확인하지 못한 것

- JEPA Similarity의 MMD 구현과 α = 40 (D8).
- MUSIQ, LAION, DINO, CLIP, Semantic Alignment, Depth-Anything 호출 세부.
- 정규화 경계의 8개 모델 목록과 백분위 계산 (D2).
- 세계 모델 쪽 정책 평가 결과 파일.
- 2.0 저장소 (범위 밖).

# 01. 논문 분석 및 리버스 엔지니어링

- 대상: Wang, Lin, Huang, Lin, Zhang, Wang, "GPT-6-Astra Understands Touch and Outperform Vision–Tactile–Language–Action Models", arXiv:2610.28457v1 [cs.RO], 2026-10-20 (Sun Yat-sen University, X-Era AI Lab, The Hong Kong Polytechnic University). 프로젝트 페이지: https://gpt-as-tactile-policy.github.io/ (이 문서에서는 접속하지 않았으므로 내용은 미확인입니다).
- 근거 자료: `paper/paper.txt` 전문과 `paper/page-01.png`, `page-03.png`, `page-10.png`, `page-11.png` 렌더 이미지를 직접 확인했습니다.
- 표기 규칙: 원문 위치는 `[§3.2]`, `[Tab.2]`, `[Fig.2]`처럼 표기합니다. 원문에 없는 내용을 추정한 경우에는 반드시 "추정"이라고 쓰고, 확인할 수 없는 사실은 "미확인"으로 표시합니다.
- 인용된 선행연구의 서지 정보는 이 논문의 참고문헌 목록에서 가져왔습니다. 해당 논문들의 본문 내용은 이 문서 작성 과정에서 직접 검증하지 않았으므로, 본문 수준의 세부 사항을 언급할 때에는 미확인으로 표시합니다.

---

## 1. Abstract (요약과 핵심 주장)

이 논문은 "범용 멀티모달 모델이 물리 상호작용을 이해하고 그 이해를 행동 결정에 활용할 수 있는가"라는 질문을, contact-rich manipulation과 촉각 이해라는 구체적인 사례로 검증하려 합니다 [Abstract, §1]. 제안 시스템인 **GPT-as-Tactile-Policy**는 파라미터가 고정된 GPT-6-Astra를 단독 action policy로 사용하며, 다음 두 단계로 구성됩니다.

1. **Offline**: 같은 GPT-6-Astra가 expert demonstration(태스크당 40개)을 분석하여 텍스트 형태의 Task Skill `K_τ`와 대표 촉각 reference 집합 `E_τ^tac`을 만듭니다 [§3.2, §4.1].
2. **Online**: scene/wrist RGB, 양손가락 촉각(이미지, 압입 깊이 통계, 마커 변위), proprioception, 에피소드 내 history를 입력으로 받아 EEF 목표 pose, gripper 목표값, 1–5 step의 실행 horizon을 출력하고, 일반 IK 제어기가 이를 실행합니다 [§3.1, §3.4].

저자가 내세우는 핵심 수치는 다음과 같습니다.

- UniVTAC 5개 태스크에서 평균 성공률 93.8%를 달성했습니다. 비교 대상인 N0-VTLA는 96.2%, N0-TWAM은 84.6%입니다 [Tab.2].
- Insert Hole에서 촉각을 제거하면 성공률이 99%에서 31%로 떨어집니다 [Tab.3].
- 추가 robotics-specific pretraining과 태스크별 파라미터 업데이트가 없습니다 [Tab.1].

제목은 "Outperform VTLA Models"라고 주장하지만, 평균 기준으로는 N0-VTLA(96.2%)보다 낮습니다. 따라서 제목의 주장은 본문 결과보다 강하게 표현되어 있습니다(§4.2 참조).

---

## 2. Method

### 2.1 상위 패러다임: GPT-as-Policy

이 논문은 같은 연구 계열의 선행 연구인 GPT-as-Policy("Systematically exploring the capabilities of GPT-6 Astra as embodied policies", Team et al., arXiv:2609.38537)의 direct-control 정식화를 그대로 계승합니다 [§2.1, §3.1]. 해당 정식화에서 GPT는 수치형 EEF/gripper 목표를 직접 생성하고, generic IK 제어기가 이를 관절 명령으로 바꾸며, 실행 후 새로운 관측이 다시 모델로 돌아갑니다. 이 논문의 기여는 그 루프에 (i) 양손가락 촉각 관측과 (ii) 데모로부터 증류한 고정 컨텍스트 `C_τ`를 추가한 것입니다. 액션 포맷, 프롬프트 구조, 출력 파싱 방식 등 세부 구현은 상위 논문에 있을 것으로 보이지만, 이 문서에서는 arXiv:2609.38537 본문을 확인하지 못했으므로 미확인입니다.

### 2.2 식 (1): 조건부 액션 생성

```
a_t ~ π_θ(· | ℓ, C_τ, o_t, H_t),   o_t = (V_t, s_t, T_t)            (1)
```

- `ℓ`: 자연어 태스크 지시입니다.
- `C_τ`: 태스크 τ에 대해 고정된 컨텍스트(식 2)입니다. 평가 전에 고정되며 같은 태스크의 모든 테스트 에피소드에서 공유됩니다 [§4.1].
- `o_t`: `V_t`(scene RGB와 wrist RGB), `s_t`(proprioception), `T_t`(양손가락 촉각, 식 3)로 구성됩니다.
- `H_t`: 에피소드 내의 이전 관측, 액션, "execution feedback"입니다. 에피소드마다 독립된 모델 세션을 시작하므로 에피소드 사이에 history가 리셋됩니다 [§4.1].
- `a_t`: 목표 EEF pose, gripper position, 짧은 실행 horizon(1–5 control step)으로 이루어집니다 [§3.1, §3.4].
- `θ`: offline 요약과 online 실행 모두에서 고정됩니다.

해설하면 식 (1)은 학습 대상 정책을 정의하는 식이 아니라, **in-context conditioning 구조를 선언하는 식**입니다. `π_θ`는 확률적 정책으로 표기되어 있지만, 샘플링 온도나 seed 같은 디코딩 설정은 보고되지 않았습니다(미확인). reasoning effort가 `xhigh`로 고정되었다는 사실만 명시되어 있습니다 [§4.1]. 결국 식 (1)이 의미하는 바는 "정책 개선이 파라미터 공간이 아니라 컨텍스트 공간(`C_τ`, `H_t`)에서 일어난다"는 것이며, 이것이 논문이 주장하는 "no parameter updates"의 실체입니다.

### 2.3 식 (2): 고정 태스크 컨텍스트

```
C_τ = { K_τ, E_τ^tac }                                               (2)
```

- `K_τ`(Task Skill)은 텍스트입니다. 원문에 따르면 Task Skill은 (a) task objectives, (b) manipulation principles, (c) stage transition의 근거(evidence), (d) 장애물 상황에서의 corrective strategies로 구성되며, "어떤 관측을 살펴보고 그것이 이후 행동에 어떻게 반영되어야 하는지"를 명시합니다 [§3.2]. 원문이 제시한 유일한 구체 예시는 "삽입 중 deformation은 증가하는데 물체가 전진하지 않으면 후퇴하여 접촉을 해소하고, 이어서 위치 또는 각도를 조정한 뒤 후속 피드백으로 그 효과를 평가한다"입니다 [§3.2].
- `E_τ^tac`은 데모에서 GPT-6-Astra가 직접 골라낸 대표 촉각 관측이며, "양손가락 접촉 분포와 국소 변형 패턴을 보존한다"고만 설명되어 있습니다 [§3.2, §4.1]. 몇 장을 고르는지, 어떤 기준으로 고르는지, 이미지 형태인지 descriptor 형태인지는 보고되지 않았습니다(미확인).

해설하면 `K_τ`는 촉각 신호에 **조작적 의미(operational meaning)**를 부여하는 규칙집 역할을 하고, `E_τ^tac`은 그 규칙을 지각적으로 grounding하는 few-shot exemplar 역할을 합니다. 구조적으로 보면 이 설계는 REFLECT(Liu et al., arXiv:2306.15724)처럼 경험을 언어로 요약하는 접근과, VLM에 few-shot 시각 예시를 주는 in-context imitation을 결합한 형태입니다.

### 2.4 식 (3): 촉각 관측 표현

```
T_t = { I_t^f, D_t^f, M_t^f }_{f ∈ {L, R}}                          (3)
```

- `I_t^f`: 손가락 f의 촉각 이미지입니다. [Fig.1A]와 [Fig.3]으로 보면 GelSight 계열의 마커 격자가 있는 RGB 이미지이며, 육안으로 대략 10×8 규모의 마커 격자가 보입니다(정확한 개수는 미확인).
- `D_t^f`: gel 압입(indentation) 맵을 공간적으로 pooling한 뒤 계산한 통계입니다. 원문에 명시된 항목은 mean indentation, peak indentation, contact region, centroid입니다 [§3.3].
- `M_t^f`: reference 위치 대비 마커의 2D 변위입니다 [§3.3]. [Fig.4]에서는 흰 배경 위의 quiver 형태로 시각화되어 있으며, [Fig.1A]에는 "Indentation" 히트맵과 "Marker motion" quiver가 별도의 패널로 그려져 있습니다. 따라서 모델이 마커 변위를 수치로 받는지, 렌더링된 quiver 이미지로 받는지, 혹은 둘 다 받는지는 확정할 수 없습니다(미확인이며, 리버스 엔지니어링 (d)에서 추정합니다).
- 두 손가락의 정보는 분리된 채로 유지되어, 모델이 좌우 접촉 차이와 그 변화를 비교할 수 있도록 합니다 [§3.3].

해설하면, 이미지(`I`)는 국소 디테일을 보존하고 descriptor(`D`, `M`)는 시점 간 정량 비교를 쉽게 만드는 이중 표현입니다. LLM은 이미지 두 장 사이의 미세한 차이를 수치적으로 비교하는 데 약하므로, 이 이중 표현은 합리적인 설계입니다. 다만 `D`는 시뮬레이터가 제공하는 정답 압입 깊이에 의존할 가능성이 높습니다. 실제 GelSight 센서에서는 photometric stereo 등으로 depth를 복원해야 하므로 sim-to-real 간극이 생깁니다(Limitations L5 참조).

또한 저자는 "연속된 관측 사이의 indentation 분포와 마커 변위를 같은 구간의 EEF/gripper 실측 이동량과 함께 비교한다"고 서술합니다 [§3.3]. 즉 모델 입력에는 **액션 전후의 차분 정보**가 포함되어야 합니다. 이 차분을 시스템이 미리 계산해서 주는지, 모델이 history에서 직접 비교하는지는 미확인입니다.

### 2.5 Offline 지식 구성 데이터 흐름

```
UniVTAC expert demos (태스크당 50개)
  ├─ 40개: knowledge construction
  │     └─ GPT-6-Astra(θ 고정)가 "액션 전후의 접촉 변화 + 물체 이동 + 태스크 진행"을 함께 분석
  │           ├─ K_τ : Task Skill 텍스트 (objectives / principles / stage evidence / corrective strategies)
  │           └─ E_τ^tac : 대표 촉각 reference
  └─ 10개: "offline validation" (용도는 서술되지 않음)
→ C_τ = {K_τ, E_τ^tac} 를 평가 전에 고정하고, 해당 태스크의 100개 테스트 에피소드 전체에서 재사용
```

근거는 [§3.2]와 [§4.1]입니다. 40개 데모 전체를 한 번의 컨텍스트에 넣었는지, 데모별로 분석한 뒤 집계(map-reduce)했는지, 사람이 결과를 검수하거나 수정했는지는 보고되지 않았습니다(미확인). 10개 validation 데모가 Task Skill을 반복적으로 수정하는 데 쓰였다면, 이는 사실상 사람이 수행하는 prompt optimization에 해당합니다(L1 참조).

### 2.6 Online 폐루프 데이터 흐름

```
for t = 0, 1, ...:
  o_t ← (scene RGB, wrist RGB, s_t, {I, D, M}_{L,R})
  a_t ← GPT-6-Astra(xhigh)(ℓ, C_τ, o_t, H_t)          # 목표 EEF pose, gripper, horizon h ∈ {1..5}
  bound/format 검사 → 통과하면 IK 제어기가 h control step 동안 실행
  o_{t+1} 획득, H_{t+1} ← H_t ∪ {o_t, a_t, feedback}
  (다음 호출에서) o_t와 o_{t+1}을 비교하여 진행 여부 판단 → 전진 / probing / 교정 선택
종료: 환경의 native success 조건을 만족하거나, 실행 예산을 소진하면 종료합니다(후자는 실패로 처리).
```

근거는 [§3.4]와 [§4.1]입니다. 원문은 액션을 "최신 측정 로봇 상태로부터 구성한다"고 서술합니다 [§3.4]. 이는 모델이 현재 실측 pose를 기준으로 목표를 계산한다는 뜻이며, 액션이 절대 목표 pose이더라도 그 값이 사실상 "현재 pose + 증분"으로 만들어진다는 점을 시사합니다. 불확실한 상황에서는 작은 병진, 자세 조정, gripper 개도 변경 같은 탐색 액션을 수행하고, 그 결과가 `H_t`에 누적됩니다 [§3.4]. [Fig.4]는 같은 에피소드 안에서 near-vertical(step 30→35→40), leftward(119→124→127), rightward(191→194→199) 방향의 probing을 차례로 시도하는 모습을 보여줍니다.

### 2.7 [Fig.1A] 구성요소 매핑

| Fig.1A 구성요소 | 식/기호 | 원문 근거 | 구현상 대응 (추정 포함) |
|---|---|---|---|
| Scene RGB, Wrist RGB | `V_t` | §3.1 | 시뮬레이터 카메라 2대의 프레임 |
| L/R touch | `I_t^{L,R}` | §3.3 | 마커가 있는 GelSight형 촉각 이미지 |
| Indentation | `D_t^f` | §3.3 | 압입 맵을 pooling한 통계 (텍스트/JSON, 추정) |
| Marker motion | `M_t^f` | §3.3 | 마커 변위 벡터장 (수치 및/또는 quiver 이미지, 추정) |
| Expert demos `D_expert` → Offline analysis | — | §3.2 | GPT-6-Astra 분석 호출 (오프라인 배치) |
| Frozen context: Knowledge `K_τ`, Touch refs `E_τ` | `C_τ` (식 2) | §3.2 | 시스템 프롬프트 + reference 이미지 블록 (추정) |
| Instruction `ℓ` | `ℓ` | §3.1 | 태스크 지시 문자열 |
| Observation `o_t` (Vision, touch, state) | `o_t` | §3.1 | 스텝별 user 메시지 (추정) |
| History `H_t` | `H_t` | §3.1, §4.1 | 에피소드 단위 세션의 대화 맥락 (추정) |
| GPT-6-Astra Policy `π_θ` | `π_θ` | §3.1, §4.1 | reasoning effort xhigh |
| Action `a_t`: EEF pose, Gripper, Steps | `a_t` | §3.1, §3.4 | 구조화된 출력 (JSON, 추정) |
| IK + execute | — | §3.4 | bound/format 검사 → generic IK → h step 실행 |
| New input `o_{t+1}, H_{t+1}` | — | §3.4 | 다음 호출의 입력 |

---

## 3. Experiments

### 3.1 설정 요약

| 항목 | 내용 | 근거 |
|---|---|---|
| 벤치마크 | UniVTAC (Chen et al., arXiv:2602.10093): 시뮬레이션 전용 | §4.1 |
| 태스크 | Insert Hole, Lift Can, Grasp Classify, Lift Bottle, Pull-out Key | §4.1 |
| 데모 | 태스크당 50개 = 40개(knowledge construction) + 10개(offline validation) | §4.1 |
| 정책 | GPT-6-Astra, reasoning effort xhigh, 모든 태스크에서 동일 설정 | §4.1 |
| 세션 | 에피소드마다 독립 세션, history는 에피소드 내에서만 유지 | §4.1 |
| 평가 | 태스크당 고정 테스트 에피소드 100개, 총 500개. 환경의 native success 기준 사용. 태스크 단순 평균 | §4.1 |
| 실행 예산 | 수치는 보고되지 않음. [Fig.2]의 w/o tactile 실패 에피소드가 step 300에서 끝나므로 Insert Hole의 상한은 300 control step으로 추정 | Fig.2 (추정) |

### 3.2 [Tab.1] 학습 비용 비교 (원문 재정리)

| Method | Specialized pretraining | Per-task training | Knowledge-guided probing |
|---|---|---|---|
| ACT + UniVTAC | Tactile encoder | 50 demos; 4,000 steps | × |
| N0-VTLA | Multi-node cluster | One policy per task | × |
| N0-TWAM | 128×H800; 8×A100 | One policy/task; 1k steps | × |
| Ours | No additional pretraining | No parameter updates | ✓ |

이 표는 **학습 비용만** 비교하고 **추론 비용**(호출 수, 토큰, 지연시간)은 다루지 않습니다. 또한 GPT-6-Astra 자체의 사전학습 비용과 학습 데이터 구성도 공개되지 않았으므로 "No additional pretraining"이라는 비교 기준은 비대칭입니다(L2 참조).

### 3.3 [Tab.2] 주 결과 재정리와 통계적 해석

| Method | Insert Hole | Lift Can | Grasp Classify | Lift Bottle | Pull-out Key | Mean |
|---|---|---|---|---|---|---|
| ACT + UniVTAC Encoder | 24.0 | 29.0 | 99.0 | 71.0 | 46.0 | 53.8 |
| π0.5 | 25.0 | 6.0 | 49.0 | **100.0** | 35.0 | 43.0 |
| N0-VTLA | 95.0 | 88.0 | **100.0** | 99.0 | **99.0** | **96.2** |
| N0-TWAM | **99.0** | 93.0 | 94.0 | 58.0 | 79.0 | 84.6 |
| Ours | **99.0** | **95.0** | 98.0 | 93.0 | 84.0 | 93.8 |

[Fig.1C]의 막대그래프 수치는 [Tab.2]와 일치합니다(page-01.png에서 직접 대조했습니다).

아래 통계는 이 문서에서 직접 계산한 것입니다. 각 셀을 n=100 Bernoulli 시행으로 가정했고, baseline도 n=100이라고 가정했습니다. baseline의 실제 시행 수와 seed 구성은 미확인입니다.

| 비교 | 차이 | 검정 결과 (two-proportion z, 양측) | 해석 |
|---|---|---|---|
| Insert Hole: Ours 99 vs N0-VTLA 95 | +4pp | z=1.66, p≈0.10 | 유의하지 않음 |
| Lift Can: Ours 95 vs N0-VTLA 88 | +7pp | z=1.77, p≈0.08 | 유의하지 않음 |
| Lift Can: Ours 95 vs N0-TWAM 93 | +2pp | z=0.60, p≈0.55 | 유의하지 않음 |
| Grasp Classify: Ours 98 vs N0-VTLA 100 | −2pp | p≈0.16 | 유의하지 않음 |
| Lift Bottle: Ours 93 vs N0-VTLA 99 | −6pp | z=−2.17, p≈0.03 | 유의하게 열세 |
| Pull-out Key: Ours 84 vs N0-VTLA 99 | −15pp | z=−3.80, p<0.001 | 유의하게 열세 |
| Mean: Ours 93.8 vs N0-VTLA 96.2 | −2.4pp | SE≈1.34pp, z≈−1.79, p≈0.07 | 경계선, 수치상 열세 |
| Insert Hole 촉각 ablation: 99 vs 31 | +68pp | Wilson 95% CI: [94.6, 99.8] vs [22.8, 40.6] | 매우 유의함 |

Wilson 95% 신뢰구간(Ours 기준)은 Insert Hole [94.6, 99.8], Lift Can [88.8, 97.8], Grasp Classify [93.0, 99.4], Lift Bottle [86.3, 96.6], Pull-out Key [75.6, 89.9]입니다.

해석하면, 저자가 "우위"로 제시한 항목(Insert Hole에서 N0-VTLA보다 높음, Lift Can 최고치)은 모두 통계적으로 유의하지 않습니다. 반면 열세인 항목(Pull-out Key, Lift Bottle에서 N0-VTLA보다 낮음)은 유의합니다. 따라서 "comparable to SOTA"라는 본문 표현은 대체로 방어할 수 있지만, 제목의 "Outperform VTLA Models"는 데이터가 지지하지 않습니다. 반면 "N0-TWAM과 모든 태스크에서 동등하거나 그 이상"이라는 주장 [§4.2]은 수치상 사실이며, 특히 Lift Bottle(93 vs 58)에서 차이가 큽니다.

### 3.4 [Tab.3] 촉각 ablation (Insert Hole 전용)

| 조건 | Online 관측 | 성공률 |
|---|---|---|
| Ours | V+P+T | 99.0 |
| w/o tactile | V+P | 31.0 |

- w/o tactile 조건에서는 촉각 관측과 촉각 reference를 모두 제거하고, **"사용 가능한 관측에 맞춘" 별도의 고정 Task Skill**을 사용합니다 [§4.3]. 두 조건은 같은 100개 테스트 설정, 같은 초기화, 같은 저수준 제어기와 실행 예산을 공유합니다 [§4.3].
- [Fig.2]의 대표 에피소드에서 w/ tactile 조건은 튜브 축을 약 30° 바꾸고 측면으로 약 40mm 이동한 뒤 step 191에 삽입을 완료합니다. w/o tactile 조건은 축 변화가 5° 미만에 머물고 측면 변위가 거의 0인 채로 step 300에 실패합니다.
- 주의할 점이 있습니다. [Fig.3D]의 keyframe("Rock to align", "Correct alignment")과 [Fig.2]의 약 30° 축 변화로 보면, Insert Hole의 구멍은 기울어진 형상일 가능성이 높습니다. 그렇다면 필요한 교정량은 "local pose error"라고 부르기에는 상당히 큽니다. 또한 vision-only 조건이 31%에 그친 원인이 촉각 정보의 부재인지, 촉각 없이 증류된 Task Skill에 "기울여서 정렬한다"는 교정 전략이 빠졌기 때문인지를 이 실험으로는 구분할 수 없습니다(L7 참조).

### 3.5 Baseline 설명

| Baseline | 정체 | 출처 | 비고 |
|---|---|---|---|
| ACT + UniVTAC Encoder | ACT(Action Chunking Transformer) 정책에 UniVTAC의 사전학습 visuo-tactile 인코더를 결합한 정책입니다. 태스크당 50 demo, 4,000 step으로 학습합니다 [Tab.1]. | Chen et al., UniVTAC, arXiv:2602.10093 | ACT 원 논문(Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023, arXiv:2304.13705)은 이 논문의 참고문헌에 없습니다. |
| π0.5 | Physical Intelligence의 VLA입니다. 이 논문에서는 vision-only baseline으로 사용됩니다. | Intelligence et al., arXiv:2504.16054 | UniVTAC 태스크에 어떻게 fine-tuning했는지는 미확인입니다. |
| E0 | Tweedie discrete diffusion 기반의 이산 액션 VLA입니다. 원문에서 vision-only baseline으로 언급됩니다 [§4.1]. | Zhan et al., NeurIPS 2026 | **[Tab.2]와 [Fig.1C]에 결과가 없습니다.** 본문 언급과 결과 표가 서로 불일치합니다. 또한 E0의 저자 중 L. Lin과 G. Wang은 이 논문의 저자이기도 합니다. |
| N0-VTLA | latent tactile token을 예측하는 경로를 가진 대규모 visuo-tactile 사전학습 VTLA이며, 저장된 배포 데이터로 offline policy improvement를 수행합니다 [§2.2]. | Team et al., arXiv:2607.23782 | 태스크별로 정책을 하나씩 학습합니다 [Tab.1]. |
| N0-TWAM | 미래의 vision, touch, action을 함께 예측하는 tactile-native world-action model입니다. 통합된 force 기반 촉각 표현과 contact-event 기반 태스크 staging을 사용합니다 [§2.2]. | Team et al., arXiv:2607.23783 | 사전학습에 128×H800과 8×A100을 사용했고, 태스크별로 1k step을 학습합니다 [Tab.1]. |

원문은 "previously evaluated on UniVTAC"인 baseline **4개**와 비교한다고 쓰면서 실제로는 **5개**(E0, π0.5, ACT+UniVTAC, N0-VTLA, N0-TWAM)를 나열합니다 [§4.1]. baseline 수치를 저자가 직접 재실행했는지, 기존 논문에서 가져왔는지도 명시되어 있지 않습니다. "previously evaluated"라는 표현으로 보아 기존 보고치를 인용했을 가능성이 높으며(확실도: 중간), 그렇다면 테스트 seed, 에피소드 예산, 성공 판정이 완전히 같다는 보장이 없습니다.

### 3.6 정성 결과

- [Fig.3]: Grasp Classify(목표 근처에서 위치 미세조정), Pull-out Key(시험 당김 → 각도 조정 → 인출), Lift Bottle(파지 → 들고 기울임 → 세움 → 놓음), Insert Hole(하강이 막힘 → 흔들어 정렬 → 전진)의 성공 사례를 보여줍니다.
- [Fig.4]: Insert Hole의 한 에피소드에서 세 방향으로 probing을 시도합니다. 전진 중 마커 변위가 커지면 후퇴하고, 변위가 작으면 계속 전진하는 패턴이 나타납니다. 이 패턴은 [§3.2]의 Task Skill 예시 규칙과 정확히 일치합니다. 그러므로 이 결과는 "모델이 촉각 물리를 이해한다"는 증거라기보다 "모델이 증류된 규칙을 충실히 따른다"는 증거로 해석하는 편이 보수적입니다.
- 사소한 불일치도 있습니다. [Fig.2]의 w/ tactile 에피소드는 step 191에 완료되는데, [Fig.4]의 에피소드는 step 191에서 rightward probing을 시작합니다. 따라서 두 그림은 서로 다른 에피소드입니다. 원문은 [Fig.4]를 "a single Insert Hole episode"라고만 설명하므로 모순은 아니지만, 독자가 혼동할 수 있습니다.

---

## 4. Limitations (비판적 검토)

저자는 별도의 Limitations 절을 두지 않았습니다. 아래는 원문에 근거한 비판이며, 각 항목에 근거 위치를 붙였습니다.

| # | 한계 | 근거 위치 | 영향 |
|---|---|---|---|
| L1 | **"Training-free" 표기와 실제 데이터 사용의 괴리.** [Fig.1C] 범례는 "Ours (Training-free)"라고 표기하지만, 태스크당 40개 데모를 in-context로 증류하고 10개를 "offline validation"에 사용합니다. validation이 Task Skill을 반복 수정하는 데 쓰였다면, 이는 사람 또는 모델이 수행하는 prompt-level optimization입니다. 데이터 사용량(50 demo)은 ACT+UniVTAC(50 demo)와 같은 수준입니다. | Fig.1C, §4.1, Tab.1 | 주장 과장. "parameter-free adaptation"이 더 정확한 표현입니다. |
| L2 | **비대칭적인 비용 비교.** [Tab.1]은 baseline의 사전학습 비용(128×H800 등)을 나열하지만, GPT-6-Astra의 사전학습 비용과 데이터(로보틱스나 촉각 데이터 포함 여부)는 공개되지 않았습니다. "No additional robotics-specific pretraining"은 검증할 수 없는 주장입니다. UniVTAC 유사 데이터가 사전학습에 섞였을 가능성(contamination)도 배제할 수 없습니다(미확인). | Tab.1, Abstract | 비교의 공정성 |
| L3 | **추론 비용과 지연시간을 보고하지 않음.** reasoning effort가 xhigh이고, 1–5 step마다 한 번씩 호출하며, 에피소드는 약 191–300 step입니다. 에피소드당 수십에서 백 회 이상의 호출이 필요할 것으로 추정되지만(리버스 엔지니어링 (g) 참조), 토큰 수, API 비용, 호출당 지연, 에피소드 wall-clock 시간이 모두 빠져 있습니다. | §4.1, §3.4, Fig.2 | 실시간 제어 가능성과 실용성을 평가할 수 없습니다. |
| L4 | **시뮬레이션 전용 평가.** 5개 태스크 모두 UniVTAC 시뮬레이션에서만 평가했습니다. 시뮬레이션은 LLM 호출 동안 시간을 멈출 수 있지만, 실제 로봇에서는 수십 초 단위의 지연 동안 접촉 상태가 유지된다는 보장이 없습니다. | §4.1 | 실세계 일반화 미검증 |
| L5 | **촉각 descriptor가 시뮬레이터의 특권 정보에 의존할 가능성.** `D_t`는 "gel indentation map"에서 계산합니다. 시뮬레이터에서는 정답 depth를 쉽게 얻지만, 실제 GelSight에서는 photometric stereo로 복원해야 하며 노이즈가 큽니다. 마커 변위 역시 실제 센서에서는 마커 추적 오차가 생깁니다. | §3.3 | sim-to-real 간극 |
| L6 | **단일 run이며 신뢰구간이 없음.** 각 설정을 100 에피소드로 한 번만 실행했고, CI, seed 분산, 반복 실행 결과가 없습니다. 3.3절의 계산에 따르면 저자가 강조한 우위는 유의하지 않고, 열세(Pull-out Key, Lift Bottle)는 유의합니다. closed model은 같은 입력에도 출력이 달라질 수 있어 run 간 분산을 보고하는 것이 특히 중요합니다. | Tab.2, Tab.3, §4.1 | 결론의 통계적 강도 부족 |
| L7 | **촉각 ablation의 confound.** w/o tactile 조건은 촉각 관측만 제거한 것이 아니라 Task Skill 자체를 촉각 없이 다시 증류했습니다("fixed Task Skill matched to its available observations"). 따라서 99→31의 차이에는 (i) online 촉각 관측의 효과, (ii) offline 증류 단계에서 촉각을 본 효과, (iii) 촉각 reference의 효과가 섞여 있습니다. 이를 분리하려면 2×2(또는 2×2×2) 설계가 필요합니다. 예를 들어 촉각 기반 Task Skill을 유지한 채 online 촉각만 제거하는 조건이 필요합니다. | §4.3 | 인과 귀속 불가 |
| L8 | **구성요소 ablation 부재.** Task Skill 유무(zero-shot), 촉각 reference 유무, descriptor(`D`,`M`)와 이미지(`I`)의 상대 기여, history 길이와 truncation, horizon 상한(1–5) 고정 대비 가변, reasoning effort(xhigh 대비 low/medium), 데모 수(40 대비 5/10/20)에 대한 ablation이 하나도 없습니다. 특히 "모델이 촉각을 이해한다"는 제목의 주장을 검증하려면 Task Skill 없는 조건이 핵심인데 이 조건이 빠져 있습니다. | §4.3 (ablation 1종) | 메커니즘 주장 근거 부족 |
| L9 | **ablation이 Insert Hole 하나뿐임.** 나머지 4개 태스크에서 촉각의 기여는 측정되지 않았습니다. Grasp Classify는 "tactile classification"을 포함한다고 설명되지만 [§4.1] 촉각 제거 결과가 없습니다. | §4.1, §4.3 | 일반화 불가 |
| L10 | **closed model 재현성.** GPT-6-Astra의 버전이나 스냅샷, 날짜, 디코딩 설정(temperature 등), 프롬프트 원문, Task Skill 원문, reference 선택 결과가 공개되지 않았습니다(프로젝트 페이지 내용은 미확인). 모델이 업데이트되면 결과를 재현할 수 없습니다. | §4.1 | 재현성 |
| L11 | **"이해(understands)"라는 주장의 범위.** 관찰된 행동([Fig.4]: 변위가 크면 후퇴하고 작으면 전진)은 [§3.2]에 명시된 Task Skill 규칙과 동형입니다. 규칙 추종과 물리 이해를 구분하는 실험(규칙에 없는 새로운 접촉 상황, 반사실적 촉각 입력, Task Skill 없는 조건)이 없습니다. | Title, §3.2, §4.4 | 제목 과장 |
| L12 | **실패 분석 부재.** Pull-out Key(84%)와 Lift Bottle(93%)에서 N0-VTLA보다 유의하게 낮은 원인(지연, horizon 제약, 동적 동작, 파싱 실패 등)을 분석하지 않았습니다. 형식 오류나 bound 위반 시 재시도 정책도 서술되지 않았습니다 [§3.4]. | Tab.2, §3.4 | 약점 파악 불가 |
| L13 | **baseline 프로토콜의 불확실성.** baseline 수치의 출처(재실행인지 인용인지)와 seed 일치 여부가 명시되지 않았습니다. E0는 baseline으로 언급되었지만 결과가 없으며, "4개 baseline"이라고 쓰고 5개를 나열합니다. | §4.1, Tab.2 | 비교 신뢰도 |
| L14 | **데이터 누설 가능성.** 50개 expert 데모의 초기 설정과 100개 테스트 seed가 분리되어 있는지 명시되지 않았습니다. 촉각 reference가 테스트와 같은 초기 조건에서 나왔다면 사실상 테스트 정보를 참조하는 셈이 됩니다(미확인). | §4.1 | 일반화 과대평가 가능성 |
| L15 | **명칭과 서술의 비일관성.** GPT-as-Tactile-Policy와 GPT-as-Policy를 혼용합니다([§4.2], [Fig.2] 캡션). 제목의 문법("Outperform")도 어색합니다. 결과에는 영향이 없지만 원고 완성도가 낮다는 신호입니다. | Title, §4.2, Fig.2 | 경미 |
| L16 | **안전과 힘 제어 부재.** 위치 목표만 출력하며 힘이나 임피던스 제어, 접촉력 상한을 서술하지 않습니다. 시뮬레이션에서는 과도한 압입이 문제가 되지 않지만, 실제 환경에서는 센서나 물체 손상 위험이 있습니다. Tactile-VLA(Huang et al., arXiv:2507.09160)의 hybrid position-force control과 대조됩니다. | §3.1, §3.4 | 실세계 적용성 |

---

## 5. Reverse Engineering

각 항목에 **확실도(높음/중간/낮음)**와 근거를 붙였습니다. "높음"은 원문에 직접 근거가 있는 경우, "중간"은 원문의 복수 단서에서 강하게 추론되는 경우, "낮음"은 합리적인 설계 추정에 불과한 경우를 뜻합니다. 아래의 프롬프트와 스키마는 재구현을 위한 **초안**이며, 저자의 실제 프롬프트가 아닙니다.

### (a) Task Skill 생성 프롬프트와 출력 구조

**출력 구조 (확실도: 높음, 4개 상위 필드)**
근거: [§3.2]가 objectives, principles, stage-transition evidence, corrective strategies를 명시합니다.

**하위 필드와 생성 절차 (확실도: 낮음–중간)**
근거: 40개 데모를 각각 수백 step 길이로 통째로 넣기는 컨텍스트 한계상 어렵습니다. 그래서 (1) 데모별로 키프레임을 subsampling하고 분석한 뒤 (2) 집계하는 2단계 map-reduce 방식이 현실적입니다. 키프레임 선택은 촉각 이벤트(접촉 시작, 압입 급증, 미끄러짐)를 기준으로 했을 가능성이 높습니다.

```yaml
# K_τ 출력 스키마 초안 (YAML)
task: insert_hole
objectives:
  - "튜브를 블록의 기울어진 구멍에 끝까지 삽입한다"
stages:                      # stage-transition evidence
  - name: approach
    exit_evidence: ["양손가락 indentation 변화 없음", "튜브 끝이 구멍 입구 위에 위치(scene RGB)"]
  - name: contact
    exit_evidence: ["하강 중 peak indentation 증가", "마커 변위가 축방향 shear로 나타남"]
  - name: align
    exit_evidence: ["전진 시 마커 변위 증가폭이 작음", "튜브가 실제로 하강(wrist RGB/EEF z)"]
  - name: insert
    exit_evidence: ["EEF가 목표 깊이에 도달", "변위 안정"]
principles:
  - "한 번에 하나의 자유도만 조정하고, 다음 관측으로 효과를 평가한다"
  - "좌우 indentation 비대칭은 튜브의 기울어짐/미끄러짐 신호로 해석한다"
corrective_strategies:
  - trigger: "deformation 증가 + EEF/물체 전진 없음"
    action: "2–5mm 후퇴 → 측면 1–3mm 또는 축 3–5° 조정 → 재전진"
    evaluate: "재전진 시 마커 변위 크기 비교"
  - trigger: "좌우 centroid가 반대 방향으로 이동"
    action: "gripper 재파지 또는 roll 보정"
action_guidelines:
  step_size: {pos_mm: [1, 5], rot_deg: [2, 5]}
  horizon: "접촉 중 1–2, 자유공간 3–5"
```

```text
# 오프라인 Stage 1 (데모 단위 분석) 프롬프트 초안
[SYSTEM] 너는 로봇 조작 전문가다. 아래 expert demonstration의 키프레임을 시간순으로 분석하라.
각 키프레임에는 scene/wrist RGB, 좌/우 촉각 이미지, indentation 통계(mean/peak/area/centroid),
marker displacement 요약, EEF pose, gripper 값, expert action이 포함된다.
[USER] 태스크: {instruction}
데모 #{k} 키프레임 {N}개: {frames...}
다음을 JSON으로 답하라:
1) 단계 분할과 각 단계 전이 시점의 관측 증거
2) 접촉 변화(좌우 비대칭, 변위 방향)와 직전 expert action의 관계
3) 막힘(obstruction)이 발생한 구간과 expert의 교정 방식, 그 효과
4) 이 데모에서 대표성이 높은 촉각 프레임 후보(인덱스와 이유)

# 오프라인 Stage 2 (집계) 프롬프트 초안
[USER] 아래는 데모 40개의 분석 결과다. 공통 패턴을 추출하여 Task Skill을
{objectives, stages(exit_evidence), principles, corrective_strategies}로 작성하라.
규칙은 관측 가능한 신호(촉각 통계·마커 변위·EEF 이동·물체 이동)에 근거해야 하며,
수치 임계값은 데모에서 관찰된 범위로 제시하라. 또한 대표 촉각 reference를 단계별로 선택하라.
```

### (b) 온라인 정책 시스템/유저 프롬프트 템플릿 초안

**구조 (확실도: 중간)**
근거: 고정 컨텍스트(`ℓ`, `C_τ`)와 가변 입력(`o_t`, `H_t`)이 분리되어 있고 [§3.1, §3.2], 에피소드 단위 "session"이 존재합니다 [§4.1]. 고정 부분을 system 또는 첫 메시지에 두고 스텝마다 user 메시지를 추가하는 구조가 가장 자연스럽고, prompt caching에도 유리합니다.

```text
[SYSTEM]
너는 7-DoF 로봇팔과 평행 그리퍼(양 손가락에 GelSight형 촉각 센서)를 제어하는 정책이다.
태스크: {ℓ}
== Task Skill ==
{K_τ}
== Tactile references ==
{E_τ^tac: 단계 라벨 + 좌/우 촉각 이미지 + 해당 descriptor}
== 관측 정의 ==
- indentation: mean/peak (mm), contact_area (px 비율), centroid (u,v ∈ [0,1])
- marker: mean_disp (px), max_disp, dominant_dir (deg), [선택] divergence, curl
== 행동 규칙 ==
- 매 스텝 직전 액션의 효과를 먼저 평가하라(EEF 이동, 물체 반응, 촉각 변화).
- 방향이 불확실하면 작은 탐색 액션(평행이동/회전/그리퍼 개도)을 선택하라.
- 출력은 지정된 JSON 스키마만 사용하라. 이동량 한계: |Δp| ≤ 10mm, |Δθ| ≤ 10°/step.

[USER @ step t]
step: {t}  /  budget: {T_max}
proprio: eef_pos={x,y,z} eef_quat={qx,qy,qz,qw} gripper={g}
Δ since last action: eef_Δp={...} eef_Δrot={...} gripper_Δ={...}
tactile L: indentation={...} marker={...}   Δ vs prev: {...}
tactile R: indentation={...} marker={...}   Δ vs prev: {...}
images: [scene_rgb] [wrist_rgb] [tactile_L] [tactile_R] ([marker_quiver_L] [marker_quiver_R])
last_action: {a_{t-1}}   exec_feedback: {ok | clipped | ik_fail | format_error}
→ 다음 액션을 JSON으로 출력하라.
```

### (c) Action JSON 스키마

- 필드 구성(EEF pose, gripper, horizon 1–5): **확실도 높음**. 근거는 [§3.1], [§3.4], [Fig.1A] "EEF pose / Gripper / Steps"입니다.
- 절대 목표 pose와 델타 중 어느 쪽인지: **확실도 중간**. "target end-effector pose"라는 표현 [§3.1]과 IK 입력이라는 점은 절대 목표를 시사합니다. 반면 "constructs a_t from the latest measured robot state" [§3.4]는 모델이 실측 pose에 증분을 더해 절대값을 만든다는 뜻으로 읽힙니다. 따라서 **출력은 절대값, 모델 내부 추론은 델타**일 가능성이 가장 높습니다.
- 회전 표현(quaternion, Euler): **확실도 낮음**. LLM이 quaternion을 산술적으로 다루는 데 약하다는 점을 고려하면, RPY(deg)나 axis-angle 델타가 더 다루기 쉽습니다. 재구현에서는 모델에게 델타 RPY를 받고 시스템이 quaternion으로 합성하는 방식을 권장합니다.
- `done`, `stage`, `rationale` 필드: **확실도 낮음**. 종료는 환경의 native success 조건이 판정하므로 [§4.1] `done` 필드는 필수가 아닙니다. 다만 stage 라벨은 Task Skill의 stage-transition evidence와 맞물리므로 포함했을 가능성이 있습니다.

```json
{
  "stage": "align",
  "assessment": "전진 시 좌측 marker 변위 증가, EEF z 정체 → 막힘",
  "eef_target": {
    "frame": "base",
    "position_m": [0.512, -0.034, 0.181],
    "rpy_deg": [178.0, 4.5, 90.0]
  },
  "gripper": 0.012,
  "horizon": 2
}
```

검증 로직(확실도: 높음, "checks on command bounds and format" [§3.4])의 내용은 다음과 같습니다. JSON 파싱, 필드 타입 확인, workspace 범위 clip, step당 최대 이동량 제한, horizon ∈ {1..5}, gripper ∈ [0, g_max]. 위반 시 재요청할지 clip할지는 미확인입니다.

### (d) 촉각 descriptor 계산 방법 후보

| 항목 | 계산 후보 | 확실도 | 근거 |
|---|---|---|---|
| mean indentation | contact mask 내부 depth 평균 (또는 전체 평균) | 높음(항목 존재) / 낮음(정의) | §3.3 |
| peak indentation | depth 최댓값 또는 상위 1% 분위수 | 높음 / 중간 | §3.3 |
| contact region | `depth > δ`(예: 0.05mm) 픽셀 비율, 연결 성분 수, bbox | 높음 / 낮음 | §3.3 "contact regions" |
| centroid | depth 가중 무게중심 (u, v), 정규화 좌표 | 높음 / 중간 | §3.3 |
| spatial pooling | depth 맵을 3×3 또는 4×4 격자로 평균 pooling한 표 | 중간 / 낮음(격자 크기) | §3.3 "spatially pool" |
| 접촉 패치 형상 | 2차 모멘트로 계산한 주축 방향과 이심률 (튜브/키의 기울기 단서) | 낮음 | 원문에 없음. 설계 제안 |
| marker 평균/최대 변위 | 마커별 ‖d_i‖의 평균과 최대, 평균 벡터 방향 | 중간 | §3.3, Fig.4 (변위 크기로 판단) |
| marker divergence/curl | 격자 보간 후 ∇·d (법선 압력), ∇×d (비틀림 토크) | 낮음 | 원문에 없음. 설계 제안 |
| L/R 차분 | ΔL−R의 peak, centroid 차이, 변위 방향 차이 | 중간 | §3.3 "bilateral contact differences" |
| 시간 차분 | t−1 대비 위 통계의 변화량 | 중간 | §3.3 "across successive observations" |
| marker quiver 이미지 | 변위를 확대한 화살표 플롯을 이미지로 제공 | 중간 | Fig.1A, Fig.4의 시각화 형식 |

### (e) 대표 tactile reference 선택 방법 후보

| 후보 | 설명 | 확실도 | 근거 |
|---|---|---|---|
| E1. GPT 직접 선택 | Stage 1 분석에서 모델이 "대표 프레임"을 지목하고, Stage 2에서 단계별로 정리합니다. | 중간–높음 | §3.2 "GPT-6-Astra also extracts representative tactile observations" |
| E2. 단계 정렬 exemplar | stage(무접촉/첫 접촉/막힘/정렬 후 미끄러짐/완료)마다 1–2쌍(L/R)을 선택하고 단계 라벨을 붙입니다. | 중간 | Task Skill의 stage-transition evidence와 짝을 이루어야 grounding이 가능합니다 [§3.2]. |
| E3. 규칙 기반 사전 필터 + GPT 선택 | indentation 급변 시점(contact event)을 자동 검출해 후보를 줄인 뒤 GPT가 최종 선택합니다. | 낮음 | 원문에 없음. 컨텍스트 비용 관점의 추정 |
| E4. descriptor 군집화 | 40개 데모의 촉각 descriptor를 k-medoids로 군집화하고 medoid를 reference로 사용합니다. | 낮음 | "preserving bilateral contact distributions"를 충실히 구현하는 방식 |
| reference 개수 | 태스크당 4–12개(L/R 쌍 2–6개) | 낮음 | 컨텍스트 예산 추정 |

### (f) History 관리

| 항목 | 추정 | 확실도 | 근거 |
|---|---|---|---|
| 범위 | 에피소드 내부에서만 유지하고 에피소드 사이에서는 리셋합니다. | 높음 | §4.1 |
| 구현 형태 | 대화 세션(multi-turn)에 스텝 메시지를 누적합니다. | 중간 | §4.1 "independent model session" |
| 내용 | 이전 관측, 액션, execution feedback(bound/IK 결과)을 담습니다. | 높음 | §3.1 |
| 이미지 보존 | 최근 K(2–5) 스텝만 이미지를 유지하고, 그 이전은 텍스트 descriptor와 액션 로그로 압축합니다. | 낮음–중간 | 191 step / h≈3이면 약 64회 호출이 필요하고, 호출당 이미지 4–6장이 전부 누적되면 이미지가 300장을 넘어 컨텍스트 비용이 2차적으로 증가합니다. 원문은 truncation을 언급하지 않습니다. |
| 탐색 기억 | 시도한 방향과 결과 요약(예: "near-vertical: 막힘, leftward: 막힘")을 유지합니다. | 중간 | Fig.4에서 같은 방향을 반복하지 않고 다른 방향을 순차적으로 시도합니다. §4.4 "responses retained in H_t" |
| reasoning 토큰 | 이전 턴의 reasoning은 history에 포함되지 않습니다(일반적인 API 동작 가정). | 낮음 | 미확인 |

### (g) 호출 수, 에피소드 길이, 비용 추정

**에피소드 길이 단서 (확실도: 중간)**
- [Fig.2]: w/ tactile 성공은 step 191, w/o tactile 실패는 step 300(예산 상한으로 추정)입니다.
- [Fig.4]: 표시된 시점 간격이 30→35→40, 119→124→127, 191→194→199로 3–5 step입니다. 이 시점들이 의사결정 경계와 일치한다면, 탐색 중 평균 horizon은 약 4입니다. 따라서 "step"은 LLM 호출 단위가 아니라 **control step**이며, 한 호출이 1–5 control step을 실행한다고 해석하는 것이 일관적입니다.

**호출 수 추정 (확실도: 낮음–중간)**

`N_calls/episode ≈ L / h̄`이고 h̄ ∈ [2, 4]로 가정합니다.

| 시나리오 | L (control step) | h̄=2 | h̄=3 | h̄=4 |
|---|---|---|---|---|
| Insert Hole 성공 (Fig.2) | 191 | 96 | 64 | 48 |
| 예산 소진 실패 | 300 | 150 | 100 | 75 |
| 다른 태스크 (길이 미보고, 150–250 가정) | 200 | 100 | 67 | 50 |

- 주 평가 500 에피소드 × 50–100회 ≈ **25k–50k회 호출**입니다.
- Tab.3 ablation 100 에피소드(31개 성공 약 200 step, 69개 실패 300 step, 평균 약 270 step) ÷ h̄ ≈ 68–135회이므로, 합계 **약 7k–13k회**입니다.
- Offline 증류: 태스크당 40개 데모 분석과 집계 호출로, 5개 태스크 합계 **수백 회 수준**이며 무시할 수 있는 규모입니다.
- 총합은 **약 3만–6만 회의 xhigh 호출**입니다. 실험 중 prompt를 반복 수정했다면 이 수치는 더 커집니다.

**호출당 토큰 추정 (확실도: 낮음)**
GPT-6-Astra의 이미지 토큰화 방식과 가격은 미확인입니다. 아래는 매개변수화한 추정입니다.

| 구성 | 추정 토큰 |
|---|---|
| 고정 prefix (system + K_τ 2–6k + reference 이미지 4–12장) | 5k–15k (cache 가능) |
| 현재 관측 (이미지 4–6장 + descriptor/proprio 텍스트) | 2k–6k |
| history (최근 K스텝 이미지 + 압축 텍스트 로그) | 3k–20k (스텝에 따라 증가) |
| 입력 합계 | **10k–40k** |
| 출력 (xhigh reasoning + action JSON) | **2k–20k** (reasoning 토큰 양은 미확인) |

**비용 공식**

```
Cost ≈ N_calls × ( T_in,uncached·p_in + T_in,cached·p_cache + T_out·p_out )
```

예시로 **가상 단가** p_in=$5/M, p_cache=$0.5/M, p_out=$30/M(실제 GPT-6-Astra 가격 아님, 미확인)를 적용하고, 호출당 입력 20k(그중 15k cached), 출력 10k를 가정하면 다음과 같습니다.
- 호출당 비용 ≈ 5k×5e-6 + 15k×0.5e-6 + 10k×30e-6 ≈ $0.33
- 4만 회 기준 약 **$13k**이고, 가정 범위 전체로는 **$2k–$60k**입니다. 비용은 출력(reasoning) 토큰이 지배합니다.

**지연 (확실도: 낮음)**
xhigh reasoning의 호출당 지연을 10–90초로 가정하면, 에피소드당 50–100회 호출에 **약 8분–2.5시간**이 걸립니다. 600 에피소드를 순차 실행하면 수백에서 천 시간 이상이 필요하므로 대규모 병렬화가 필수였을 것입니다. 이 지연 수준에서는 실제 로봇의 폐루프 제어가 사실상 불가능하므로, L3과 L4를 정량적으로 뒷받침합니다.

### (h) 재구현 시 대체 전략 (참고)

이 환경에는 torch와 LLM SDK가 없고 GPT-6-Astra에 접근할 수 없습니다. 따라서 재구현은 (1) 정책 인터페이스(`ℓ, C_τ, o_t, H_t → a_t`)를 추상화하고, (2) mock/규칙 기반 정책으로 루프, descriptor, 스키마, 검증기를 먼저 검증한 뒤, (3) 실제 VLM 백엔드(사용 가능한 closed/open 모델)를 교체해 넣는 순서가 현실적입니다. 아래 ablation을 기본 실험 계획에 포함할 것을 권장합니다. 대상은 L7과 L8에서 지적한 2×2 촉각 ablation(offline 증류 시 촉각 사용 여부 × online 촉각 사용 여부), Task Skill 제거, reference 제거, history 길이, horizon 상한, reasoning effort입니다.

---

## 6. 미해결 이슈 (원문만으로 확인할 수 없는 것)

1. GPT-6-Astra의 정확한 버전과 스냅샷, 디코딩 설정, 이미지 토큰화 방식, 가격을 확인할 수 없습니다.
2. 상위 논문 GPT-as-Policy(arXiv:2609.38537)의 액션 포맷과 프롬프트 세부 사항을 확인하지 못했습니다.
3. 프로젝트 페이지에 프롬프트, Task Skill 원문, 코드가 공개되어 있는지 확인하지 못했습니다.
4. UniVTAC(arXiv:2602.10093)의 센서 모델(depth/marker 제공 방식), control 주기, 태스크별 step 예산, success 조건을 확인하지 못했습니다.
5. baseline 수치가 재실행 결과인지 인용인지, seed가 일치하는지 알 수 없습니다. E0 결과가 누락된 이유도 알 수 없습니다.
6. 10개 "offline validation" 데모의 실제 용도를 알 수 없습니다.
7. 데모의 초기 설정과 테스트 seed가 분리되어 있는지 알 수 없습니다.

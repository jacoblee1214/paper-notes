# 04. 논문과 공식 코드의 대조

대상은 공식 저장소 `showlab/ForceHarness`의 `main`, 커밋 `733407c`(2026-11-17)입니다. `repo/`에 얕은 clone을 만들어 **정적으로만** 읽었습니다. 코드를 실행하거나 import하지 않았고, 패키지를 설치하지 않았습니다. 저장소 안의 문서와 설정 파일도 데이터로만 읽었습니다.

## 1. 확인 범위

- 식 1–5(외력 추정, 리드, admittance, 질량, 마찰)와 부록 표 A2(상수), 실험 설정(시드, 예산, 힘 상한, 손상 기준)을 대조했습니다.
- 통계 검정 코드(McNemar, Fisher)와 실험 결과 파일의 존재를 확인했습니다.
- 실로봇 경로(`forceharness/real`, `robot_side/`)와 ForceBench 경로(`forceharness/backends/forcebench`)는 상수만 확인하고, 동작은 읽지 않았습니다.

## 2. 일치 항목

| 항목 | 논문 | 코드와 설정 | 판정 |
|---|---|---|---|
| 힘 부호 (식 1) | F = −(w_F − b) | 설정 `force_sign: -1.0`, 추정기 주석에 토크 부호 보정 기록 | 일치 |
| 리드 상한 (식 2) | 자유 30 mm, 보호 12 mm, 접촉 탐색 6 mm (표 A2) | `library.py`: `max_lead_free = 0.03`, `max_lead_guard = 0.012`, `max_lead_contact = 0.006` | 일치 |
| admittance 이득 (식 3) | push, pull, press turn에서 0.0045, slide와 search에서 0.0015 (표 A2) | push(`push`)와 pull(`pull`)은 `kf * 3.0`, slide와 search는 `self.kf` 0.0015 | 일치 (표의 행 대응은 텍스트로 분명하지 않음, 코드가 이 해석을 따름) |
| 힘 상한과 목표 상한 | 힘 상한 25 N, 목표 힘 상한 20 N (표 A2) | `f_cap = 25.0`, 목표 힘은 `min(F, 0.8 × f_cap)` = 20 N | 일치 |
| 제어 주기 | MuJoCo 0.05 s, ForceBench 1/15 s (표 A2) | MuJoCo `control_freq = 20`, ForceBench `control_freq = 15` | 일치 |
| 접촉 탐지 임계 | MuJoCo 3 N, ForceBench 2 N (표 A2) | 설정 `f_contact: 3.0` (MuJoCo) | 일치 (ForceBench 쪽은 읽지 않음) |
| 힘 잡음과 편향 | 틱당 0.5 N, 에피소드당 0.3 N (표 A2) | 설정 `force_noise_std: 0.5`, `force_bias_std: 0.3` | 일치 |
| 질량과 σ (식 4) | m = \|F̄_z\|/g, σ = √(s_z²/n + σ_b²)/g | `est_payload_and_com`: `sqrt((std/√n)² + bias²)/g`, `G = 9.81` | 일치 (코드는 모집단 표준편차 ddof 0을 씀. 큰 n에서 차이는 무시할 수준) |
| 마찰 (식 5) | 상대 오차 제곱 합 | `est_friction`: 두 표준오차의 상대 오차 제곱 합 | 일치 |
| 접촉 높이 | 탐지기 지연 보정 | `p_contact = p_ev − u × lag` | 일치 |
| 손상 기준 | 손잡이 15 N 초과 | `DL_FDAMAGE = 15.0` | 일치 |
| 에피소드 예산 | 40턴, 80호출, 300 s | 설정 `max_turns: 40`, `max_tool_calls: 80`, `max_sim_time: 300.0` | 일치 |
| ForceBench 예산 | 30턴, 60호출, 100 s, 25 N | `paper_settings.yaml`: `--max-turns 30`, `--max-tool-calls 60`, `--f-cap 25` | 일치 |
| 테스트와 개발 시드 | 테스트 9100–9119, 개발 9000–9019 (부록 C.5) | 설정 `seed_base: 9100`, 개발 시드는 문서 | 일치 |
| 학습 기준선 데이터 규모 | 과제당 약 200 (표 5) | 데이터 메타 기준 1,423개 (과제당 185–210) | 근사 일치 (D1) |
| 신뢰구간 | Wilson 95% | `forceharness/eval/metrics.py`의 `wilson` | 일치 |
| 비용 | 에피소드당 약 $0.28 (힘)과 $0.50 (무힘) | 설정 주석 (목록가 추정) | 일치 (추정치) |
| 실행 모델 설정 | Astra low, Fable high, Sol low (ForceBench) | 설정 파일과 `paper_settings.yaml` | 일치 |
| 표 6 분모 | Sol 70, 소형 125 | `force_form_sol.yaml`: 과제 7 × 시드 10 = 70. `force_form_cheap.yaml`: 과제 5 × 시드 5 × 모델 5 = 125 | 일치 (`03` §4) |
| 표 4 값 | 개발 시드 표 | 저장소 `docs/forcebench.md` 표와 `paper_settings.yaml` 설명 | 일치 |

## 3. 차이와 빈칸

차이는 심각도 순서가 아니라 번호 순서로 적었습니다. 심각도는 각 항목 끝에 표시합니다.

**D1. 학습 데모 수 (낮음).** 논문 표 5는 학습 기준선마다 "Demos 200"이라고 적고, 부록 G.1은 "성공한 전문가 에피소드를 모두 쓴다(Peg는 185)"고 합니다. 저장소 데이터 문서는 과제별 저장 수를 Weight 210, CoM 208, Wipe 207, Stack 206, Peg 185, Latch 197, Lock 210으로 적고 합은 1,423입니다. 200은 근사값입니다. 학습 데이터 양이 과제마다 다르므로 기준선끼리의 비교에 영향이 있습니다.

**D2. 공개 시점 문구 (낮음).** 부록 J.2는 "코드와 설정은 게재 시 공개한다(will be released upon publication)"고 씁니다. 그러나 저장소는 2026-11-16 생성, MIT 라이선스로 이미 공개되어 있고, 데이터셋도 Hugging Face에 있습니다. 논문 문구가 뒤처져 있습니다.

**D3. 통계 검정 절차가 재현되지 않음 (중간).** 논문은 Fisher 검정(실로봇), 짝 McNemar(개발 시드 p = 0.009와 0.001, 표 6 p = 0.023), Wilson 구간(부록 C.5)을 씁니다. 저장소에는 `wilson`만 있고, McNemar와 Fisher 코드는 없습니다. 에피소드별 결과 파일(`results.jsonl`)도 저장소에 없습니다. 그래서 짝 검정은 공개 집계로만 다시 계산할 수 있었습니다(`03` §2, §6).

**D4. 튜닝 시드 서술 (높음, ForceBench 해석에 직접 영향).** 본문 5.2절은 "tools and perception thresholds were tuned on eight of the ten seeds, and Stove Press and Nut Thread were revised after initial runs"라고 씁니다. 저장소 `docs/forcebench.md`는 더 구체적으로 적습니다. 도구와 인식 임계값은 시드 300–307에서 만들었고, 시드 300–309에서 처음 실행한 뒤 두 과제를 고쳤다고 합니다. Nut Thread는 `locate_object`와 지시문을 얻었고, Stove Press는 저항 토크를 추가하고 8–25도 창을 가진 측별 보고를 얻었는데, 그 창은 **시드 300–309의 기록 56회를 재생해서** 고른 것입니다. 따라서 본문의 "8/10 시드 튜닝"은 실제 노출보다 좁게 적혀 있습니다. 보고한 개발 시드 결과는 10개 시드 전체에 노출되어 있었습니다.

**D5. Astra의 Stove Press 행은 현재 코드로 재현되지 않음 (중간).** 저장소 설정 `paper_settings.yaml`과 문서는 Astra의 논문 Stove Press 행이 **이전 도구 결과**를 썼다고 적습니다(각 턴의 두 번째 절반 평균 저항 토크, 읽기 주석 없음). 이 버전은 현재 코드에 없습니다. 또 Weight Sort 에피소드 세 개는 6 m 간격으로 돌렸습니다. 논문 본문에는 이 내용이 없고, "revised after initial runs"만 있습니다. 표 4의 Astra 행 중 한 셀은 공개 코드로 재현할 수 없습니다.

**D6. 보류 시드 결과가 논문에 없음 (중간).** 시드 320–329에서 Astra는 62.5%(무힘)에서 82.5%(ForceHarness)로 올랐습니다. 저장소 `paper_settings.yaml`, 문서, 프로젝트 페이지에 있지만 논문 본문과 표 4에는 없습니다. 이 결과는 본문 주장보다 낮은 이득(17.5 → 20.0 pp)과 함께 보아야 하며, 본문이 개발 시드를 주된 근거로 쓰는 이유를 설명하지 않습니다.

**D7. Fable의 무힘 대조가 없음 (낮음).** 표 4와 표 5에서 Fable은 ForceHarness 행만 있고 무힘 행이 없습니다(표 5에는 있음). 저장소 설정은 "논문이 보고하지 않는 Fable 무힘 행"이라고 명시합니다. ForceBench에서 Fable 75.0%는 힘의 효과와 분리되지 않습니다.

## 4. 구현할 수 있는지 (판정)

**판정: 부분적으로 구현할 수 있습니다.** 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서는 힘 추정층과 물리량 식을 옮기는 일과 MuJoCo 안의 프리미티브 루프까지는 가능합니다. 시뮬 전체 재현은 설치 승인과 API 비용이 필요합니다. ForceBench, 실로봇, π0.5와 ForceVLA 학습은 맥에서 할 수 없습니다.

| 구성 | 맥에서 가능? | 필요한 자원 | 노력(추정) |
|---|---|---|---|
| 식 1–5의 추정층과 σ 전파 | 가능 (numpy, scipy 설치됨) | 합성 힘 신호 | 1–3일 |
| 프리미티브 루프 (식 2–3)를 1자유도 스프링 모델에서 재현 | 가능 (numpy) | 없음 | 2–4일 |
| HiddenPhys 전문가 상태 기계 (7과제) | 조건부: robosuite 1.5.1과 MuJoCo 3.2.7 설치 승인 필요. 이미지를 쓰지 않으므로 CPU로 가능 (속도는 측정하지 않음) | 설치 승인 | 1–2주 |
| HiddenPhys에서 LLM 루프 한 과제 (무힘과 힘) | 조건부: 모델 API 키와 비용 | 설치 승인과 API 비용. 설정 주석 기준 280 에피소드 약 $110 | 추가 1주 |
| ForceBench (Isaac Lab 3.0.0b2, Isaac Sim 6.0.1) | 불가 | NVIDIA GPU, 드라이버 550.90.07 이상 | - |
| 실로봇 (Franka FR3, Polymetis) | 불가 | 로봇, 안전 감시자 | - |
| π0.5 + 힘, ForceVLA 학습 | 불가 | GPU | - |
| Diffusion Policy 학습 (과제당) | 조건부 (데이터 수십 GB 추정, 메모리와 시간 미측정) | 데이터 다운로드 승인 | 1주 이상 |

**노력 요약 (추정).** 맥에서 "힘 추정 + 프리미티브 + 물리량" 층을 처음부터 옮기는 데 2–4주, 시뮬 한 과제를 LLM과 함께 돌리는 데 추가 1–2주입니다. 논문의 표 4와 표 5 전체를 재현하는 것은 이 맥의 범위 밖입니다.

## 5. 저자에게 물을 질문

1. 본문 "8 of 10 seeds"와 저장소 문서의 "시드 300–309 재생으로 Stove Press 창 선택"이 같은 실험을 말하는지, 그리고 Stove Press 창을 고른 뒤 재실행 결과가 있는지.
2. Astra의 Stove Press 행에 쓴 이전 도구 결과를 저장소에 함께 공개할 계획이 있는지.
3. 짝 McNemar와 Fisher 검정의 입력(에피소드별 성공 여부)과 스크립트를 공개할 수 있는지.
4. 표 6의 판정 정확도 분모가 75(소형)와 30(Sol)이 맞는지.
5. 학습 기준선 "Demos 200"을 과제별 실제 수(185–210)로 고쳐 쓸 계획이 있는지.

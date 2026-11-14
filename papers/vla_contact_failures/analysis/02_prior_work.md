# 선행연구와 관련 자료 (arXiv:2608.01402)

분석 대상은 arXiv v2입니다. 이 문서는 (1) 논문이 기대는 선행 계보, (2) 구성 요소별 출처, (3) 이 저장소의 관련 분석, (4) GitHub 저장소 표를 다룹니다. GitHub 정보는 2026-11-14에 공개 API로 확인했습니다.

## 1. 논문의 위치: 두 축

| 축 | 선행 흐름 | 논문의 입장 |
|---|---|---|
| 힘을 어디에 넣나 | 힘 융합 모듈(ForceVLA, TA-VLA, FAVLA), 보조 손실(CRAFT, FD-VLA) | 힘 입력 구조(시간 인지, 게이트)는 **그대로 두고** 학습 시간 샘플링도 함께 고친다 |
| 실패가 어디서 오나 | 힘이 빠져서 실패한다는 감각 문제 가정 | 실패를 정밀(샘플링)과 힘(입력 활용)으로 **둘로 나눈다** |

§2 Related Work는 접촉 중심 조작을 고전 제어(Whitney 1982, Khatib 1987, Lefebvre 2005, Chhatpar 2001)와 학습 기반(FORGE, 시각-힘 융합, 촉각-시각 융합)으로 나눕니다. 힘 증강 VLA는 세 가지 난제로 나눕니다. ① 힘이 행동 선택에 실제로 영향을 주게 하기, ② 행동 공간 해상도(느린 VLM과 빠른 행동 모듈), ③ 모달리티 불균형(힘이 시각에 묻힘). 논문은 ①–③ 어디에도 속하지 않는 **학습 샘플링**을 추가합니다(§2.2 마지막 문단).

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 논문에서의 쓰임 | 확인 |
|---|---|---|---|
| Flow matching, 정렬된 보간 | Lipman et al. 2022 (`lipman2022flow`) | §3 식 (1) | 인용만 확인 |
| Beta(1.5, 1) 시간 스케줄 | π0.5 (Black et al. 2025, `black2025pi05`), π0 (`black2024pi0`) | 기존 샘플링. 코드 `jax.random.beta(1.5, 1)` | 코드 확인 (`04` D8) |
| Logit-Normal 시간 샘플링 | Esser et al. 2024, SD3 (`esser2024scaling`) | 제안의 LN 식 (2)–(3) | 식 일치 확인 (`04` D8) |
| AdaRMS 시간 조건 | π0.5 (adaRMSNorm) | 힘 Δγ를 스케일에 더함 | 코드 확인 (`04` D12) |
| 인과 TCN | Bai et al. 2018 (`bai2018tcn`) | 힘 이력 인코더, 4 블록(확장 1, 2, 4, 8), 은닉 64 | 코드 확인 (`04` D10) |
| 임피던스·운동 공간 제어 | Khatib 1987, Lee 2020 (`lee2020makingsense`) | 부록 A.3 제어기 | 로봇 쪽이라 코드 없음 |
| 힘 필터 | One-euro filter (Casiez 2012) | F/T 전처리 | 코드 없음 |
| 기준선 ForceVLA | Yu et al. 2025 (`yu2025forcevla`) | π0.5 위 재구현 | 이름이 같은 저장소(`ft-robotic/ForceVLA`)가 있으나 공식 여부 미확인. 재구현 코드는 없음 |
| 기준선 TA-VLA | Zhang et al. 2025 (`tavla2025`, CoRL 2025) | π0.5 위 재구현 | 공식 저장소 있음, 재구현 코드는 없음 |
| 힘 증강 비교 대상 | FAVLA (`li2026favla`), CRAFT (`zhang2026craft`, arXiv:2602.12532), FD-VLA (`zhao2026fdvla`, arXiv:2602.02142) | 소개만, 실험 비교는 없음 | 저장소 미확인 |
| GR00T N1 | Bjorck et al. 2025 (`bjorck2025gr00t`, arXiv:2503.14734) | 두 번째 VLA 계열 실험 (v2에 추가) | 현재 저장소는 N1.7 (`02` §4) |

## 3. 이 저장소의 관련 분석

아래 폴더는 같은 질문(접촉·힘을 VLA가 실제로 쓰는가, 힘 게이트, 촉각 잔차, 시간 샘플링)을 다룹니다. 연결은 참고용입니다. 이 분석의 판단은 각 폴더의 판단과 독립입니다.

| 폴더 | 논문 | 연결 지점 |
|---|---|---|
| restacvla | ResTacVLA (arXiv:2607.03387) | π0.5에 촉각 잔차 토큰을 붙이고 불확실성 σ로 게이트. **힘 게이트 설계를 비교할 때** 좋은 짝입니다. 공식 코드 미공개 |
| contact_disassembly | CONTACT (arXiv:2603.08560) | 시각 단독 대 촉각 영상 대 촉각 힘장. **힘 입력이 실제로 정보를 주는가**라는 질문이 이 논문의 노이즈 대체 실험과 같습니다 |
| tactilereflex | TactileReflex (arXiv:2605.23568) | 그리퍼 반사 제어(접촉 보호). 힘 임계와 접촉 판정의 **임계값 과적합** 문제를 다룹니다. FACT의 δ=0.5 N 설정을 같은 눈으로 볼 수 있습니다 |
| stiffness_copilot | Stiffness Copilot (arXiv:2603.14068) | 접촉 안전 지표(최대 접촉력)와 조건 교란 문제. **힘 조건의 효과를 강성 같은 다른 변수와 분리**하는 문제를 참고합니다 |
| beyond_implicit_force | Beyond Implicit Force (arXiv:2607.14578) | 원격조작 데이터의 암묵 힘(δ)을 실제 토크 입력으로 바꾸는 2×2 실험. **힘 입력 대체 실험의 설계**를 참고합니다 |
| imle_vla | IMLE-VLA (arXiv:2609.10915) | 흐름/확산 샘플링 효율과 행동 헤드 비용. **시간 샘플링 분포**가 학습과 추론 비용에 미치는 영향을 같이 생각할 때 참고합니다 |

## 4. GitHub와 코드 (2026-11-14 확인)

공개 API(`api.github.com/repos/...`, 읽기 전용)로 확인했습니다. 라이선스와 최근 push는 API 값입니다. "대조 commit"은 실제로 받아서 읽은 것만 씁니다.

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | 별 | 최근 push | 대조 commit |
|---|---|---|---|---|---|---|
| [stanford-iprl-lab/fact](https://github.com/stanford-iprl-lab/fact) | FACT 공식 코드(JAX, openpi 포크형). 프로젝트 페이지의 "Code" 링크가 가리킴 | **공식** | Apache-2.0 | 20 | 2026-10-26 | `2ccfe8f` (2026-08-03, shallow clone을 `repo/`에 받음, 정적 읽기만) |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0, π0.5 원 구현(FACT 코드의 바탕) | 구성 요소 공식 | Apache-2.0 | 14,646 | 2026-08-24 | 대조 안 함 |
| [ZZongzheng0918/TA-VLA](https://github.com/ZZongzheng0918/TA-VLA) | TA-VLA (CoRL 2025) 공식 | 공식(저장소 설명에 CoRL 2025 명시) | Apache-2.0 | 126 | 2025-10-25 | 대조 안 함 |
| [ft-robotic/ForceVLA](https://github.com/ft-robotic/ForceVLA) | ForceVLA. 이름이 같은 저장소 가운데 가장 많이 별을 받음 | 공식 여부 README 미확인 | Apache-2.0 | 119 | 2026-01-27 | 대조 안 함 |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | GR00T 공식. 현재 **N1.7** 소개 | 공식 | Apache-2.0 | 8,371 | 2026-11-10 | 대조 안 함. 논문은 N1(arXiv:2503.14734)을 씀 |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | SmolVLA 구현과 데이터 도구 | 구성 요소 공식 | Apache-2.0 | 28,873 | 2026-11-14 | 대조 안 함 |
| FAVLA, CRAFT, FD-VLA | 검색 결과 없음 (이름 검색 0건) | 찾지 못함 | — | — | — | — |

**요약:**
- 공식 코드는 있습니다(`stanford-iprl-lab/fact`). 논문 v2 부록은 "BOM과 STL 파일은 코드와 함께 공개 예정"이라고 했는데, 저장소에는 BOM·STL이 없습니다(`04` D14).
- 저장소 안의 `README.md`는 openpi 원문입니다. FACT 설명은 `examples/fact/README.md`와 `FACT_CHANGESET.md`에 있습니다.
- 공식 코드의 설정 이름(`pi05_ln_fact_delta_low_mem_finetune`, `pi05_full_fact_fact_delta_low_mem_finetune`)은 논문의 LN 전용과 전체 FACT에 대응합니다.
- 비교 대상 기준선(ForceVLA, TA-VLA)은 논문이 **π0.5 위에서 다시 구현**했습니다. 저자 저장소의 구현은 π0 기반이므로 직접 비교되지 않습니다.
- 대조 세부는 `analysis/04_code_verification.md`입니다.

## 5. 이름이 같은 저장소 주의

- `ForceVLA`로 검색하면 세 저장소가 나옵니다. `ft-robotic/ForceVLA`(별 119), `tshiamor/ForceVLA`(별 3), `leversparrow/ForceVLA`(별 3). 공식 여부는 README로 확인하지 않았으므로, 표에서는 "이름 일치, 공식 여부 미확인"으로 둡니다.
- 논문 v2에는 ForceVLA의 공식 링크가 없습니다.

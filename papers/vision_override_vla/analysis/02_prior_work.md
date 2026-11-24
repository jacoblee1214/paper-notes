# 02. 선행연구와 관련 저장소

이 문서는 CAG와 LIBERO-CF의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드와 모델이 어디에 있는지를 정리합니다. 대괄호 번호는 논문(v3) 참고문헌 번호입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 비조건 예측과 조건 예측을 외삽 | Classifier-free guidance [66] (Ho & Salimans 2022) | CAG의 직접 출발점입니다. CFG는 하나의 모델을 조건 드롭아웃으로 함께 학습하지만, 표 4의 전략 2(언어 드롭아웃 미세조정)는 CF-Spatial 접촉률 F 9.5%로 가장 나빴다고 논문이 보고합니다(§6.2) |
| 행동 공간에서 안내 | CAG 자체. π0류 흐름 모델과 OpenVLA-OFT류 회귀 헤드 둘에 적용(§5.1) | 확산·흐름 모델의 속도장이 아니라 최종 행동 청크를 외삽합니다. 이 분석의 토이(`03` §5)가 두 방식의 차이를 보입니다 |
| 언어 편향·모달리티 불균형 | Counterfactual VQA [44], CF-VLM [49], MLLM 모달리티 편향 [50], 모달리티 방해 진단 [48] | VQA와 MLLM의 반사실 증강, 재가중, 정규화(§2). VLA에서는 "아직 덜 연구됨"이라고 씁니다 |
| 언어 따르기 평가·증강 | LIBERO-Plus [53], LIBERO-PRO [55], CAST [54], From Intention to Execution [56], Counterfactual VLA [62] | 같은 과제 안의 문장 바꿔 쓰기 또는 데이터 증강입니다. LIBERO-CF는 **다른 과제**를 지시하는 점이 다르다고 §3.2가 구분합니다. v1 관련 연구는 LIBERO-Plus를 같은 과제 안의 문장 바꿔 쓰기로, CAST와 Counterfactual VLA를 내비게이션과 주행 영역의 데이터 증강으로 서술했습니다 |
| 베이즈 분해와 언어 안내 | Bayesian-VLA [64], Seeing to act, prompting to specify [61], Stable language guidance [63] | 논문이 §3.1에서 시각 지름길 관련으로 인용합니다. 제목으로 보아 [61, 64]는 베이즈 분해를 정책 구조에 쓰는 작업입니다. 논문은 식 1을 "개념적 관점"으로만 쓰고 이 계열과 방법을 비교하지 않습니다 |
| 테스트 시점 검증 | Scaling verification [57] | 행동 후보를 검증해 고르는 방식. CAG는 후보를 만들지 않고 외삽합니다(§2) |
| 벤치마크 기반 | LIBERO [65] (Liu 외 2023) | 장면 배치와 초기 상태를 그대로 쓰고 지시와 성공 조건만 새로 정의합니다(§4) |
| 평가 대상 VLA | OpenVLA-OFT [51], π0 [5], π0.5 [52], X-VLA [29] | 공식 LIBERO 체크포인트를 그대로 씁니다(부록 B.1) |
| 실세계 기반 | DROID 플랫폼 [59], π0.5-DROID | 같은 하드웨어 구성과 사전학습 체크포인트에서 미세조정합니다(§7.1) |

**가까운 대안과의 차이.** (1) 데이터 증강(CAST, Counterfactual VLA)은 훈련 데이터를 바꾸고 CAG는 추론만 바꿉니다. 논문 표 5는 두 방법이 함께 쓰일 수 있음을 보입니다(시연 1–2개 추가에서 CAG가 더해 주는 성공 수, `03` V11). (2) 언어 드롭아웃 미세조정(CFG의 정석 학습)과 비교해 행동 외삽 + 별도 VA가 낫다고 표 4가 보입니다. (3) 베이즈 분해 계열 모델은 구조를 바꾸므로 같은 평가에서 직접 비교되지 않습니다(논문에 비교 없음).

## 2. 구성 요소별 출처 (코드 쪽)

| 구성 요소 | 공식 저장소 안의 출처 | 비고 |
|---|---|---|
| 평가 루프 | `eval/main_cf.py`(openpi LIBERO 예제 `main.py`에서 파생된 형태) | 조건별 접촉·성공을 에피소드마다 JSON으로 저장 |
| π0·π0.5 서빙과 CAG | `openpi-cf/`(openpi 포크, Apache-2.0) 안의 `scripts/serve_policy_TF.py`, `serve_policy_VA.py`, 학습 설정 `*_libero_unconditioned` | VA 가중치는 Hugging Face |
| OpenVLA-OFT 서빙과 CAG | `openvla-oft-cf/`(OpenVLA-OFT에 덮어쓰는 파일, MIT) 안의 `serving/serve_openvla_oft_{TF,VA}.py`, `vla-scripts/finetune.py`(`empty_language` 옵션) | 업스트림 저장소에 복사해 사용 |

## 4. 코드와 저장소 (GitHub 표)

| 저장소·자원 | 관계 | 라이선스 | 별·갱신 | 대조 기준 | 상태 |
|---|---|---|---|---|---|
| [yuffish/VLA-CF](https://github.com/yuffish/VLA-CF) | 공식 구현 (LIBERO-CF와 CAG). arXiv와 프로젝트 페이지가 가리킴 | MIT (루트 LICENSE, 하위 `openpi-cf`는 Apache-2.0, `openvla-oft-cf`는 MIT) | 16★, 포크 2, 마지막 push 2026-10-30 | `8280a500a01f7f7d45efe202cb416ca362aa8c1a` (2026-10-30, README 갱신). 첫 커밋 2026-04-23 | 정적 읽기만 (`04`). 실행, 설치, import 안 함 |
| [vla-cf.github.io](https://vla-cf.github.io/) | 프로젝트 페이지 | — | — | v1의 주소는 `vla-va.github.io` | 읽기만 |
| [Hugging Face VLA-CF](https://huggingface.co/VLA-CF/models): `openvla-oft-libero-VA`, `pi0-libero-VA`, `pi05-libero-VA` | VA 가중치(LIBERO 미세조정, 언어 없음) | 모델 카드에 라이선스 표기 없음 | 최종 수정 2026-04-23 | 파일 목록 메타데이터만 확인: 15.9 GB, 12.0 GB, 12.4 GB | 받지 않음 |
| [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | LIBERO-CF의 기반 | MIT | 2,567★, 2025-03-15 | — | 받지 않음 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0·π0.5 코드, `openpi-cf`의 업스트림 | Apache-2.0 | 14,747★, 2026-08-24 | — | 받지 않음 |
| [moojink/openvla-oft](https://github.com/moojink/openvla-oft) | OpenVLA-OFT 코드, `openvla-oft-cf`가 덮어쓰는 업스트림 | MIT | 1,450★, 2025-09-09 | — | 받지 않음 |
| [2toinf/X-VLA](https://github.com/2toinf/X-VLA) | 부록 B의 X-VLA | Apache-2.0 | 762★, 2026-10-20 | — | 공식 저장소에 X-VLA용 CAG 코드는 없음(`04` D13) |

- 별 수와 푸시 날짜는 2026-11-24 `gh api`로 확인했습니다.
- 같은 이름이 비슷한 `EmbodiedAI-RoboTron/CF-VLA`(arXiv:2604.24622)는 다른 논문(흐름 기반 VLA의 행동 전문가)이라 관련이 없습니다.
- 저장소 전체는 git 이력 포함 약 218 MB(GitHub API 크기)로, 대부분 LIBERO 에셋(질감, 메시)입니다. 이 분석에서는 에셋과 초기 상태 파일을 뺀 얕은 복제(약 5 MB)만 읽었습니다.

## 5. 참고: 이 논문의 주장이 기대는 선행 결과

- **LIBERO가 언어에 약하게 의존한다는 증거.** 표 1의 V-only 결과 외에도 별도 감사 연구(arXiv:2606.04233)가 같은 방향의 관찰을 보고합니다. LIBERO에서 언어 인코더가 없는 작은 프로브가 최고 보고치와 1 %p 안쪽이라는 결과입니다. 이 논문은 "시각만으로 69.1%(OpenVLA-OFT)"를 반사실 실패의 근거로 쓰고, 감사 연구는 같은 종류의 관찰을 "벤치마크가 무엇을 재는가"의 문제로 읽습니다.
- **CFG의 일반적 성질.** 안내 배율을 키우면 조건 충실도가 오르고 품질이 떨어진다는 알려진 결과(Ho & Salimans [66])가 그림 4의 접촉률 증가와 성공률 감소 모양과 맞습니다.

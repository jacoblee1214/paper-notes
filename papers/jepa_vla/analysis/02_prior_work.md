# JEPA-VLA 선행연구와 관련 연구

## 1. 계보

| 줄기 | 대표 연구 (논문 참고문헌 기준) | 이 논문에서의 역할 |
|---|---|---|
| JEPA | LeCun 2022(개념), I-JEPA(Assran 2023), V-JEPA(Bardes 2024), V-JEPA 2(Assran 2025, arXiv:2506.09985), 직관 물리 출현(Garrido 2025) | 예측 임베딩의 원천. V-JEPA 2를 고정 인코더로 씀 |
| VLA | RT-2(Zitkovich 2023), OpenVLA(Kim 2025b), OpenVLA-OFT(Kim 2025a), GR00T N1(Bjorck 2025), SpatialVLA(Qu 2025), CoT-VLA(Zhao 2025) | 기준 VLA. OpenVLA-OFT와 자체 기본 VLA |
| 세계 모델 + 행동 | WorldVLA(Cen 2025b), RynnVLA-002(Cen 2025a), UniPi류 Du 2023, Vidar(Feng 2025) | 비디오 생성 prior를 정책으로 쓰는 흐름. WorldVLA는 LIBERO-plus 비교 대상 |
| 시각 표현 (로봇) | R3M(Nair 2023), VIP(Ma 2023), MVP(Xiao 2022) | 로봇 전용 사전학습 표현. 논문은 직접 비교하지 않음 |
| 시각 표현 (범용) | DINO/DINOv2/DINOv3(Caron 2021, Oquab 2024, Siméoni 2025), CLIP(Radford 2021), SigLIP(Zhai 2023), SigLIP 2(Tschannen 2025) | 비교 대상. 논문이 "문제의 원인"으로 지목 |
| 벤치마크 | LIBERO(Liu 2023), LIBERO-plus(Fei 2025), RoboTwin 2.0(Chen 2025), CortexBench(Majumdar 2023) | 평가 환경 |

논문이 스스로 밝힌 위치는 "VLA의 시각 이해를 개선하는 방법들(3D, 시각 CoT, 깊이, 명시적 인지 모듈)은 복잡하거나 효과가 불확실하므로, 더 단순하고 일반적인 방법을 찾는다"입니다(Sec. 5, Vision-Language-Action Models).

## 2. 가장 가까운 대안

1. **V-JEPA 2 자체를 정책 백본으로 쓰는 방법.** JEPA-VLA는 VLM을 유지하고 V-JEPA 2를 보조 표현으로 붙입니다. 반대로 V-JEPA 2를 주 백본으로 쓰는 방법은 논문에서 다루지 않습니다.
2. **이미지 기반 SSL 또는 언어-이미지 표현을 추가하는 방법.** Table 7이 이것을 대조합니다. 다만 "교체"이고 "추가"가 아닙니다.
3. **비디오 생성 모델을 정책 prior로 쓰는 방법 (Vidar, UniPi류, WorldVLA).** 관련 흐름이지만 목적이 다릅니다. JEPA-VLA는 생성이 아니라 예측 임베딩을 쓰므로 픽셀 복원 부담이 없습니다. 논문 Sec. 2는 이 차이를 "생성 목적은 예측 불가능한 세부까지 학습한다"로 설명합니다.
4. **VLA-JEPA (Sun 2026, arXiv:2602.10098).** 같은 시기에 나온 논문으로, 역시 V-JEPA 2 인코더를 씁니다. 공식 코드가 있으므로(`github.com/ginwind/VLA-JEPA`) 같은 방향의 구현 비교 대상입니다. 이 저장소의 `vla_jepa/` 폴더가 따로 분석하고 있습니다(이 폴더는 수정하지 않았습니다). 두 논문은 제목이 비슷하지만 방법이 다릅니다. JEPA-VLA는 게이트드 교차주의로 붙이고, VLA-JEPA는 잠재 행동 표현을 씁니다(`vla_jepa/README.md` 참조).

## 3. 이 저장소의 관련 분석

아래 연결은 이 논문과 관련된 부분만 적었습니다. 각 폴더의 결론은 그 폴더에서 확인하십시오.

| 폴더 | 논문 | 이 논문과의 연결 |
|---|---|---|
| `vla_jepa/` | VLA-JEPA (arXiv:2602.10098) | 같은 V-JEPA 2 인코더를 쓰는 VLA. 공식 코드와 가중치가 있어 구현 비교 기준 |
| `dreamzero/` | DreamZero, World Action Models (arXiv:2602.15922) | 영상 확산 모델을 행동과 함께 예측. "비디오 prior를 정책에" 흐름 |
| `agentic_wam/` | AW-0 (arXiv:2610.30136) | 영상과 행동을 함께 denoise하는 WAM |
| `openwam/` | OpenWAM (arXiv:2609.07398) | WAM 설계 선택을 부품별로 비교한 연구. 비디오 prior의 설계 공간 |
| `dreamx/` | DreamX (arXiv:2611.05377) | 영상 확산 모델의 latent action 토큰. 표현을 행동 인터페이스로 씀 |
| `one_step_vla/` | One-Step VLA (arXiv:2610.30735) | 행동 생성 방식 비교. 이 논문의 대상(표현)과 직교 |
| `flashvla/` | FlashVLA (arXiv:2608.27384) | VLA 추론 속도. 추가 인코더의 지연 비용을 따질 때 참고 |
| `robofqe/` | RoboFQE (arXiv:2610.18699) | 오프라인 정책 평가. LIBERO 류 평가의 분산 문제와 연결 |
| `benchmark_audit/` | arXiv:2606.04233 | 벤치마크 점수 해석. LIBERO 계열 점수의 신뢰도 점검 |
| `hydra0/`, `dextouch_wm/`, `insertion_wm/`, `cross_embodiment_wm/` | 세계 모델 계열 | 예측 기반 표현이 제어에 쓰이는 다른 사례 |
| `gr00t2_whitepaper/` | GR00T 2 백서 | 대형 VLA 백본의 맥락. 기업 백서이므로 동료 심사 없음 |
| `motion_latent_action/`, `adavla/`, `imle_vla/`, `vla_contact_failures/` | LIBERO·RoboTwin을 쓰는 VLA | 같은 벤치마크에서의 수치 맥락 |

## 4. 구성 요소별 출처

| 구성 요소 | 출처 | 논문에서 새로운 부분 |
|---|---|---|
| V-JEPA 2 인코더 (고정) | Assran 2025 | 아님 (사용만) |
| 게이트드 교차주의 | Flamingo (Alayrac 2022) | 로봇 VLA에 적용하고, 학습률 비율을 정함 |
| 기본 VLA (Chameleon + 256 토큰 행동) | Chameleon (Team 2024), WorldVLA (Cen 2025b) | 기본 VLA 구성은 기존 것 |
| OpenVLA-OFT 설정 | Kim 2025a | 설정은 기존 것. 결과 재현 |
| 표현 분석 방법 (프로브) | 논문 자체 (App. A) | 프로브 설계는 논문이 만듦 |

## 5. 이 논문이 기여하는 것과 아닌 것

- **기여로 볼 수 있는 것:** 로봇 VLA의 시각 입력으로 비디오 예측 임베딩을 붙이는 간단한 방법과, LIBERO 계열에서 나온 일관된 방향의 결과.
- **기여로 보기 어려운 것:** 새로운 학습 목적, 새로운 아키텍처 원리. 게이트드 교차주의와 V-JEPA 2 모두 기존 것입니다.
- **아직 증명되지 않은 것:** "예측 목적이 무관 요인을 버린다"는 기전. 이 논문의 실험은 상관을 보여 주지만 기전을 직접 보이지는 않습니다(`04_claim_verification.md` C3, C4).

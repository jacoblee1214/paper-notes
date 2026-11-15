# 02. 선행 연구와 가까운 대안

이 논문은 **동결한 VLA 위에 시뮬레이션에서만 학습한 잔차 정책을 얹고, 그 잔차의 관측을 물체 포즈로 제한해 zero-shot 전이를 얻는다**는 주장입니다. 아래는 그 주장을 떠받치는 계보, 가까운 대안, 구성 요소의 출처입니다. 모든 항목은 논문 본문과 참고문헌 번호(§, [번호])를 따랐고, 원문을 읽지 않은 항목은 "확인 안 함"이라고 적었습니다.

## 1. 선행 연구 계보

| 계열 | 대표 연구 (논문 참고문헌) | 이 논문이 주장하는 차이 |
|---|---|---|
| 잔차 RL의 시작 | Residual policy learning [14] (Silver 2018), Residual RL for robot control [15] (Johannink 2019) | 기반 정책 위의 보정이라는 틀은 같음 |
| 특권 상태 잔차 + 증류 | ResiP [16] (Ankile 2024, CoRL) | 특권 상태로 학습 후 이미지 학생으로 증류. 증류에서 성능이 떨어짐 |
| 시뮬레이션 + 재구성 | RialTo [22] (Torne 2024) | 작업 공간 3D 스캔과 점군 정책. 과제마다 재구성 비용이 큼 |
| 이미지 잔차, 실로봇 RL | ResFiT [17] (Ankile 2025) | 실로봇에서 학습. 안전 장치와 리셋이 필요 |
| VLA 자기 개선 + 잔차 RL | PLD [18] (Xiao 2025) | 실로봇 RL과 보상 분류기가 필요. 세 단계 파이프라인 |
| VLA 증류 (시뮬레이션) | RPD [23] (Jülg 2025) | 시뮬레이션에서 VLA 지식을 RL 학생에 증류. 실로봇 배포는 없음 |
| 피드백 기반 교정 | TRANSIC [32] (Jiang 2024, CoRL) | 온라인 사람 교정으로 잔차 학습 |

이 논문의 위치는 "이미지 잔차의 시각 격차"와 "특권 상태 잔차의 증류 손실"을 함께 피하는 것입니다(§1, §2). 가장 가까운 대안은 ResiP와 RPD입니다. 둘 다 시뮬레이션 잔차라서 비교 대상이 됩니다. 논문은 ResiP 방식을 Table 2(b)의 "Distillation-based" 베이스라인으로 직접 구현해 비교했습니다. RPD는 실로봇 배포가 없어서 실험 비교가 없습니다.

## 2. 구성 요소별 출처

| 구성 요소 | 논문에서 쓴 것 | 출처와 상태 |
|---|---|---|
| 기반 VLA | GR00T-N1.5 [8], π0.5 [7] | GR00T 공개 저장소(NVIDIA/Isaac-GR00T, Apache-2.0, `gh` 메타데이터 확인). π0.5는 openpi(Physical-Intelligence/openpi, Apache-2.0). 논문은 두 모델을 "open-source"로 씀 |
| 물체 포즈 추정 | FoundationPose [20] | NVlabs/FoundationPose 저장소(라이선스 "Other", 확인 필요). 학습용 가중치 사용 조건은 읽지 않음 |
| 마스크 | SAM 2 [21] | facebookresearch/sam2 (Apache-2.0, `gh` 확인) |
| 잔차 학습 알고리즘 | TD3 [38] | sfujim/TD3 (MIT, 마지막 활동 2023) |
| 시뮬레이션 | MuJoCo [39] | google-deepmind/mujoco (Apache-2.0, `gh` 확인) |
| 시뮬레이션 재생 | MimicGen [34] | NVlabs/mimicgen (라이선스 "Other"). 논문은 "replaying the same teleoperation actions"라고만 씀 |
| 시연 | 원격조작 30회/과제 [40] | "What matters in learning from offline human demonstrations" [40], 시연 데이터는 공개 여부 확인 안 함 |
| 도메인 무작위화 | Tobin 2017 [25], Peng 2018 [26] | 논문 A.3에서만 보조 베이스라인에 씀 |

## 4. GitHub와 코드 자료 (확인 2026-11-15)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| 프로젝트 페이지 | https://www.microsoft.com/en-us/research/articles/object-centric-residual-rl/ | - | - | 영상, 소개, 학회 표기(CoRL). 코드·가중치 링크 없음 |
| arXiv 페이지 | https://arxiv.org/abs/2606.18953 | CC BY 4.0 | - | "Code, Data & Media" 항목에 공식 저장소 없음 |
| 공식 코드 | 찾지 못함 | - | - | `gh search repos`로 제목·arXiv 번호 검색, 결과 없음. 논문 본문에 코드 공개 문구 없음 |
| 공식 데이터·가중치 | 찾지 못함 | - | - | 없음 |
| GR00T-N1.5 | https://github.com/NVIDIA/Isaac-GR00T | Apache-2.0 | 별 8,373, 2026-11-10 | 기반 VLA (논문 [8]) |
| π0.5 (openpi) | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 별 14,648, 2026-08-24 | 비교용 기반 VLA (논문 [7]) |
| FoundationPose | https://github.com/NVlabs/FoundationPose | Other (확인 필요) | 별 3,700, 2026-04-29 | 포즈 추정 (논문 [20]) |
| SAM 2 | https://github.com/facebookresearch/sam2 | Apache-2.0 | 별 20,149, 2026-05-30 | 마스크 (논문 [21]) |
| MuJoCo | https://github.com/google-deepmind/mujoco | Apache-2.0 | 별 17,981, 2026-11-13 | 시뮬레이션 (논문 [39]) |
| MimicGen | https://github.com/NVlabs/mimicgen | Other (확인 필요) | 별 670, 2026-10-21 | 시뮬레이션 재생 (논문 [34]) |

공식 코드가 없으므로 `repo/`에 아무것도 받지 않았습니다. 코드 대조 대신 `analysis/04_claim_verification.md`에서 주장별 검증을 했습니다.

## 5. 가까운 대안과 비교 질문

이 논문과 비교하면 답해야 할 질문입니다. 실험으로 답하지 않았습니다.

1. 같은 시뮬레이션에서 **포즈 기반 잔차와 이미지 기반 잔차를 같은 학습 예산으로** 비교하면 격차가 얼마나 남는가 (논문은 시각 격차를 주장하지만, 이미지 베이스라인에 사실적 렌더링을 쓰므로 학습 조건이 다릅니다).
2. 증류 베이스라인은 ResiP의 "특권 상태 학습 후 증류"를 그대로 따랐는가. 증류 손실의 크기는 얼마인가.
3. RPD처럼 시뮬레이션에서 VLA 지식을 증류하는 방식과 잔차 방식의 실로봇 차이는 얼마인가 (RPD는 실로봇 결과가 없음).

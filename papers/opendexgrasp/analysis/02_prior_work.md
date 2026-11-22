# 02. 선행연구, 구성 요소의 출처, GitHub 자료

## 1. 선행연구 계보

이 논문은 세 줄기가 만나는 자리에 있습니다.

1. **대규모 합성 파지 데이터와 최적화 기반 생성.** DexGraspNet(ICRA 2023, [1]), DexGraspNet 2.0(CoRL 2024, [2]), BODex(ICRA 2025, [5]). 논문은 OpenDex-Scale의 자세를 BODex로 만듭니다.
2. **생성 모델로 파지 자세 만들기.** GenDexGrasp([6], ICRA 2023), UGG([8]), DexDiffuser([9]), Dexterous grasp transformer([10], CVPR 2024), score 기반 파지 원시 동작([11], NeurIPS 2023). 논문은 여기에 흐름 매칭([38])과 π0 식 VLA 구조([27])를 붙입니다.
3. **기능과 언어에 따른 파지.** 언어 조건 파지(Grasp as you say [15], SemGrasp [16]), 개방 어휘 affordance([17], LASO [20]), 부분 추론(PartSLIP [21], Embodied3DBench [22]), 시연 기반 기능 파지(DexCap [23], DexMV [24]), 접촉 지도(GenDexGrasp [6]), 부품 사전 확률 기반 일반화(G-DexGrasp [14]).

**이 논문이 채우는 자리:** 앞의 두 줄기는 기능 의미를 잘 다루지 못하고, 셋째 줄기는 affordance를 두 단계 파이프라인으로 씁니다. 저자는 affordance를 **학습 시 보조 지도**로만 쓰는 구조로 두 줄기를 합쳤습니다(§1, §2).

## 2. 구성 요소별 출처 (논문 인용 기준)

| 구성 | 출처 (논문 번호) | 비고 |
|---|---|---|
| 자세 파라미터화 6D 회전 | Zhou 외 CVPR 2019 [35] | 표준 |
| 흐름 매칭 | Lipman 외 ICLR 2023 [38] | 확산 대신 흐름 |
| 액션 헤드 구조 (π0 식) | π0 [27], Diffusion Policy [28] | DiT와 교차 어텐션 조합은 저자 설계 |
| 의미 인코더 | Qwen3-VL [40] (4B Instruct) | 표 3의 핵심 변수 |
| 기하 인코더 | PointNet++ [41] | 표준 |
| 기준선 특징 | CLIP [26] | 기준선과 절제 행에서 사용 |
| 점 단위 손실 | focal loss, Dice loss | 표준 |
| 물체 원천 | Omni6DPose [36] (ECCV 2024) | 범주 수준 좌표계가 있어 전이에 필요 |
| 자세 전이 | OakInk 방식의 밀집 대응 [37] | DeepSDF와 SAPIEN 정리는 저자 절차 |
| 평가 판정자 | GPT-5 (OpenAI 시스템 카드 [39]) | 판정 설정과 샘플 수 미기재 |
| 최적화 변형 | GenDexGrasp의 affordance 후 최적화 [6] | 저자가 만든 대조군 (부록 D.1) |

## 4. GitHub와 자료 표

확인일 2026-11-22. 별 수와 마지막 push는 GitHub API(공개 정보)에서 가져왔습니다. 라이선스는 API가 감지한 SPDX 값이고, 감지되지 않으면 "없음"으로 적었습니다.

| 자료 | 링크 | 라이선스 | 별 / 마지막 push | 상태 |
|---|---|---|---|---|
| **OpenDexGrasp 공식 저장소** | https://github.com/opendexgrasp/OpenDexGrasp | 없음 (감지 안 됨) | 5 / 2026-06-28 | **README 한 줄("coming soon!")과 빈 README 템플릿만 있음.** 커밋 2개(초기 커밋과 플레이스홀더 수정). 코드, 데이터, 가중치 없음. clone은 하지 않음 |
| 프로젝트 페이지 저장소 | https://github.com/opendexgrasp/opendexgrasp.github.io | 없음 | 0 / 2026-11-10 | 페이지 소스. 논문 요약, BibTeX, 코드 링크 |
| 프로젝트 페이지 | https://opendexgrasp.github.io/ | - | - | 논문 요약과 "CoRL 2026 Oral" 표기. 공개 데이터 없음 |
| DexGraspNet 2.0 (기준선 계열) | https://github.com/PKU-EPIC/DexGraspNet2 | 감지 안 됨 | 165 / 2025-01-23 | 이름과 조직으로 보아 논문 [2]의 공식 저장소로 보이나 README는 읽지 않음 |
| BODex (OpenDex-Scale 합성) | https://github.com/JYChen18/BODex | NOASSERTION (확인 필요) | 156 / 2026-04-21 | ICRA 2025 공식으로 보임. 설치 요구는 미확인 |
| Omni6DPose API (물체 원천) | https://github.com/Omni6DPose/Omni6DPoseAPI | MIT | 142 / 2024-09-01 | 데이터셋 API. 데이터 자체는 별도 다운로드 |
| OakInk (전이 방법 원천) | https://github.com/oakink/OakInk | MIT | 153 / 2024-12-01 | 밀집 대응 방법의 원 논문 저장소 |

**판정:** 공식 코드가 없으므로 `04_claim_verification.md`로 대조했습니다. 코드 clone 디렉터리는 만들지 않았습니다(코드가 없음).


## 5. 메모

- 기준선 DexGraspNet 2.0⋆은 CLIP을 붙인 **저자의 개조판**입니다. 원래 DexGraspNet 2.0이 아니므로 기준선의 실제 성능을 독립적으로 확인할 수 없습니다.
- 저자 인용 [22] Embodied3DBench(arXiv:2605.29074)는 같은 저자 그룹의 벤치마크입니다. 이 논문의 VLM 평가 근거로 쓰이지는 않습니다.

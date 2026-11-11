# 02. 선행연구와 비교 대상

## 1. 계보

- **행동 복제와 확산 정책:** BC(Ross 2010/2011 [9][10]), 가우시안 혼합·이산 행동·잠재 행동 계열 [11]–[15], 확산 정책 DP [2](IJRR 2025), 조건부 DDPM [16].
- **확산 트랜스포머:** DiT-Block Policy [6](ICRA 2025)은 adaLN-Zero 블록으로 다중 토큰을 조건으로 줍니다. VolumeDP 디코더는 이 블록을 그대로 씁니다(§III-C).
- **3D 표현 정책:** DP3 [3](포인트 클라우드), Act3D [4], RVT [25], PerAct [23], GNFactor [24]. 모두 깊이 또는 점군 입력을 요구합니다.
- **3D 점유 예측 계열:** Occ3D [28], CVT-Occ [29], BEVFormer [27]. VolumeDP의 deformable 볼륨 리프팅은 이 계열의 기법(deformable attention [26])을 가져옵니다.
- **토큰 축약:** TokenLearner [31]. 학습 가능한 softmax 가중치로 복셀을 N개 토큰으로 줄입니다.

**새로움:** 기법 조합은 기존 요소의 결합입니다(원문도 TokenLearner와 deformable attention을 명시합니다). 새로움은 "RGB만으로 볼륨을 만들고, 볼륨에서 토큰을 골라 DiT 디코더에 넣는다"는 파이프라인과 실험 설계에 있습니다.

## 2. 기준선 출처 검증 (승격 조건의 핵심)

| 표 | 행 | 원문 표시 | 확인 결과 |
|---|---|---|---|
| Table I (LIBERO) | BC-Trans, BC-VILT | 원문 본문: "[32]에서 인용" | **일치 확인.** [32] IB-BC 표 (arXiv 2502.02853)는 열 순서가 Goal, Object, Spatial, Long입니다. 이를 Spatial, Object, Goal, Long으로 바꿔 보면 BC-Trans는 68.00 / 41.83 / 67.83 / 15.83, BC-VILT는 67.17 / 43.00 / 76.17 / 6.50 이고, 표 I의 값과 소수점 한 자리까지 같습니다 |
| Table I | Diffusion Policy [2] | 별표 없음 | **출처를 찾지 못함.** [32]에는 DP가 LIBERO-Long 한 셀(78.0, 50 epoch)로만 나옵니다. 본 논문의 DP 행(78.3 / 92.5 / 68.3 / 50.5, 평균 72.4)과 맞지 않습니다. 웹 검색으로도 이 수치 조합을 찾지 못했습니다. 다른 논문의 DiT 정책 표에도 근사 수치 조각만 보였고 확정하지 못했습니다 |
| Table I | DiT-Block Policy* | 별표(자체 구현) | 자체 실행 표시. 레시피 상세는 원문에 없음 |
| Table II (ManiSkill) | DP*, DiT* | 별표(자체 구현) | 자체 실행. 공개 ManiSkill DP 코드와 비교 가능 (`04` D11) |
| Table III (LIBERO-Plus) | DP*, DiT* | 별표(자체 구현) | 자체 실행 |

**해석.** 같은 표 안에서 BC 행은 인용임이 확인되고, DiT 행은 자체 실행임이 표시되어 있습니다. DP 행은 두 가지 중 하나입니다. (a) 어떤 외부 문헌의 값을 인용했다. (b) 자체 실행했으나 별표를 빠뜨렸다. 원문 근거만으로는 어느 쪽인지 정할 수 없습니다. 어느 쪽이든 +16.4%p 주장의 핵심 기준선이 같은 레시피인지 **확인되지 않았습니다**. ManiSkill과 LIBERO-Plus에서는 DP가 자체 실행인데, LIBERO에서만 다른 출처라면 그 차이를 설명해야 합니다.

## 3. 공정성 비교 축 (같은 저장소의 다른 분석과 비교)

| 분석 | 교훈 | VolumeDP에 대한 시사점 |
|---|---|---|
| `temporal_policy/` (arXiv 2607.29482) | 비교 대상마다 시연 수, 학습 epoch, 샘플러가 달라 수치 차이가 방법 차이로 읽히기 쉽습니다. CMH 검정은 과제 층화 후 결과가 달라질 수 있습니다 | VolumeDP도 시연 수(LIBERO 50, 실물 100)와 epoch를 명시하지 않았습니다. 층화 검정(MH)을 기본으로 봐야 합니다(`03` §2) |
| `contact_disassembly/` (arXiv 2603.08560) | 확산 정책 비교에서 샘플러(DDPM 100 대 DDIM 등)가 성능 차이를 만들 수 있습니다 | VolumeDP는 "100 denoising steps"만 명시. DP와 디코더의 샘플러가 같은지 확인되지 않았습니다 |
| `imle_vla/` (arXiv 2609.10915) | 생성 헤드 비교에서 평가 시드 수가 결론을 바꿉니다 | VolumeDP의 LIBERO ±는 정의가 없어 시드 기반 검정이 불가능 |
| `adavla/` (arXiv 2608.29208) | VLA 비교에서 교란 평가의 표본이 작으면 차이가 불안정합니다 | LIBERO-Plus 교란 평가의 시험 수가 원문에 없습니다 |
| `geometric_entropy/` (arXiv 2606.20871) | 기하 정보를 조건으로 주는 설계의 이득을 분리하려면 대조군이 필요 | 표 IV의 GT depth 대조가 이에 해당하지만 단일 suite |
| `chronospolicy/` (arXiv 2610.13929) | 시계열 조건과 공간 조건의 기여 분리 | 볼륨 공간 토큰과 텍스트·proprio 토큰의 기여는 분리되지 않음 |
| `ba_t/` (arXiv 2606.03287) | 3D 기하를 쓰는 정책의 깊이 의존성 | VolumeDP는 깊이를 쓰지 않지만 카메라 보정이 필요 |

## 4. 관련 3D·볼륨 정책과 직접 비교 가능 여부

| 방법 | 입력 | 공개 여부 | 비교 가능성 |
|---|---|---|---|
| DP3 [3] | 점군(깊이 필요) | 공개(확인 안 함) | VolumeDP가 인용하지 않음. 가장 중요한 빠진 대조 |
| RVT [25], Act3D [4] | 다시점 RGB-D | 공개(확인 안 함) | 깊이 요구 |
| DiT-Block Policy [6] | 2D 토큰 | 공개 여부 확인 안 함 | 디코더 대조군으로 사용됨 |
| TokenLearner [31] | 일반 | 공개 | 기법 원형 |

## 5. GitHub 저장소 (2026-11-11 확인, `gh api`)

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | 별 | 최근 커밋 / 해시 |
|---|---|---|---|---|---|
| `yzc0731/volumedp_maniskill` | ManiSkill 3.0.0b19 기반 저장소. **VolumeDP 고유 코드(볼륨, 토큰, 디코더)는 없음** (`04` §1) | 저자 계정. 프로젝트 페이지 링크는 없음 | Apache-2.0 | 0 | `d2ec13a` (2026-06-22, 단일 커밋 "volumedp") |
| `ZTX2021/LIBERO_openvla` | 프로젝트 페이지의 "Code (Coming soon)" 버튼이 가리키는 주소 | 프로젝트 페이지 링크 | 확인 불가 | — | **404 (저장소 없음 또는 비공개)** |
| `mani-skill/ManiSkill` (구 haosulab) | 시뮬레이터 원본 | 원 개발자 | Apache-2.0 | 3,504 | 2026-08-04 push |
| `real-stanford/diffusion_policy` | DP 원본 | 원 개발자 | MIT | 4,721 | 2024-12-24 push |
| `Lifelong-Robot-Learning/LIBERO` | LIBERO 벤치마크 | 원 개발자 | MIT | 2,528 | 2025-03-15 push |
| `SudeepDasari/dit-policy` | DiT-Block Policy 관련 저장소(공식 여부 미확인) | 추정 | MIT | 171 | 2024-10-15 push |

**요약:** 공개된 코드 중 VolumeDP 고유 구현은 **없습니다**. 공개 저장소는 ManiSkill 포크로, 기준선 DP(state/RGB-D)와 BC 예제만 들어 있습니다. LIBERO 학습·평가 코드와 실물 코드, 복셀 볼륨 모듈은 찾지 못했습니다.

## 6. 파생 방향과 관련 자료

- VolumeDP와 같은 복셀 볼륨 정책은 아직 공개 구현이 없어서, 직접 재구현 비용이 큽니다.
- 원문 [41] Galaxea 오픈월드 데이터셋과 G0 VLA 모델은 실물 로봇 플랫폼의 배경입니다. 재현에는 필요하지 않습니다.

# 02. 선행연구, 구성요소 출처, 공개 자료

## 1. 계보

SimFoundry는 세 갈래를 한 파이프라인에 묶습니다.

1. **디지털 트윈과 digital cousin.** Dai 외 (arXiv:2410.07408, 논문 [17])의 ACDC가 "한 장 이미지에서 장면을 만들고 객체 인스턴스를 바꾼 cousin으로 정책을 학습"하는 틀을 만들었습니다. SimFoundry는 이를 영상 입력, 관절 물체, 배치·작업 cousin으로 넓혔고, 공식 코드의 일부는 ACDC(`cremebrule/digital-cousins`, Apache-2.0)에서 파생됐다고 저장소 README가 밝힙니다.
2. **시뮬 평가의 실제 상관.** SIMPLER (arXiv:2405.05941, [47])가 Pearson r과 MMRV를 도입했고, PolaRiS (arXiv:2512.16881, [32])가 가우시안 스플랫 장면과 환경 composer로 DROID 평가를 확장했습니다. 이 논문의 직접 비교 대상은 PolaRiS입니다(부록 J.3).
3. **합성 데이터 생성.** MimicGen ([58])과 DexMimicGen ([37])이 소수의 사람 시연을 증식하는 방법이고, 공동 학습은 Maddukuri 외 ([55])의 sim-and-real co-training 연구를 따릅니다. 작업 cousin은 "시연 총수를 고정하고 관련 작업 시연을 섞는" 설계로 이 흐름 위에 있습니다.

## 2. 가까운 대안과 차이

논문의 표 1(시스템 비교)은 15개 시스템을 10개 기능으로 비교하며 SimFoundry만 전부 체크됩니다. 이 표는 저자 자체 평가이고 기능 정의(예: "automatic scene construction")가 표에 없습니다.

| 계열 | 대표(논문 번호) | SimFoundry와의 차이 (논문 §1, §2, 부록 D) |
|---|---|---|
| 실제→시뮬→실제 정책 학습 | Re3Sim [27], GSWorld [34], X-sim [19], digital cousins [17] | 대부분 단일 단계 집기·놓기와 강체. SimFoundry는 양팔, 관절, 다단계. 정적 트윈이라 다양성이 제한된다고 주장 |
| 실제→시뮬 정책 평가 | SIMPLER [47], PolaRiS [32], RobotArena∞ [33], Badithela 외 [3] | 상관은 보이지만 시뮬 데이터로 실제 정책을 학습시킨 보고는 없다고 주장. 일반 관절 물체는 지원 안 한다고 서술 |
| 시뮬 준비 3D 장면 생성 | SceneSmith [63], Molmospaces [40] | 물리 상호작용, 작업 지정, 데이터 생성 기계가 부족하다고 서술 |
| 렌더 전용 | [102] | 접촉이 많은 작업에 부적합 |

주의: "no systems support tasks involving general articulated objects"(부록 D.1)는 강한 부정 주장이며 이 분석에서 검증하지 않았습니다.

## 4. 구성요소별 출처

| 단계 | 구성요소 | 쓰인 모델 (논문 부록 E.2) | 코드 기본값 (`04` D3) |
|---|---|---|---|
| 깊이 | V_im2depth | DepthAnything3 (스테레오면 FoundationStereo) | DA3 |
| 분할 | V_seg | SAM3 (영상 전파는 SAM2) | SAM3 |
| 장면 이해, 물성, 관절 | V_scene, V_articulation | "Gemini-Pro-3" | `gemini-3.1-pro-preview` |
| 이미지 편집, 업샘플 | V_image | "Gemini-Pro-3-Image-Preview" | `gemini-3-pro-image` |
| 깊이 인페인팅 | V_inpaint | PriorDepthAnything | 같음 |
| 2D→3D 메시 | V_mesh | Hunyuan3D-2.1 또는 TRELLIS.2 | `hunyuan` (선택: trellis2, pixal3d, direct3d) |
| 포즈 | V_pose | FoundationPose | `use_foundationpose: true` |
| 메시 분할 | V_seg^mesh | P3-SAM (대안 Segment Any Mesh, PartField) | 같음 |
| 영상 객체 제거 | V_inpaint^video | VOID (두 번 청크 인페인팅) | VOID |
| 충돌 | CoACD | CoACD | 같음 (hull 32) |
| 안정화 | PyBullet | PyBullet | 같음 |
| 배경 | 3D Gaussian Splat | Kerbl 외 | 3DGUT 계열 설치 스크립트 포함 |
| 데이터 증식 | MimicGen | [58] | 별도 구현으로 보임 |

## 5. GitHub와 공개 자료 표

확인일 2026-11-23 (`gh api`). 별은 확인 시점 값입니다.

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [NVlabs/SimFoundry](https://github.com/NVlabs/SimFoundry) | **공식 구현** (프로젝트 페이지 GitHub 링크) | Apache-2.0 (일부 선택 의존성은 비상업·제한) | 518 | 2026-08-27 (커밋 `9e34ebef`, 이 분석의 기준) | 얕은 복제, **정적 읽기만** |
| [Hugging Face: nadunRanawaka1/simfoundry-assets](https://huggingface.co/datasets/nadunRanawaka1/simfoundry-assets) | 공식 예시 장면·자산 (README 링크) | 카드에 명시 없음 | (90개 파일, 약 0.36 GB) | 2026-08-27 | 받지 않음 (크기만 확인) |
| [cremebrule/digital-cousins](https://github.com/cremebrule/digital-cousins) | ACDC, 공식 코드의 파생 원본 | Apache-2.0 | 257 | 2025-03-31 | 확인만 |
| [StanfordVL/BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) | OmniGibson과 BEHAVIOR 에셋 (장면·방해 물체 데이터셋) | MIT | 1805 | 2026-11-21 | 확인만 |
| [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | 기본 메시 생성기 | NOASSERTION (Tencent 커뮤니티 라이선스, **EU·영국·한국 제외 지역 한정**, `04` §1) | 4524 | 2025-10-17 | 확인만 |
| [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | 분할 (Hugging Face 게이트 모델) | NOASSERTION | 12375 | 2026-11-17 | 확인만 |
| [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) | 포즈 정제 | NOASSERTION (NVIDIA 라이선스) | 3709 | 2026-04-29 | 확인만 |
| [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | 깊이 | Apache-2.0 | 6723 | 2026-07-27 | 확인만 |
| [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | 대체 메시 생성기 | MIT | 13153 | 2026-07-10 | 확인만 |
| [SarahWeiii/CoACD](https://github.com/SarahWeiii/CoACD) | 충돌 분해 | MIT | 1221 | 2026-09-23 | 확인만 |
| [simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv) | 평가 지표의 원본 | MIT | 1189 | 2026-09-30 | 확인만 |
| [NVlabs/mimicgen](https://github.com/NVlabs/mimicgen) | 시연 증식 | NOASSERTION (NVIDIA 라이선스) | 674 | 2026-10-21 | 확인만 |

- 프로젝트 페이지: <https://research.nvidia.com/labs/gear/simfoundry/>. 논문과 GitHub 링크가 있고 영상과 3D 뷰어 중심입니다.
- 이름이 같거나 비슷한 제3자 저장소(예: `liangynTHU/SimFoundry`, `WUJIAHAO-HKU/SimFoundry-Real2Sim2Real`, `strongegg3/simfoundry_plus`)는 별 0이고 공식과 무관해 보여 다루지 않았습니다.
- PolaRiS(arXiv:2512.16881) 저장소와 SceneSmith, Molmospaces 코드는 확인하지 않았습니다.

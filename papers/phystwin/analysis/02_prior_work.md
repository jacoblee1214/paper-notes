# 02. 선행연구와 관련 자료 (PhysTwin)

## 1. 문제의 계보

PhysTwin이 놓이는 자리는 세 줄기가 만나는 곳입니다(논문 §2 Related Works 기준).

1. **동적 장면 재구성:** D-NeRF [43], Deformable 3D-GS [66], Dynamic 3D-GS [34], 4D-GS [59] 등. 외형과 움직임은 맞추지만 **물리가 없어** 새 상호작용을 예측하지 못함.
2. **물리 기반 역문제:** 사전 스캔 형상이 필요한 초기 방법(DiffPD [9], ADD [15], DiffSim [44]) → NeRF나 3D-GS와 결합한 방법: PAC-NeRF [27], PIE-NeRF [12], VR-GS [22], PhysGaussian [63], Spring-Gaus [72]. 저자의 가장 가까운 선행연구는 **Spring-Gaus**이고, 중력과 운동량 보존이 부정확하며 초기 상태에 촘촘한 시점이 필요하다는 점을 들어 구별합니다.
3. **학습 기반 시뮬레이션:** 그래프 네트워크 계열(Li 외 2018 [28], MeshGraphNets [42], 판 [51, 52]), 그리고 영상에서 학습하는 **GS-Dynamics [69]**. 데이터를 많이 요구하고 물체와 동작 범위가 제한됨. 논문은 GS-Dynamics에 데이터 30개를 줘도 PhysTwin 1개를 못 이긴다고 주장(부록 C).

PhysTwin의 차별점은 **이 세 줄기를 한 최적화로 묶은 것**입니다. 스프링-질량(물리) + 생성형 형상 사전(TRELLIS [62]) + 추적 사전(CoTracker3 [23]) + 등방성 Gaussian 외형(LBS).

## 2. 구성 요소별 출처

| 구성 | 쓰인 것 | 출처 (논문 참고문헌 번호) | 비고 |
|---|---|---|---|
| 물리 모델 | 스프링-질량, 명시적 오일러 | 논문 §3 (일반 기법) | 댐핑과 충돌은 임펄스 기반 |
| 미분 가능 시뮬 | NVIDIA Warp | [37] | 저장소는 CUDA 장치를 하드코딩 (`04` §4) |
| 형상 사전 | TRELLIS (이미지→3D) | [62] | 저장소에서 천과 상자 9개는 쓰지 않음 (`04` D6) |
| 초해상도 | 마스크 영역 확대 | [48] | `image_upscale.py` |
| 분할 | Grounded-SAM2 | [46] (+ GroundingDINO) | 손과 물체 마스크 |
| 추적 | CoTracker3 | [23] | 의사 정답을 3D로 올림 |
| 대응점 | SuperGlue | [50] | 등록 단계 |
| 자세 추정 | EPnP | [25] | PnP |
| 등록 | ARAP | [53] | 비강체 등록 |
| 0차 최적화 | CMA-ES | [18] | 5개 변수 (`04` D4) |
| 외형 | 3D Gaussian Splatting | [20, 54, 69] (LBS 인용) | 등방성, 1만 반복 (`04` D21) |
| 평가 지표 | CD, IoU, PSNR, SSIM, LPIPS | LPIPS [70] | 추적 평가는 TAPIR 도구 [7] |

## 4. GitHub와 관련 자료 (확인 2026-11-21)

| 자료 | 링크 | 라이선스 | 별 / 마지막 push | 상태 |
|---|---|---|---|---|
| **공식 코드** | https://github.com/Jianghanxiao/PhysTwin | MIT | 491 / 67 포크, 2026-08-24 | **`repo/`에 얕은 clone (`81c7187`), 정적 읽기만** |
| 프로젝트 페이지 | https://jianghanxiao.github.io/phystwin-web/ | - | - | "ICCV 2025 Research Project", 영상과 BibTeX |
| 후속: PhysTwin-MPC (MPPI 계획 예시) | https://github.com/Jianghanxiao/PhysTwin-MPC | 없음 (`null`) | 9 / 2026-05-07 | 저장소 README에서 언급. 라이선스 없음이므로 재사용 조건 불명 |
| 후속: Boba (배치 시뮬, ECCV 2026) | 프로젝트 페이지 jianxiapyh.github.io/Boba-project-page/ | 미확인 | 미확인 | README 언급만 확인. 저장소 확인 안 함 |
| 후속: Deform360 (데이터셋, ECCV 2026) | https://deform360.lhy.xyz/ | 미확인 | 미확인 | README 언급만 확인 |
| 후속: Real2Sim-Eval | https://real2sim-eval.github.io/ | 미확인 | 미확인 | README 언급. 이름으로 저장소를 찾지 못함 |
| 기준선 Spring-Gaus | https://github.com/Colmar-zlicheng/Spring-Gaus (이름으로 찾은 후보) | Apache-2.0 | 124 / 2026-11-14 | 공식 여부 미확인. 받지 않음 |
| 기준선 GS-Dynamics | https://github.com/robo-alex/gs-dynamics (이름으로 찾은 후보) | MIT | 122 / 2024-12-24 | 공식 여부 미확인. 받지 않음. 논문은 "PhysTwin 저자가 CoTracker3로 강화"했다고 씀 |
| 형상 사전 TRELLIS | https://github.com/microsoft/TRELLIS | MIT | 14,363 / 2026-10-27 | 저장소가 의존. 설치하지 않음 |
| 데이터 (Hugging Face) | https://huggingface.co/datasets/Jianghanxiao/PhysTwin | 미확인 | - | `data.zip`, `experiments*.zip`, `gaussian_output.zip`. **받지 않음** |
| 논문 | arXiv:2503.17973 (v1, 2025-03-23) | arXiv non-exclusive | - | `paper/paper.pdf` |

**메모:** 기준선 저장소 두 개는 이름으로 찾은 후보입니다. 논문의 기준선이라는 확인은 저자 README나 논문 링크로 해야 합니다.

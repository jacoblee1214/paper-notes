# UMI 선행연구와 관련 자료

## 1. 계보

UMI는 두 줄기를 잇습니다. 하나는 **손에 드는 그리퍼로 시연을 모으는 흐름**이고, 다른 하나는 **사람 영상에서 행동을 얻는 흐름**입니다. 논문 §II가 두 흐름을 정리합니다.

1. **그리퍼 시연 (quasi-static 행동의 출발점).** Song et al. 2020(Grasping in the wild, RA-L, 참고 [41])이 손목 카메라와 RGB-D로 집기 시연을 모았습니다. Young et al. 2021(Visual imitation made easy, CoRL, [50])은 저비용 시연 도구로 시각 모방 데이터를 모았습니다. Doshi et al. 2023(arXiv:2311.01832, [10])은 손에 드는 그리퍼와 사람 조작 사이의 형태 차이를 다뤘습니다. UMI는 이들의 "모은 시연의 행동을 정확히 얻지 못한다"는 문제를 SLAM으로 풉니다.
2. **원격 조작과 장치 (teleoperation).** Spacemouse, VR/AR, 스마트폰, 햅틱 장치 [9, 35, 3, 13, 19, 31, 51, 12, 44, 45, 22, 38, 47, 43, 26, 4]. ALOHA(Zhao et al. RSS 2023, [53])와 GELLO([46]) 같은 리더-팔로워 방식은 저렴하지만 실로봇이 필요합니다. 외골격([14, 20])은 실로봇 미세 조정이 필요합니다.
3. **사람 영상 학습.** 비용 함수, 어포던스, 밀집 기술자, 사전학습 표현 같은 방법([37, 8, 1, 21, 2, 40, 24, 39, 33, 28, 23, 48])이 있습니다. UMI는 사람 영상의 행동 추정 문제(손 자세 추정 [44, 1, 38, 28])와 embodiment 간극 문제를 피하려고 행동을 SLAM으로 직접 얻습니다.
4. **Dobb-E (Shafiullah et al., 2023, arXiv:2311.16098, [36]).** iPhone을 단 "reacher-grabber"로 Stretch 로봇에 단팔 시연을 모읍니다. 정책 배포에 환경별 미세 조정이 필요하고 준정적 과제에 머문다고 UMI가 비교합니다.
5. **정책 학습.** Diffusion Policy (Chi et al., RSS 2023, [9])를 씁니다. 행동을 조건부 확산으로 모델링하고, 관측과 행동 지평을 두고 DDIM으로 샘플링합니다.

**UMI의 자리.** 로봇을 시연에 쓰지 않으므로 teleoperation보다 설정 비용이 작고, 사람 영상보다 행동 정보가 정확합니다. 대신 시연자가 그리퍼 무게와 형태를 감수해야 하고, 로봇 배포 시 운동학 한계를 미리 모르는 문제가 남습니다(01 §4).

## 2. 가까운 대안

| 대안 | 무엇이 다른가 | 비교 위치 |
|---|---|---|
| Dobb-E (arXiv:2311.16098) | iPhone 기반 단팔 시연, 준정적 과제, 환경별 미세 조정 | 논문 §II-B |
| Doshi et al. (arXiv:2311.01832) | 손에 드는 그리퍼의 형태 차이 분석 | 논문 §II-C |
| Mobile ALOHA (arXiv:2401.02117) | 원격 조작으로 양팔 모바일 시연 | 논문 §II-A |
| GELLO | 저렴한 리더-팔로워 원격 조작 | 논문 §II-A |
| DexMV (Qin et al., ECCV 2022, [28]) | 사람 영상의 손 자세 추정을 로봇 손으로 옮김 | 논문 §II-B |

## 4. 공식 코드와 저장소

| 저장소 | 관계 | 라이선스 | 최근 push | 별 | 상태 |
|---|---|---|---|---|---|
| [real-stanford/universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface) | 공식 구현 (SLAM 파이프라인, 학습, 배포) | MIT | 2026-05-26 | 1,647 | 얕은 복제 `repo/`, HEAD `d095ba9`(2026-05-26). 정적 읽기만 |
| [umi-gripper.github.io](https://umi-gripper.github.io) | 프로젝트 페이지, 데이터 수집 안내 링크 | 해당 없음 | 해당 없음 | 해당 없음 | 페이지의 "RSS 2024 Best Systems Paper Award Finalist" 표기는 공식 proceedings에서 확인하지 못함 |
| [cheng-chi/ORB_SLAM3](https://github.com/cheng-chi/ORB_SLAM3) | 공식 README가 쓰는 SLAM 포크 (맵 초기화, 마커 초기화) | GPL-3.0 | 2024-01-11 | 57 | 저장소 밖. 이 분석에서 읽지 않음 |
| [UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) | 상위 원본 ORB-SLAM3 | GPL-3.0 | 2024-07-24 | 9,266 | 참고용 |
| [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | 정책 학습 틀 (Chi et al., RSS 2023) | MIT | 2024-12-24 | 4,737 | 저장소 밖. 참고용 |

- 별 수, 최근 push, 라이선스는 GitHub 메타데이터(`gh repo view`, 2026-11-21 조회)입니다.
- 공식 저장소에는 `example/`(예시 시연 세션 다운로드 안내), `scripts_slam_pipeline/`, `umi/real_world/`, `diffusion_policy/config/`가 있습니다. 학습 설정은 04에서 논문과 대조합니다.
- 공식 저장소의 SLAM 실행은 Docker 이미지(`chicheng/orb_slam3`)를 씁니다. 이 맥에서는 리눅스 컨테이너 실행과 GPL 포크 빌드가 필요하므로, 실행하지 않았습니다.
- 학습 설정(`train_diffusion_unet_timm_umi_workspace.yaml`)은 CUDA 전제입니다(`device: cuda:0`).

## 5. 라이선스 메모

- 논문: arXiv 비배타 배포 라이선스(non-exclusive distribution license 1.0). CC BY 4.0이 아니므로 논문 그림은 공개 산출물에 넣지 않았습니다. 본문 인용과 수치만 씁니다.
- 공식 코드: MIT (`LICENSE`에 Columbia AI & Robotics Lab, 2023 저작권 표기).
- 저장소에서 쓰는 SLAM 포크는 GPL-3.0입니다. 이 분석은 포크 코드를 읽지도 빌드하지도 않았습니다.

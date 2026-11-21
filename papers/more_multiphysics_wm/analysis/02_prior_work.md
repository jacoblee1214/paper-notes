# 02. 선행연구와 공식 저장소

## 1. 계보와 가까운 대안

MoRE는 세 계열 사이에 놓입니다(논문 §2, 표 1).

| 계열 | 대표 연구 (논문 번호) | MoRE와의 차이 |
|---|---|---|
| 물리 기반 시뮬 (미분 가능) | PhysTwin [13], PhysGaussian [14], DiffPD [17] | 시스템 식별과 테스트 시 최적화가 필요합니다. MoRE는 이력에서 암묵적으로 추정합니다(§2) |
| 학습 기반 입자와 그래프 동역학 | AdaptiGraph [12], ParticleFormer [15], PGND [11], Points2Plans [16] | 재질이나 체제별로 학습합니다. ParticleFormer는 장면마다 학습한다고 표 1에 표시합니다 |
| 2D 특징 또는 영상 세계 모델 | DINO-WM [1], RoboDreamer [2], Ctrl-World [6] | 3D 기하와 접촉 기반이 약합니다. MoRE는 3D 점 궤적을 예측합니다 |
| 3D 점 세계 모델 (가장 가까움) | PointWorld [3] | 단일 암묵 예측기입니다. MoRE는 체제별 전문가로 나눕니다. 표 4의 비교 대상입니다 |

- **MoE 계보:** 희소 게이트 MoE [30], Switch [31]. 로봇 동역학에서 쓰는 예는 논문에 없습니다.
- **학습 방식:** 자기 롤아웃(self-forcing [63])을 동역학에 옮겼습니다.
- **입력 표현:** 행동을 툴 점군으로 두는 방식은 [60–62]의 관절이나 키포인트 행동과 다릅니다.
- **시뮬 데이터 도구:** MagicSim [64], Isaac Lab [69], RoboTwin 2.0 [67], RoboSmith [66].
- **학습 기반 정적 기하 인코더:** Uni3D [72] (동결).

## 2. 공식 코드와 저장소

| 항목 | 주소 | 상태 (2026-11-21 조회) | 판단 |
|---|---|---|---|
| 공식 코드 | https://github.com/AlwaySleepy/MoRE-WM | 생성 2026-06-27, 마지막 push 2026-09-16, 별 4, 포크 1, 기본 브랜치 `model`. 프로젝트 페이지의 Code 링크(`MoRE-Robo/MoRE-WM`)가 여기로 리다이렉트됩니다. **LICENSE 파일 없음**(API license null) | 복제: `repo/`, `model` 브랜치 `a3eee7fa67`(2026-09-16). 정적 읽기만 함 |
| 공식 코드, 데이터 파이프라인 브랜치 | 같은 저장소 `data` 브랜치 | 커밋 `7c740d70f9`(2026-06-28). **Apache-2.0 LICENSE** 있음 | `FETCH_HEAD`로 목록만 읽음. DROID 처리 파이프라인(Qwen, SAM) |
| 프로젝트 페이지 | https://alwaysleepy.github.io/MoRE-WM-site/ | CoRL 2026 · Spotlight 표기, 논문 PDF `assets/MoRE_CoRL2026.pdf`, 데이터와 모델 링크 | 공식. 소스 저장소 `AlwaySleepy/MoRE-WM-site`(생성 2026-06-03, push 2026-11-20, 라이선스 없음) |
| 저장소 README의 프로젝트 주소 | https://more-robo.github.io/ | 논문 페이지와 다른 주소 | 두 주소가 섞여 있음 (`04` D16) |
| 데이터 | https://huggingface.co/datasets/MoRE-WM/MoRE-Data | 파일 306개, 약 **10.6 GB**(메타데이터 조회만, 받지 않음), 2026-07-31 수정, 라이선스 메타 없음 | 받지 않음 |
| 모델 | https://huggingface.co/MoRE-WM/MoRE-Model | `model_mix_droid.pth` 730 MB, 2026-07-31 | 받지 않음 |
| 외부 체크포인트 | BAAI/Uni3D `modelzoo/uni3d-b/model.pt` | 저장소 README가 `MoE/models/ckpt/`에 두라고 함 | 받지 않음 |
| 외부 설치 | Pointnet2_PyTorch (erikwijmans) | CUDA 확장 필요 | 설치하지 않음 |

GitHub 검색으로 찾은 다른 저장소(`operatic-ai/corl-explorer`의 목록, 개인 포크 등)는 공식이 아닙니다.

## 4. 그림과 라이선스 판단

- **논문 그림:** 라이선스는 CC BY 4.0이고 권리 표시 충돌은 없습니다. 그래도 그림 1, 3, 5, 6, 7, 8은 실로봇 사진, DROID 장면 프레임, 시뮬 렌더링이 섞여 있습니다. 그림 2(구조도)와 그림 4(MPC 곡선)만 후보입니다. 이번 공개본은 두 그림을 빼고 우리가 그린 그래프 세 장만 씁니다. 그래서 공개 출력에는 논문 그림이 없습니다. (나중에 그림 2만 넣기로 하면 출처와 CC BY 4.0 표기를 달면 됩니다.)
- **공식 코드:** `model` 브랜치에 LICENSE가 없습니다. 기본 권한이 없으므로 코드를 공개본에 복사하지 않습니다. 설명만 씁니다.

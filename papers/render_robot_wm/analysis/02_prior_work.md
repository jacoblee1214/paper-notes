# 02. 선행연구와 이 저장소의 관련 분석

## 1. 계보 (논문 §2 기준)

| 축 | 대표 연구 (논문 인용) | Doppel과의 차이 |
|---|---|---|
| 행동 조건 영상 world model | IRASim [21], Ctrl-World [20], Vid2World [16], EgoWM [2] | 행동 벡터만 입력. Doppel이 기준선으로 쓰는 EgoWM 방식은 타임스텝 변조에 행동을 넣는다 |
| 기하 조건 | OSCAR [23] (골격), PointWorld [24], ORV [25] (점유), TesserAct [26] (4D), BridgeV2W [27] (임베디먼트 마스크), EnerVerse-AC [28] (말단 맵), Kinema4D [29] (포인트맵), Dexterous WM [30] | 몸 전체를 렌더링하고 수치 행동을 함께 넣는다. 논문은 ORV, EnerVerse-AC만 수치 행동을 같이 쓴다고 구분한다 |
| 이미지 공간 임베디먼트 교체 | Mirage [39], RoViAug [40], Phantom [41], SHADOW [42] | 학습 데이터의 로봇을 바꾸는 방식. Doppel은 같은 렌더 공간에 여러 몸을 둔다 |
| 로봇 데이터 합성 | AnchorDream [31], RoboDream [32] | 렌더된 로봇 운동으로 데모를 합성한다. Doppel은 예측 모델 조건으로 쓴다 |
| 사전학습 영상 백본 | Wan 2.1 [15], Cosmos [12], Stable Video Diffusion [13] | Doppel은 Wan 2.1 I2V 14B를 쓴다 |
| 다중 임베디먼트 정책 | Open X-Embodiment [33], Octo [34], π0 [36] | 정책 쪽 공유. Doppel은 world model 쪽이다 |

## 2. 가까운 대안 (구현 비교 기준)

- **OSCAR (arXiv:2606.04463)**: 골격 조건 world model. 논문의 시스템 수준 기준선이다. 백본은 Cosmos-Predict2.5 2B다. 코드를 받지 않았다(§4 참조).
- **EgoWM 방식 (Bagchi et al., ECCV 2026)**: 행동만 조건으로 쓰는 Wan 2.1 I2V 기준선. Doppel 어댑터와 같은 백본을 쓰므로 가장 공정한 대조군이다.

## 4. 공식 코드와 가중치 (GitHub API 메타데이터, 2026-11-21 조회)

| 항목 | 링크 | 라이선스 | 생성 / 마지막 push | 별 | 비고 |
|---|---|---|---|---|---|
| 공식 코드 | https://github.com/DoppelWM/Doppel | MIT (코드). 가중치는 CC BY-NC 4.0 (README) | 2026-11-06 / 2026-11-07 | 3 | 정적 읽기용으로 `repo/`에 shallow clone (commit 306fc88) |
| 프로젝트 페이지 | https://doppelwm.github.io/ | - | - | - | 논문 링크와 같다 |
| 가중치 (Doppel-8) | https://huggingface.co/sourlreapwr/doppel-emb8 | CC BY-NC 4.0 (README) | - | - | 받지 않음 |
| 가중치 (Dr.Robot 모델) | https://huggingface.co/sourlreapwr/doppel-drrobot | - | - | - | 받지 않음 |
| 백본 | https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers | Wan 2.1 Apache-2.0 (README) | - | - | 받지 않음 |
| 카메라 외부 파라미터 | https://huggingface.co/datasets/sourlreapwr/doppel-camera-extrinsics | - | - | - | 받지 않음 |
| 보정 참조 | RobotArena∞ [51] (ICLR 2026) | - | - | - | 보정 절차의 원 논문 |

- 코드 저장소 크기는 13 MB다. 실행하지 않았다. 설치, import, 테스트도 하지 않았다.
- 코드 라이선스: `LICENSE`는 MIT다. `calibration/drrobot/`의 일부 파일은 3D Gaussian Splatting 라이선스(비상업 연구 전용)를 따른다(`calibration/README.md`).

## 5. 주요 참고 문헌의 상태

- 논문 [2] EgoWM (Bagchi et al., ECCV 2026)과 [23] OSCAR는 본 저장소에 분석 폴더가 없다. 행동 전용 기준선의 구조는 코드(`doppel/action_only_adapter.py`)에서 확인했다.
- 논문 [49] Dr. Robot (Liu et al., CoRL 2024)은 로봇 렌더러의 원 논문이다. 공식 코드 `calibration/README.md`가 arXiv:2410.13851을 가리킨다.

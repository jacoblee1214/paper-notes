# 02. 선행연구와 관련 저장소

## 1. 선행연구 계보 (구성 요소별 출처)

| 구성 요소 | 출처 (논문 번호) | DepthWorld에서 한 일 |
|---|---|---|
| 행동 조건 영상 세계 모델 (백본, 학습 시작점) | Ctrl-World [1] (SVD, DROID, 프레임별 엔드이펙터 행동) | 같은 SVD 체크포인트와 행동 조건을 쓰고, RGB 기준선으로 같은 데이터에 재학습 |
| 정책 평가 용도의 행동 조건 영상 모델 | IRASim [19] 외 [20] | 배경 인용만 |
| 롤아웃 안정화 (RL 후처리) | PersistWorld [2], 같은 저자 그룹 (arXiv:2603.25685) | 한계 절에서 인용만 |
| 기하 기반 모델 (점지도 회귀) | DUSt3R [6], MASt3R [7], VGGT [8], MapAnything [11] | 점지도 헤드(VGGT 융합 몸통), DUSt3R식 신뢰도 가중과 MapAnything식 강건 손실 |
| 이미지 확산 VAE가 깊이를 잘 인코딩 | Marigold [24], DepthCrafter [25] | 깊이를 RGB 옆에 타일로 붙이는 근거 (§4.1) |
| 깊이 채널 확장, 이중 분기 | 채널 확장 [4, 26], SyncVP [27] (이중 분기) | 두 방식을 같은 평가로 비교 (표 A2) |
| 스테레오 깊이 | S²M² [35], FoundationStereo [45], ZED SDK | 깊이 소스로 S²M² 선택 (표 A3, 부록 A.1) |
| 대응 매칭 | RoMa v2 [36], LoMa/LoMa-G [37, 38], LightGlue [49], LoFTR 계열 | 보정 Stage 1–2의 조밀·2D–3D 대응 |
| 장면별 보정 (기준선) | PointWorld [34]의 장면별 정제, VGGT 초기화 | 같은 깊이 소스로 바꾼 기준선 (부록 D.1) |
| 3D 점 세계 모델 | PointWorld [34] (점 흐름, DROID로 학습) | 비교 대상 (표 A5) |
| 4D 세계 모델 | TesserAct [4] (RGB-D-노말, CogVideoX) | 비교 대상, 행동 조건으로 바꿔 재학습 (부록 D.5) |
| DROID 데이터셋 | DROID [18] | 보정과 학습의 원본 |

- **가장 가까운 대안:** 저자가 보기에 깊이를 같이 생성하는 기존 방식은 두 부류입니다. 입력 채널을 늘리는 방식(VAE와 U-Net 입출력 수정, 가중치 재초기화 필요)과 이중 분기 U-Net입니다(§1, §4.1). DepthWorld는 U-Net 구조를 건드리지 않는다는 점이 차이입니다.
- **데이터 쪽 대안:** Open X-Embodiment, RH20T, BridgeData는 기하 정보가 단안 RGB나 단일 RGB-D 수준입니다. 시뮬 코퍼스(RoboCasa, ARNOLD)는 완전한 기하를 주지만 sim-to-real 간극이 있습니다(§2).

## 2. 코드와 저장소 (GitHub 표)

2026-11-22 기준 GitHub API와 Hugging Face API로 확인했습니다.

| 저장소 | 관계 | 라이선스 | 기준 (날짜, 커밋) | 별·포크·이슈 | 상태 |
|---|---|---|---|---|---|
| [Jai2500/depthworld](https://github.com/Jai2500/depthworld) | 공식 구현 (학습, 평가, 데이터 준비 스크립트, 보정 데이터 다운로더) | MIT (`LICENSE.txt`, 2026 Bardhan·Šivic·Petrík, 2025 Tsinghua 표기 포함) | 마지막 푸시 2026-10-09, 커밋 `6823c13` (2026-10-09) | 5 · 0 · 1 | 얕은 복제 `repo/`, 정적 읽기만. 실행과 설치 안 함 |
| [jaibrdhn/depthworld](https://huggingface.co/jaibrdhn/depthworld) (HF 모델) | 학습된 체크포인트 두 개: `horiz_90k_ema.pt` (6.1 GB), `horiz_dpt_vggt_90k_ema.pt` (6.2 GB) | 카드 기준: 코드 MIT, 가중치는 SVD Community License 적용, VGGT 초기화 헤드는 **CC BY-NC 4.0** (비상업 연구) | 마지막 수정 2026-10-07 | 1 좋아요 · 다운로드 40 | 받지 않음 (가중치 다운로드 금지 조건) |
| [jaibrdhn/droid_3d_extrinsics](https://huggingface.co/datasets/jaibrdhn/droid_3d_extrinsics) (HF 데이터) | DROID-3D 보정 자세(`extrinsics.jsonl`, 71,006 에피소드), 내부 파라미터, 실패 목록 | 보정 자세와 내부 파라미터는 **CC BY 4.0**, `pseudolabels/`는 **CC BY-NC 4.0**(VGGT-Ω 출력, 비상업). 전체 약관은 별도 Google 문서 | 마지막 수정 2026-10-22 | 7 좋아요 · 다운로드 267 | 접근 제한(gated, 수동 승인). 받지 않음 |
| 프로젝트 페이지 [jaibardhan.com/depthworld](https://www.jaibardhan.com/depthworld) (arXiv comments에 적힌 주소) · [jai2500.github.io/depthworld](https://jai2500.github.io/depthworld/) (페이지 내부 링크) | 프로젝트 사이트 (비디오, 결과) | 사이트 하단에 CC BY-SA 4.0 링크가 있으나, 사이트 템플릿 고지로 보입니다. 논문 라이선스와 별개로 봅니다 | 2026-11-22 확인 | — | 본문만 확인, 비디오 미확인 |

- **공식 코드의 범위:** 저장소에 있는 것은 세계 모델 학습(`scripts/train_wm*.py`, `models/`), 데이터셋 로더(`dataset/`), 점지도 생성(`depth_extras/pointmap.py`), 평가(`scripts/eval_horiz_*.py`), 데이터 추출(`dataset_example/`)입니다. **저장소에 없는 것은 DROID-3D 보정 최적화기(요인 그래프, Schur LM, RoMa·LoMa 대응, 손목 정합)와 S²M² 추론 코드입니다.** 보정 결과 파일은 HF에서 받아야 하고, 원시 깊이(`disparity_lo`)는 `recordings/s2m2_v2/…/stereo_s2m2.npz`에서 읽도록 되어 있습니다(`dataset_example/export_disparity_lo.py`). 이 파일을 만드는 스크립트는 저장소에 없습니다.
- **데이터 접근:** 깊이 데이터는 공개 카드에서 "접근 양식으로 요청"입니다. 보정 데이터는 자동 승인 게이트이고 약관 문서에 동의해야 합니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **Marigold [24]:** 이미지 확산 VAE가 깊이 맵을 거의 손실 없이 인코딩한다는 관찰. 타일링의 전제입니다. 논문은 VAE 왕복 바닥(부록 D.4에서 이동 점 약 45 mm)을 따로 인정합니다.
- **DUSt3R [6], MASt3R [7], MapAnything [11]:** 신뢰도 가중 로그 공간 강건 손실. 점지도 헤드의 손실 형식(식 A8–A12)이 여기서 왔습니다.
- **VGGT [8]:** 헤드 초기화 몸통. 공개 체크포인트는 VGGT-1B 가중치에서 시작하며, 이 가중치는 CC BY-NC 4.0입니다(§2 표).
- **S²M² [35]와 ZED 스테레오:** 깊이 소스 선택(표 A3)의 근거. S²M²가 ZED NEURAL과 거의 같고 ULTRA보다 낫다는 결과는 RGB 기준으로 재현 가능한 수준입니다(`analysis/03_analysis.md` §1.4).
- **PersistWorld [2]:** 같은 저자 그룹의 RL 후처리 연구. DepthWorld의 자기회귀 한계를 메우는 다음 단계로 인용됩니다.

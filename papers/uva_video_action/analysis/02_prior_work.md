# 02. 선행연구와 관련 자료

## 1. 공식 코드, 프로젝트 페이지, 관련 자료

| 자료 | 주소 | 라이선스 | 규모·최근 활동 | 메모 |
|---|---|---|---|---|
| 공식 코드 (PyTorch) | github.com/ShuangLI59/unified_video_action | MIT (LICENSE 파일 확인) | 별 419, 포크 36, 기본 브랜치 main, 언어 Python. 생성 2025-03-04. 최신 커밋 fe3dec8 (2026-08-20, "Update README.md"). API의 pushed_at은 2026-08-21 | 정적 읽기만 함. 저장소 복제본은 `repo/` (gitignore). `04` |
| 프로젝트 페이지 | unified-video-action-model.github.io | 바닥글 CC BY-SA 4.0 (템플릿 표기) | 영상 예시와 실로봇 롤아웃 | 실로봇 롤아웃과 영상 생성 비교는 여기서만 볼 수 있다. 논문 본문에는 없다 |
| Colab (PushT) | README 링크 | 확인 안 함 | 사전학습 체크포인트로 PushT 추론 | 실행하지 않음 |
| 체크포인트 | 구글 드라이브 링크 (PushT, PushT-M, Libero10, UMI 다중 과제) | 확인 안 함 | 크기 확인 안 함 | 다운로드하지 않음 (요청 제약) |
| 사전학습 VAE, MAR-B | `unified_video_action/utils/download.py`로 받음 (MAR 저장소 LTH14/mar) | MAR 저장소 라이선스 확인 필요 | 확인 안 함 | 학습의 출발점. 다운로드하지 않음 |
| 데이터 | PushT (Diffusion Policy), Libero10 (LIBERO), Toolhang (robomimic), UMI 데이터 3종, PushT-M (저자 제공) | 각 원 데이터 라이선스 | PushT-M은 README에 247개 시연이라고 적혀 있음 | 본 분석은 데이터를 받지 않음 |
| 실로봇 스택 | arx5-sdk, umi-arx, umi-on-legs | 확인 안 함 | 외부 저장소 | 본 분석 범위 밖 |

- 공식 코드 저장소에는 `prepared_data/`가 들어 있다: FVD 계산용 I3D 가중치 `i3d_pretrained_400.pt` (약 51 MB), 언어 임베딩 `language_latents.pkl` (6 KB), 샘플 인덱스 json (20 KB). 복제 저장소 전체는 약 81 MB (.git 포함)다. 이 파일들은 저장소 복제로만 받았고, 별도 다운로드는 하지 않았다.

## 2. 선행연구 계보 (논문 참고문헌 기준)

| 구성 요소 | 대표 선행연구 (논문 번호) | UVA에서의 쓰임 |
|---|---|---|
| 영상 생성을 정책 학습에 쓰기 | UniPi [14], Dreamitate [28], PAD [19], Video Prediction Policy [24], Flow as the cross-domain interface [49] | UniPi는 기준선. PAD는 "행동이 영상 생성을 기다리는 구조라 느리다"는 비판 대상 |
| 영상 생성 모델을 동역학 모델로 쓰기 | GameGen-X [9], Genie [5], Diffusion models as game engines [43] | 서론과 관련 연구에서 동기로 인용 |
| 마스크 기반 학습 | MAE [20], MaskGIT [8], Masked Autoencoding for decision making [31], Masked Trajectory Models [48], Radosavovic et al. [34] | 마스킹 방식을 과제 구성에 맞춘다는 점에서 차이 |
| 자기회귀 연속 토큰 영상·이미지 생성 | MAR [27] (VQ 없는 확산 헤드), VAR 계열 [13, 17] | UVA의 Transformer 초기값과 확산 헤드 구조 출처. 논문은 MAR-B를 쓴다 |
| 확산 정책 | Diffusion Policy [10], UMI [11] | 시뮬 기준선 DP-C, DP-T. 실로봇 기준선 DP-UMI (CLIP ViT-B/16 인코더) |
| VLA | OpenVLA [25], π0, π0-FAST [2] | 시뮬 기준선. 논문은 π0 기준선을 v3에서 추가 |
| 벤치마크 | PushT [10, 16], ToolHang [32], LIBERO [30], UMI 과제 [11, 29] | 평가 환경 |
| 역동역학, SLAM | ORB-SLAM3 [6] | 역동역학 비교 기준 (Table VI) |
| 사람 영상 | Action-free reasoning [12] (Human Video 데이터) | 부록 E |

## 4. 논문이 빠뜨린 선행연구 메모

- UniPi는 공식 코드가 없어 [26] 구현을 썼다고 명시한다 (§V-C). 기준선 강도가 원래 구현과 같은지 알 수 없다.
- PAD [19]는 같은 시기 "영상과 행동 공동 확산" 연구다. UVA는 PAD의 추론 지연을 문제로 지적하지만, 본문에 PAD와의 정량 비교가 없다.
- 마스크 학습 정책 [31, 48]은 저차원 상태를 쓴다. UVA가 영상 토큰으로 옮긴 차이의 설명은 있으나 정량 비교는 없다.

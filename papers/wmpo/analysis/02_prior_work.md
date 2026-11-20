# 02. 선행연구, 공식 저장소, 이 저장소의 관련 분석

## 0. 학회 확인

- arXiv 댓글과 저널 참조에는 학회가 없습니다 (2026-11-20 abs 페이지).
- **ICLR 2026 포스터**: `https://iclr.cc/virtual/2026/poster/10007263` (제목과 저자 일치, 페이지 제목 "ICLR Poster WMPO ..."). 학회 가상 사이트의 포스터 목록은 채택 논문만 올라가므로 공식 채택 목록으로 봅니다. OpenReview 포럼 `qE2FyvRvuF`는 봇 검증 페이지에 막혀 확인하지 못했습니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| 구성 요소 | WMPO에서 쓴 것 | 본문 참고문헌 번호 |
|---|---|---|
| 기저 VLA | OpenVLA-OFT, 병렬 디코딩, 이산 토큰 행동 | [2], [24] |
| 온폴리시 RL 기반 | GRPO (DeepSeekMath), 클리핑과 동적 샘플링 (DAPO) | [30], [44] |
| 비디오 세계모델 백본 | OpenSora (STDiT), SDXL 2D VAE, AdaLN 프레임 제어 (IRASim) | [35], [36], [37], [38] |
| 보상 모델 | VideoMAE 인코더 + 선형 헤드 | [39] |
| 데이터 | Open X-Embodiment 사전학습, MimicGen 시뮬 | [21], [23] |
| 비교 기준선 | GRPO (실제 환경), DPO (오프라인, 쌍 선호) | [28], [30] |
| 잠재공간 세계모델 대비 | RSSM, Dreamer 계열 (잠재공간 RL) | [15]–[18] |
| 픽셀 세계모델 RL 선행 | 확산 세계모델로 Atari 픽셀 보존 (DIAMOND 계열) | [31] |
| 로봇 세계모델 RL 선행 | World4RL (확산 세계모델로 로봇 정책 정제) | [32] |
| 큰 세계모델 | Cosmos, Genie 3 | [12], [13] |

**계보에서 볼 점.** WMPO의 차별점은 "세계모델로 RL을 한다"가 아니라 (1) 픽셀 공간 생성과 정책 행동 정렬, (2) 자기회귀 클립 생성으로 끝까지 궤적을 만드는 것, (3) 보상 모델을 쓴 온폴리시 GRPO입니다. 그런데 본문은 [32] World4RL을 같은 절에서 인용하면서 "처음으로 검증했다"고 씁니다. 이 표현은 좁혀야 맞습니다 (`01` §5, `03` C7). World4RL의 실험 세부는 이 분석에서 읽지 않았습니다.

## 2. 공식 코드와 저장소 (확인 2026-11-20)

GitHub API(`api.github.com/repos/WM-PO/WMPO`)와 얕은 clone으로 확인했습니다. 저장소 메타와 최신 커밋을 적었습니다.

| 자료 | 링크 | 라이선스 | 별 / 포크 | 마지막 push | 복제 커밋 | 상태 |
|---|---|---|---|---|---|---|
| **공식 코드 (WMPO)** | https://github.com/WM-PO/WMPO | Apache-2.0 (LICENSE 파일과 API 일치) | 246 / 14 | 2026-01-04 | `c836d74` "clean repo" (2026-01-04) | `repo/WMPO/` 얕은 clone, 정적 읽기만. 실행과 설치 안 함 |
| 프로젝트 페이지 | https://wm-po.github.io/ | 페이지 자체 CC BY-SA 4.0 (정적 템플릿 Nerfies 기반) | - | - | - | 본문 머리말과 공식 저장소 버튼에 연결 |
| 체크포인트 (HF) | https://huggingface.co/fangqi/WMPO | 미확인 | - | - | - | 받지 않음. README는 세계모델 364 GiB + 데이터 530 GiB라고 적음 |
| 의존 저장소 (벤더링) | `dependencies/opensora` (OpenSora), `dependencies/openvla-oft` | 각 LICENSE 파일 (OpenSora Apache, OpenVLA-OFT는 저장소 파일에 있음) | - | - | 저장소 내부 복사본 | 정적 읽기만. 세계모델 백본과 VLA 구현 확인에 사용 |

- 공식 코드는 `verl` 기반 RL 학습 루프, `reward_model/`(VideoMAE 학습과 임계값 탐색), `examples/`(과제 4개 × P 2개의 학습 스크립트와 평가 스크립트)로 구성됩니다.
- 이 저장소의 `repo/`는 gitignore되어 있고 `.git` 이력이 129 MB입니다. 공개본에는 들어가지 않습니다.
- `dependencies/opensora/`에 SDXL 계열 2D VAE(`opensora/models/vae/video_sdxl/`)가 있어 논문의 VAE 선택과 맞습니다.

## 3. 이 저장소의 관련 분석 (같은 저장소 안 연결)

이 목록은 WMPO와 직접 맞닿는 분석만 적었습니다. 세부 판단은 각 폴더의 `01`–

| 폴더 | 논문 | 연결 |
|---|---|---|


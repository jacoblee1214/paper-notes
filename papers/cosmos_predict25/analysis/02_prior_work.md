# Cosmos-Predict2.5 선행연구, 가까운 대안, 공식 코드

## 1. 계보

| 단계 | 연구 | 이 보고서와의 관계 |
|---|---|---|
| 확산 기반 | EDM (Karras 2022) | Cosmos-Predict1이 쓴 전처리 방식. 2.5는 flow matching으로 바꿈 (§3.1) |
| 확산 기반 | Flow matching (Lipman 2022), 정규화 이동 (Esser 2024) | 식 (1)–(4)의 학습 목표와 β 이동 |
| 전신 | Cosmos-Predict1 (NVIDIA 2025) | 직접 전신. DiT 구조, 데이터 파이프라인, 블록별 링 어텐션을 물려받음 |
| 전신 | Cosmos-Transfer1 (NVIDIA 2025) | Transfer2.5의 제어 구조(VACE형 제어 블록)와 학습 데이터 설계의 출처 |
| 구성 요소 | WAN2.1 VAE (Wan 2025) | 시각 토크나이저. 2.5는 Predict1의 토크나이저를 바꿈 |
| 구성 요소 | Cosmos-Reason1 (NVIDIA 2025) | 텍스트 인코더. T5 대체 (§3.2) |
| 구성 요소 | Qwen2.5-VL (Bai 2025) | 캡션, 필터 VLM |
| 학습 | VideoAlign (Liu 2025), GRPO (Guo 2025), Ye 2025 | RL 보상과 정규화. 공식 코드에 없음 (`04` D12) |
| 증류 | rCM (Zheng 2025, arXiv:2510.08431) | 4 스텝 증류 (§4.2.3). 공식 코드에서 DMD2가 중심 (`04` D13) |
| 응용 평가 | DreamGen Bench (Jang 2025) | VLA 합성 데이터 평가 (§6.5) |
| 응용 평가 | PAI-Bench (Zhou 2025, arXiv:2512.01989), PAIBench-Transfer | 종합 점수, 변환 평가 (표 10–12) |
| 응용 | IRASim (Zhu 2024, arXiv:2406.14540) | 행동 조건 세계 모델의 Bridge 평가 관례 (§6.6) |
| 응용 | Diffusion Policy (Chi 2023) | 실로봇 정책 구조 (§6.2) |
| 데이터 | Bridge (Walke 2023), DROID, AgiBot (Bu 2025), GR00T, RoboMIND, 1X, OpenX | 로봇 학습 데이터 (표 2) |
| 시스템 | FSDP2, TorchTitan (Liang 2025), Ulysses (Rasley 2020), NATTEN (Hassani 2025) | 학습 인프라 (§4.3) |

## 2. 가까운 대안

| 모델 | 종류 | 보고서 안의 위치 | 비고 |
|---|---|---|---|
| Wan 2.2 27B-A14B, 5B | 공개 영상 확산 | 표 10–11, 그림 6–7 비교 대상 | 2.5 2B는 5B보다 작음 |
| Wan 2.1 14B, 1.3B | 공개 영상 확산 | 표 10–11, 그림 6–7 | |
| Cosmos-Predict1-7B | 전신 | 표 14, 16, 19 | 로봇 행동 조건 비교의 기준 |
| CogVideoX, HunyuanVideo | 공개 영상 확산 | 표 18 (DreamGen) | |
| Ctrl-World, IRASim | 로봇 행동 조건 세계 모델 | 보고서에는 없음 (IRASim은 참조만) | 저장소의 `ctrl_world/`, `worldeval/`과 비교 가능 |

**외부 독립 평가가 있는 행동 조건 Cosmos 모델:** 저장소의 `robofqe/`는 Cosmos 2.5-AC를 기준선으로 씁니다. `worldarena/`는 Cosmos-Predict 2.5 행동 조건판을 정책 평가자로 씁니다. 두 결과 모두 이 보고서에 없는 정책 평가 증거입니다. 자세한 내용은 `03_analysis.md` §4.

## 4. 공식 코드와 저장소

| 저장소 | 관계 | 라이선스 (LICENSE 파일, 2026-11-20 확인) | 별 / 포크 (API) | 마지막 push (API) | 클론한 HEAD | 상태 |
|---|---|---|---|---|---|---|
| [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | 공식 구현 (Predict2.5, action-cond, multiview, 증류, 로봇 정책) | Apache-2.0. README: 모델 가중치는 NVIDIA Open Model License | 1,403 / 200 | 2026-06-08 | `a2c298b` (2026-06-08) | README 상단: Cosmos 3 공지, 활발한 개발 종료, 제한적 유지보수 |
| [nvidia-cosmos/cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) | 공식 구현 (Transfer2.5, robot multiview, auto multiview) | Apache-2.0 (같은 README 구조) | 769 / 139 | 2026-06-30 | `2ff49d0` (2026-06-07) | 같은 Cosmos 3 공지 |
| 프로젝트 페이지 (Predict2.5) | `research.nvidia.com/labs/dir/cosmos-predict2.5` 주소는 영구 이동. 현재 `research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5/` ("Cosmos-Predict2.5 — Cosmos Lab") | — | — | — | — | 요약 페이지. 확인함 |
| 프로젝트 페이지 (Transfer2.5) | `research.nvidia.com/labs/cosmos-lab/cosmos-transfer2.5/` ("Cosmos-Transfer2.5 — Cosmos Lab") | — | — | — | — | 확인함 |
| Hugging Face `nvidia/Cosmos-Predict2.5-2B`, `-14B`, `nvidia/Cosmos-Transfer2.5-2B` | 체크포인트 | NVIDIA Open Model License (README) | — | — | — | 다운로드하지 않음 (가중치 금지) |
| [NVIDIA/Cosmos](https://github.com/NVIDIA/Cosmos) | 후속 플랫폼 Cosmos 3 (README 링크) | 미확인 | — | — | — | 클론하지 않음. 후속 기술 보고서는 확인하지 않음 |

- **클론 위치:** `repo/cosmos-predict2.5/`, `repo/cosmos-transfer2.5/`. `GIT_LFS_SKIP_SMUDGE=1`, `--depth 1`로 받았습니다. 정적 읽기만 했고 설치, 실행, 가중치 다운로드는 하지 않았습니다. 저장소 내용은 신뢰하지 않는 자료로 취급했습니다.
- **라이선스 불일치:** 보고서 초록은 "source code, pretrained checkpoints, and curated benchmarks under the NVIDIA Open Model License"라고 씁니다. 실제로는 코드가 Apache-2.0이고 가중치만 NVIDIA Open Model License입니다(`04` D21).
- **Cosmos 3 공지:** 두 저장소 모두 Cosmos 3으로 옮겨 가라고 안내합니다. 이 보고서의 실행 가능성과 재현 가치는 이 사실을 반영해서 판단해야 합니다(`04` §3).

## 5. 이 보고서가 나온 맥락 (확인 가능한 것만)

- 2025-10-28에 v1이 나왔고 2026-02-24에 v2가 나왔습니다. 저장소 README 릴리스 노트에 따르면 2025년 12월에 Diffusers 지원과 2B 증류 체크포인트가, 2026년 2월에 Robot/Policy 체크포인트(LIBERO, RoboCasa)가 추가됐습니다. 이 보고서에는 그 정책 모델이 없습니다.
- 2026년 6월 두 저장소에 Cosmos 3 안내 커밋이 들어갔고(커밋 날짜 기준), 지금은 유지보수 모드입니다. 따라서 "Cosmos-Predict2.5가 최신인가"에 대한 답은 아니오입니다.

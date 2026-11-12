# 02. 선행 연구, 공개 저장소, 관련 분석

## 1. 계보와 이 블로그의 위치

| 계열 | 무엇 | 이 블로그와의 관계 | 공개 여부 (이번 확인) |
|---|---|---|---|
| Alpamayo-R1 / Alpamayo 1.5 | 추론 VLA (10B), SFT·RL·추측 디코딩 레시피 | 전 세대. AlpaGym의 공개 정책이 이 모델(1.5, 10B, GPU 2장) | 코드와 레시피 공개(alpagym, alpamayo-recipes) |
| Alpamayo 2 Super | 교사 모델. CoC 자동 라벨 생성, SFT 레시피 있음 | 블로그가 증류 교사로 씀 | 레시피 공개(`alpamayo2_sft`). 가중치 여부는 미확인 |
| Cosmos 3 Edge | 2B급 Nemotron 계열 백본 | Alpamayo 2 Edge의 백본 (config의 `cosmos3_edge`) | 모델 카드의 base_model로 표시됨 |
| **Alpamayo 2 Edge** | 2B 추론 VLA, 확산 expert | **본 블로그의 대상** | 가중치 (OpenMDW-1.1), PyTorch 추론 (Apache-2.0), Thor 레시피 (Apache-2.0) |
| AlpaJudge (RB, RF) | 추론 판정기, 참조 기반·참조 없음 | 블로그의 평가·보상 모델 | 가중치 (OpenMDW-1.1), 추론 패키지 (Apache-2.0) |
| AlpaGym | 폐루프 RL, AlpaSim 기반 | 블로그의 "+48.9%" 학습 환경 | 공개 정책은 Alpamayo 1.5 10B만 있음. Edge용 RL 경로는 없음 |
| Qwen-Drive-1.0 (4B) | 블로그 비교 대상 | 비교 상대 | 공개 자료 확인 못 함 |
| LingoQA / Lingo-Judge | 주행 추론 평가 벤치마크·판정기 (Wayve) | 모델 카드가 쓰는 추론 점수 | 공개 벤치마크 (이번에 확인하지 않음) |
| DFlash / DFlash2 | 블록 확산 기반 추측 디코딩 | 블로그의 초안 모델 방식 | z-lab/dflash (MIT) |
| TensorRT-Edge-LLM | NVIDIA 엔진 빌드·런타임 | 레시피가 v0.11.1로 고정 | Apache-2.0 |

블로그는 "Alpamayo 2 Edge는 2B"라고 쓰지만, 공개 config로 세면 전체는 **2.65B**입니다(백본 2.45B, expert 0.2B). 모델 카드도 이 구분을 씁니다.

## 2. GitHub와 Hugging Face 목록

확인일 2026-11-12. `gh api`로 읽은 메타데이터와 커밋이고, 가중치는 받지 않았습니다. 설정 파일 두 개(config.json)와 README만 받았습니다.

| 저장소 | 역할 | 라이선스 | 확인 커밋 (날짜) | 마지막 push | 이번에 본 범위 |
|---|---|---|---|---|---|
| [NVlabs/alpamayo2-edge](https://github.com/NVlabs/alpamayo2-edge) | 2B 추론 PyTorch 코드, 샘플링 설정, 가속 경로 | Apache-2.0 (가중치 OpenMDW-1.1) | `02f55a7` (2026-11-10) | 2026-11-10 | README, `config.py`, `inference.py`, `docs/acceleration.md`, `accel/decode.py`, `pyproject.toml`, 테스트 목록 |
| [NVlabs/alpamayo-recipes](https://github.com/NVlabs/alpamayo-recipes) | Edge의 Thor 배포 레시피(FP8, TensorRT), SFT, 추측 디코딩, RL 그레이더 | Apache-2.0 | `622f9d2` (2026-11-11) | 2026-11-11 | `alpamayo2edge_edgellm` README·`thor/runtime/speculative.py`·`host/quantize.py`·`coc_vocab.json`, `alpamayo1_x_rl`의 `BaseReasoningGrader` |
| [NVlabs/alpajudge](https://github.com/NVlabs/alpajudge) | AlpaJudge RB·RF 추론 패키지 | Apache-2.0 (가중치 OpenMDW-1.1) | `607ae9d` (2026-10-29) | 2026-10-29 | `reference_free/model.py`(점수 식), 예제 노트북(기록된 출력), README 한계 절 |
| [NVlabs/alpagym](https://github.com/NVlabs/alpagym) | 폐루프 RL 프레임워크 | Apache-2.0 | `972d160` (2026-07-21) | 2026-09-03 | 파일 목록. 공개 정책은 `alpamayo_r1`(1.5, 10B)뿐 |
| [NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) | 엔진 빌드·런타임 (레시피가 `v0.11.1` 고정) | Apache-2.0 | 미확인 | 2026-11-06 | 메타데이터만 |
| [z-lab/dflash](https://github.com/z-lab/dflash) | DFlash 블록 확산 추측 디코딩 | MIT | 미확인 | 2026-08-18 | 메타데이터만 |
| [HF nvidia/Alpamayo2-Edge](https://huggingface.co/nvidia/Alpamayo2-Edge) | 모델 카드, config.json | OpenMDW-1.1 | 리비전 `f8951d09` (PyTorch README), `aaaa32c` (Thor 레시피) | 모델 카드 릴리스 2026-11-10 | README, config.json (가중치 미다운로드) |
| [HF nvidia/alpajudge](https://huggingface.co/nvidia/alpajudge) | RB·RF 체크포인트 | OpenMDW-1.1 | 미확인 | 미확인 | README, config.json (가중치 미다운로드) |

라이선스 메모
- 코드는 Apache-2.0이고, 모델 가중치는 OpenMDW-1.1입니다(모델 카드 기준). 두 저장소 모두 이 구분을 명시합니다.
- AlpaJudge 백본은 Qwen3-VL형 config로 보입니다(`model_type: qwen3_vl`, `backbone_kind: alpamayo1`). 상위 백본 라이선스는 확인하지 않았습니다.
- 블로그 본문과 그림은 NVIDIA 저작물로 보고, 그림은 게시하지 않았습니다.

## 3. 같은 저장소의 관련 분석 (블로그 §3 연결)

이 블로그는 AV 전용이지만 **온디바이스 추론의 배포 원리**는 기존 분석과 같습니다. 아래 분석의 산술과 판단 기준을 재사용했습니다.

| 기존 분석 | 가져온 것 | 이 블로그에 적용한 곳 |
|---|---|---|
| `nemotron35_bolt/` | 토큰당 읽는 바이트 = 가중치 + KV, 대역폭 상한(tok/s), MoE 초안 검증 비용, 24 GiB 맥 메모리 판정 | `03` §4의 디코드 상한, `03` §2의 메모리 산술 |
| `nemotron35_super_vl/` | NVIDIA 공식 서빙이 CUDA·NVFP4 커널 전용이라 맥에서 못 돌린다는 판정 기준 | `04` §4의 맥 판정 |
| `no_long_context/` | 비용 모델(처리량·메모리 한계)을 구조별로 계산하는 방식 | 지연 예산과 추론 토큰 수의 관계(`03` §5) |
| `imle_vla/`, `adavla/` | 행동 헤드 지연을 "스텝 수 × 한 스텝 비용"으로 보는 관점, 맥 MPS에서 행동 헤드를 돌릴 때 CUDA 그래프 의존성 | Euler 5단계 대 10단계의 지연 차이(`03` §3) |
| `pacevla/` | 행동 expert를 시간 해상도별로 나누는 설계, GR00T 기반 VLA의 재현 장벽 | 로봇 VLA로 옮길 때의 연결 |

## 4. 구성 요소별 출처와 검증 가능성

| 구성 요소 | 출처 | 검증 방법 | 검증 결과 |
|---|---|---|---|
| 백본·expert 크기 | HF config.json, 모델 카드 | config 치수로 파라미터 계산 | 일치 (2.649B) |
| 입력 구성(22프레임, 픽셀 예산, 20 자세) | README, config.py, 모델 카드 | 정적 대조 | 일치 |
| 샘플링 설정(seed 43, top_p 0.98, temp 0.6, 256, 10단계) | 블로그 코드, `inference.py` 기본값 | 정적 대조 | 일치 |
| 지연 160–190 ms | 블로그 | 레시피 단계 시간과 대조 | 불일치 가능성 (`03` §3) |
| FP8 양자화·추측 디코딩 | 레시피 README, `quantize.py`, `speculative.py` | 정적 대조 | 구현 존재. 초안 모델은 공개 안 됨 |
| AlpaJudge RF 식 | `reference_free/model.py` | 식 대조, 노트북 기록값 재계산 | 일치 (`03` §6) |
| AlpaJudge RB 0.999 | 노트북 마크다운 기록 | 재계산 불가 (체크포인트 실행 필요) | 확인 불가 |
| 폐루프 RL +48.9% | 블로그 | 공개 코드·수치 검색 | 확인 불가 |
| NOMINAL 토큰 | 블로그 | 데이터 준비 스크립트 검색, 추론 코드·어휘 검색 | 학습 라벨은 존재. 추론 빈도는 확인 불가 |

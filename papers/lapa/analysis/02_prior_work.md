# LAPA: 선행연구, 공식 저장소, 관련 분석

분석일 2026-11-21. GitHub 수치는 GitHub API에서 같은 날 읽었습니다.

## 1. 선행연구 계보

| 계열 | 대표 논문 (본문 인용) | LAPA와의 관계 |
|---|---|---|
| 잠재 행동 세계모델 | Genie (Bruce et al., ICML 2024) | LAQ의 원형. LAPA는 같은 틀을 쓰되 (1) 교차 어텐션, (2) 로봇 VLA 사전학습에 씀 (§3.1) |
| 잠재 행동 게임 정책 | Edwards et al. 2018, Schmidt & Jiang 2024 | 잠재 행동으로 사전학습 후 미세조정. 게임 영역 (§2) |
| 영상으로 행동 학습 | VPT (Baker et al., NeurIPS 2022) | 기준선. IDM으로 의사 행동 라벨 생성. LAPA와 같은 VLM 백본을 씀 (§4.2) |
| 영상 생성으로 계획 | UniPi (Du et al., NeurIPS 2023) | 기준선. 확산 모델로 영상 계획 후 IDM으로 행동 추출 (§4.2) |
| 행동 라벨 VLA | RT-2 (Brohan 2023), OpenVLA (Kim 2024) | OpenVLA가 실로봇 비교 기준. 이산 행동 256 bin은 OpenVLA 방식 (§3.3) |
| 영상 토크나이저 | C-ViViT (Villegas et al.), NSVQ (Vali & Bäckström 2022), VQ-VAE (van den Oord 2017) | LAQ 인코더와 양자화 (§3.1, 부록 A) |
| 대형 백본 | LWM (Liu et al. 2024, 7B) | LAPA 백본. 다음 프레임 예측 목적이 행동 이해에 도움이 된다고 본문이 추정 (§4.6) |
| 데이터 | Bridgev2 (Walke 2023), Open-X (2023), SSv2 (Goyal 2017), Language Table (Lynch 2023), SIMPLER (Li 2024) | 사전학습과 평가 |

- 본문은 시각 표현 사전학습(Nair 2022, Dasari 2023), 손 자세 추정 기반 재타깃팅, 궤적 시각화 방법을 관련 연구로 나열합니다 (§2). 이 방법들은 실험 기준선이 아닙니다.
- 본문은 **OpenVLA와 비교해 "같은 백본(LWM)이 아니다"** 라는 점을 밝히지 않습니다. OpenVLA는 Prismatic 백본입니다 (§4.6). 본문은 이 차이를 효율 설명에 씁니다.
- 본문이 기준선에서 뺀 방법: Yang et al. 2024b(비공개), Ko et al. 2024(행동 복제 기준선 아님) (§4.2 각주 4). 따라서 "영상 기반 기준선 전부 대비"는 아닙니다.

## 2. 공식 코드와 관련 저장소

| 저장소 | 역할 | 라이선스 | 최근 push | 스타 | 상태 |
|---|---|---|---|---|---|
| [LatentActionPretraining/LAPA](https://github.com/LatentActionPretraining/LAPA) | **공식 코드** (LAQ, 잠재 사전학습, 미세조정, 배포) | MIT (`LICENSE`, GitHub API `mit`) | 2025-01-22 | 587 | 얕은 복제: 로컬 복제, commit `46aca51` (2025-01-22). 정적 읽기만 |
| [latentactionpretraining.github.io](https://latentactionpretraining.github.io) | 프로젝트 페이지 | 명시 없음 | — | — | 페이지에 코드 버튼은 Hugging Face 링크 |
| [latent-action-pretraining/LAPA-7B-openx](https://huggingface.co/latent-action-pretraining/LAPA-7B-openx) | 가중치 (`params`, `tokenizer.model`, `vqgan`, `laq_openx.pt`)와 사전학습 jsonl | MIT (모델 카드) | 2024-11-22 (수정일) | — | 받지 않음 |
| [LargeWorldModel/LWM](https://github.com/LargeWorldModel/LWM) | 백본 코드 기반 (README 감사 인사) | Apache-2.0 | 2024-10-19 | 7,417 | 참조만 |
| [simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv) | SIMPLER 평가 (LAPA 저장소에 포크 포함) | MIT | 2026-09-30 | 1,187 | 참조만 (저장소 안 `SimplerEnv/` 디렉터리) |
| [openvla/openvla](https://github.com/openvla/openvla) | 배포 코드 참조 (README 감사) | MIT | 2025-03-23 | 7,327 | 참조만 |
| [lucidrains/phenaki-pytorch](https://github.com/lucidrains/phenaki-pytorch) | LAQ 구현 참조 (README 감사) | MIT | 2024-07-29 | 790 | 참조만 |
| [MHVali/Noise-Substitution-in-Vector-Quantization](https://github.com/MHVali/Noise-Substitution-in-Vector-Quantization) | `laq_model/nsvq.py`의 출처 (파일 머리 주석) | MIT | 2026-10-14 | 12 | 참조만 |
| [facebookresearch/polymetis](https://github.com/facebookresearch/polymetis) | 실로봇 텔레오퍼레이션 스택 (본문 부록 B) | MIT | 2021-07-01 | 93 | **보관됨(archived)** |
| [google-research/language-table](https://github.com/google-research/language-table) | Language Table 환경 | Apache-2.0 | 2026-10-07 | 368 | 참조만 |

- 공식 저장소에 **Language Table, 실로봇, Bridge 학습 스크립트는 없습니다.** 학습 스크립트는 `scripts/latent_pretrain_openx.sh`, `scripts/finetune_real.sh`, `scripts/finetune_simpler.sh` 세 개입니다. LAQ 학습은 `laq/train_sthv2.py` 한 개입니다.
- 공식 저장소 README에 2024-11-10 "CoRL 2024 LangRob Workshop Best Paper Award"와 2025-01-22 "ICLR 2025 채택" 소식이 있습니다. 워크숍 수상은 본문에 없습니다.


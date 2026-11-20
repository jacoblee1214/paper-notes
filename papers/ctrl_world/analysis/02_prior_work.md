# 02. 선행연구, 가까운 대안, 코드와 자료

## 1. 계보

Ctrl-World는 두 흐름이 만나는 자리에 있습니다.

- **행동 조건 영상 세계모델** (Finn & Levine 2017 Visual Foresight, Ebert 2018, Yang 2023 Learning Interactive Real-world Simulators, Zhu 2024 IRASim, Quevedo 2025 WPE). 로봇 행동을 입력으로 받아 미래 영상을 만드는 흐름입니다. 대부분 단일 3인칭 시점이고, 정책을 루프에 넣은 롤아웃은 드뭅니다.
- **사전학습 영상 확산 백본** (SVD, Blattmann 2023a; Wan 2025; Cosmos, Agarwal 2025). 텍스트나 이미지에서 영상을 만드는 모델을 로봇 데이터로 적응시키는 흐름입니다.
- **정책 평가용 세계모델의 근접 연구**: WPE (Quevedo 2025, 세계모델 기반 정책 평가), 그리고 Ctrl-World와 같은 시기의 다른 연구들이 있습니다. 본문 §2는 Ren 2025 (Cosmos-Drive-Dreams), Jiang 2025 (EnerVerse-AC), Hafner 2025 (Dreamer 4) 등을 인용합니다.

**Ctrl-World의 기여 위치:** 기존 행동 조건 세계모델에 없던 세 요소(다시점 공동 예측, 자세 기반 메모리 검색, 프레임 단위 행동 조건)를 SVD에 합치고, 이것을 **정책-루프 롤아웃에서 정책 평가와 정책 개선**에 쓴 것입니다. 개별 요소는 Zhu 2024(프레임 단위 조건)와 He 2025(영상 세계모델 조건)에서 가져왔습니다 (본문 §4.1).

## 2. 가까운 대안과 기준선

| 방법 | 시점 | 정책 루프 | 비고 |
|---|---|---|---|
| WPE (Quevedo 2025) | 단일 3인칭 | 본문은 생성 품질만 비교 | 본문 기준선. 다시점 버전 추가 (v3 표 1) |
| IRASim (Zhu 2024) | 단일 3인칭 | 본문은 생성 품질만 비교 | 본문 기준선. 다시점 버전 추가 (v3 표 1) |
| Ctrl-World 단일 시점 변형 | 단일 3인칭 | 예 (표 1 비교는 생성 품질) | 같은 학습 설정, 입력만 줄인 절제 (본문 §5.2) |

## 4. GitHub, 프로젝트 페이지, 관련 자산 (확인 2026-11-20)

| 자산 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (Ctrl-World)** | https://github.com/Robert-gyj/Ctrl-World | MIT (GitHub API, `LICENSE.txt`는 "Copyright (c) 2025 Tsinghua University") | 별 585, 생성 2025-10-09, 최신 커밋 `99fb206` (2026-04-08) | `repo/`에 얕은 clone (정적 읽기만). 저장소 설명은 "ICLR 2026 Paper: Ctrl-World" |
| 프로젝트 페이지 | https://ctrl-world.github.io/ | - | - | 본문 머리말의 페이지. 저장소 README는 `sites.google.com/view/ctrl-world` 링크도 씀 |
| 체크포인트 (Ctrl-World) | https://huggingface.co/yjguo/Ctrl-World | 미확인 | - | README 표 "~8G". **다운로드하지 않음** |
| SVD 백본 | https://huggingface.co/stabilityai/stable-video-diffusion-img2vid | 미확인 | - | README 표 "~8G". **다운로드하지 않음** |
| CLIP 텍스트 인코더 | openai/clip-vit-base-patch32 | 미확인 | - | README 표 "~600M". 다운로드하지 않음 |
| DROID 데이터 | https://huggingface.co/datasets/cadene/droid_1.0.1 | 미확인 | - | README 표 "~370G". 다운로드하지 않음 |
| π0.5-DROID 체크포인트 | openpi (`Physical-Intelligence/openpi`) | 미확인 | - | 코드 `config_eval.py`/`config.py`의 기본 경로는 클러스터 경로 |
| VLAW (후속, 사후 학습) | arXiv:2602.12063 (README 인용) | - | - | 저장소가 세계모델 사후 학습 절차를 포함한다고 명시 |

**공개 범위와 권리:** 코드는 MIT이고 저작권 표기가 대학입니다. 공개본에는 코드를 복제하지 않고 링크와 우리 분석만 씁니다. 논문 PDF는 arXiv 비독점 배포 라이선스이므로 개인용에만 그림을 넣습니다 (`01` §7).

## 5. 재현에 필요한 자원 요약

| 자원 | 크기 (README) | 이 맥에서 |
|---|---|---|
| 세계모델 체크포인트 | ~8G | 다운로드 승인 필요 (1 GB 이상) |
| SVD 백본 | ~8G | 다운로드 승인 필요 |
| CLIP | ~600M | 1 GB 미만이나 SVD와 함께 받아야 함 |
| DROID | ~370G | 불가 (용량) |
| 학습 | 2×8 H100 2–3일 (본문), 1–2노드 A100/H100 (README) | 불가 |
| 추론 한 상호작용 | ~10초 A100, ~5초 H100 (README) | MPS 측정 없음 (`04` §5) |

## 6. 이 분석에서 확인하지 않은 것

- Table 1–2 같은 영상 지표 (PSNR, SSIM, LPIPS, FID, FVD)를 재계산할 시행별 자료나 평가 스크립트가 저장소에 없습니다 (`04` D8).
- 공개된 ICLR 2026 심사 기록 (OpenReview 748bHL2BAv)은 열지 않았습니다. 게재 여부는 공식 ICLR 2026 채택 페이지 (poster 페이지)로 확인했습니다 (README 참고).

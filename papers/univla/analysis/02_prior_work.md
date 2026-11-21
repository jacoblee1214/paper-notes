# 02. 선행연구 계보, 가까운 대안, GitHub와 자료

확인일 2026-11-21. 인용 번호는 논문 참고문헌 번호입니다 (`[n]`).

## 1. 학회와 공개 상태

| 항목 | 확인 결과 | 근거 |
|---|---|---|
| 학회 | **RSS 2025** (Robotics: Science and Systems XXI) | roboticsconference.org 2025 accepted 목록에 제목 "Learning to Act Anywhere with Task-centric Latent Actions"가 있음. 온라인 proceedings `roboticsproceedings.org/rss21/p014.pdf`도 확인 |
| arXiv 표기 | "Accepted to RSS 2025" (Comments) | abs 페이지 |
| 프로젝트 페이지 | **없음** | 공식 README는 "Demo Page (Coming Soon)"만 씀. `univla.github.io`는 404. `opendrivelab.com`에서 UniVLA 전용 페이지를 찾지 못함 |
| 기술 보고서 여부 | 아님 (학회 논문) | — |

## 2. 선행연구 계보 (논문이 기대는 것)

### 2.1 구성 요소별 출처

| 구성 요소 | 논문에서 쓴 것 | 출처 (참고문헌 번호) |
|---|---|---|
| 생성 정책 백본 | Prismatic-7B (DINO+SigLIP 융합 비전, LLaMA-2) | Prismatic VLM [37], LLaMA-2 [75], OpenVLA [39] 방식 |
| 자동회귀 행동 토큰 | 어휘 확장 + 다음 토큰 손실 | RT-2 [10], OpenVLA [39], LAPA [87] |
| 시각 특징 | DINOv2 패치 특징(예측 목표), SigLIP | DINOv2 [62], SigLIP [90] |
| 지시문 인코더 | T5 텍스트 인코더 | T5 [67] |
| 잠재 행동 양자화 | VQ-VAE 목표 | VQ-VAE [76] |
| 예측 공간 학습 | JEPA 방식 "특징 공간 예측" 동기 | I-JEPA [5], V-JEPA [6], DINO-WM [96] |
| 잠재 행동 학습 (픽셀 재구성) | 비교 대상 | Genie [12], LAPO [70], DynaMo [20], LAPA [87], IGOR [16], AdaWorld [27] |
| 행동 청크 | 청크 크기 12 | Zhao 외 (ACT) [93] |
| 시간 앙상블 | 논문 본문에 없음. 코드에만 있음 (`04` D8) | ACT [93] 계열 |
| 이력 입력 | 잠재 행동 토큰을 프롬프트에 덧붙임 | 사고 사슬 [80], 이력 연구 [60, 42, 45], 영상 입력 [91] |
| LoRA 미세조정 | r = 32 (코드) | LoRA [33] |
| 교차 임베디먼트 | 잠재 행동을 공통 어휘로 씀 | Octo [28], CrossFormer [23], 수작업 정렬 [86] |
| 흐름·객체 중심 표현 | 비교 문헌 | ATM [81], Im2Flow2Act [83], SPOT [32], FLIP [26] |
| 내비게이션 | R2R-VLN-CE, NaVid, LLaVA-Nav | VLN [4], VLN-CE [41], NaVid [91], LLaVA [49] 기반 LLaVA-Nav |
| 벤치마크 | LIBERO, CALVIN, SimplerEnv, R2R | LIBERO [48], CALVIN [56], SimplerEnv [47], R2R [4] |
| 데이터 | OXE [63], GNM [72], Ego4D [29], Bridge-V2 [78] | — |

### 2.2 가장 가까운 대안: LAPA [87] 대 UniVLA

- **같은 목표:** 라벨 없는 영상에서 이산 잠재 행동을 만들고, 이를 VLA 어휘로 예측합니다. 논문도 LAPA를 가장 가까운 비교 대상이라고 합니다 [Sec. IV-A].
- **다른 점 (논문 기준):**
  1. 예측 공간: 픽셀 (LAPA) 대 DINOv2 패치 특징 (UniVLA) [Sec. III-A].
  2. 지시문: 없음 (LAPA) 대 1단계에서 T5 조건 (UniVLA) [Sec. III-A].
  3. 두 단계 코드북: 단일 (LAPA) 대 과제 무관 코드북 고정 + 과제 중심 코드북 (UniVLA) [Sec. III-A].
  4. 디코더: 자동회귀 역변환 (LAPA) 대 시각 질의 어텐션 풀링 + 청크 (UniVLA) [Sec. III-C, Tab. IV].
  5. 이력 입력: 없음 (LAPA) 대 잠재 토큰 4개 (UniVLA) [Sec. III-C].
- **섞인 변수:** 실로봇에서 LAPA는 저자가 Prismatic-7B와 같은 디코더로 **재현**한 것입니다 [Sec. IV-A3]. 원래 LAPA 설정과 같지 않습니다. 따라서 LAPA 대비 차이에는 구현 차이가 섞입니다.
- **증명된 것과 아닌 것:** Tab. III는 같은 디코더와 사람 영상 사전학습에서 "과제 중심" 코드(88.7)가 "과제 무관" 코드(56.5)와 Genie 방식(82.3)보다 높음을 보입니다. 이 비교는 잠재 행동 **정의**만 바꾼 통제 실험이라 해석이 가장 깔끔합니다.

### 2.3 사이 관계 (정리)

- Genie [12] → LAPO [70], DynaMo [20] → LAPA [87], IGOR [16] (VLA 어휘로 넣기) → UniVLA (특징 공간 + 과제 조건 + 두 단계 코드북).
- 정책 쪽은 RT-2 [10] → OpenVLA [39] (Prismatic 백본) → UniVLA (같은 백본, 행동 토큰 대신 잠재 토큰).

## 4. GitHub와 자료 (확인 2026-11-21)

| 자료 | 주소 | 라이선스 | 스타 / 포크 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| **공식 코드** | https://github.com/OpenDriveLab/UniVLA | Apache-2.0 (GitHub API, LICENSE 파일도 Apache 2.0) | 1,145 / 72 | 2025-11-19 (기본 브랜치 main) | 얕은 clone (`univla/repo/`, 커밋 `0ab9e9d`, 2025-11-19 "upload training logs for reference"). **정적 읽기만**. 실행, import, 설치 없음 |
| 프로젝트 페이지 | 없음 (README "Demo Page (Coming Soon)") | — | — | — | — |
| 사전학습 VLM | https://huggingface.co/TRI-ML/prismatic-vlms (prism-dinosiglip-224px+7b) | README 기재 | — | — | 미다운로드 |
| UniVLA 체크포인트 | huggingface.co/qwbu/univla-7b (+ latent-action-model, libero/calvin/r2r/simpler-bridge sft) | README 기재 | — | — | 미다운로드 |
| LIBERO 데이터 | huggingface.co/datasets/openvla/modified_libero_rlds | README 기재 | — | — | 미다운로드 |
| arXiv 논문 | arxiv.org/abs/2505.06111 (v1, v2, v3) | arXiv non-exclusive distribution 1.0 | — | v3 2025-11-03 | `paper/paper.pdf` (v3), `paper/paper_v1.pdf`, `paper/paper_v2.pdf` (gitignore) |
| RSS 2025 온라인 proceedings | roboticsproceedings.org/rss21/p014.pdf | — | — | — | 확인 |

- 공식 코드는 Apache-2.0이므로 재사용 가능합니다. 그러나 이 분석에서는 복사하거나 실행하지 않았습니다.
- 공식 코드 저장소에는 `vla-scripts/extern/ego4d_rlds_dataset_builder/`에 별도 LICENSE가 있습니다(Ego4D 빌더).
- 최근 커밋은 학습 로그 업로드(2025-11-19)입니다. 그 이전 커밋 이력은 보지 않았습니다.

## 5. 확인하지 못한 것

- 공식 코드가 논문 v1·v2의 수치와 맞는지(버전별 코드 차이): 커밋 이력을 깊게 보지 않았습니다. 최신 커밋 기준으로만 대조했습니다.
- 사전학습 가중치(univla-7b, latent-action-model)가 README 표와 같은 설정인지: 가중치를 받지 않았으므로 확인 못 함.

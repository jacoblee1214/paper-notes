# 02. 선행연구와 관련 자료

## 1. 계보 한눈에 보기

| 흐름 | 대표 연구 (논문 참고문헌) | CauSyncVLA와의 관계 |
|---|---|---|
| 청크 단위 행동 생성과 비동기 실행 | Action chunking [6, 42], SmolVLA [33], RTC [4], Acting While Understanding [37], AAC [20] | 청크 경계 문제를 푸는 기존 방식. 논문은 이들이 "stale anchor" 문제를 남긴다고 봄 |
| 잠재 세계 모델 | DreamerV3 [11], TD-MPC [12], ManiGaussian [27], SuperMimic [21] | 목표 공간이 픽셀·상태가 아니라 **VLM의 prefix KV 공간**이라는 점이 다름 |
| VLA 안의 기억·예측 | MAP-VLA [19], Latent Bridge [25], EagleVLA [40] | 가장 가까운 대안. Latent Bridge R0는 공개 체크포인트로 기준선 비교 |
| 궤적 재사용과 빠른 흐름 샘플링 | Consistency Policy [31], real-time iteration [7], A2A [14], Δ-flow [45], VLA-Rolling [13] | PFWS는 VLA-Rolling의 꼬리 재사용과 가까움. 차이는 전문가 가중치를 바꾸지 않고 한 스텝 보정 |
| 가속 구조와 스케줄링 | OpenVLA-OFT [16], EfficientVLA·HBVLA [38, 39] | 모델 크기·양자화 축. 논문과 직교 |

## 2. 가까운 대안과 차이

- **Latent Bridge R0 [25].** 추론 사이의 KV 차이를 예측하는 방식입니다. CauSyncVLA는 같은 "KV 공간 예측"을 쓰지만 실행 이력(s, u)을 입력으로 쓰고, 행동 손실로 학습합니다. 논문 Table 1에서 Latent Bridge R0 성공률은 91.80%입니다(RTC와 같은 값).
- **RTC [4].** 청크 경계에 안내 인페인팅을 씁니다. 논문은 RTC를 "10-step + guidance"로 비교하며 Table 1에서 91.80%입니다. 논문은 RTC의 10단계 안내가 남기는 오버헤드를 지적합니다(§4.2).
- **VLA-Rolling [13].** 꼬리 재사용을 쓰지만 백본(LeRobot)과 H/C = 50/10가 다릅니다. Table 1에서 93.40%이지만 평균 계획 지연이 166.47 ms입니다(§4.2). 논문은 이 차이를 명시합니다(§B.4).
- **FlashVLA (arXiv:2608.27384, CoRL 2026 포스터).** 같은 "비동기 추론"을 다루지만 구조 설계(스태거드 시간 버퍼, 인과 마스크)로 풉니다. 두 논문 모두 가속 배수를 모델 단독 지연으로 재는 점이 비슷하며, 3절에서 연결합니다.

## 4. 공식 코드, 프로젝트 페이지, 관련 저장소

조회일 2026-11-22. GitHub API와 Hugging Face API로 확인했습니다.

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | 최근 push | 별 | 커밋·상태 |
|---|---|---|---|---|---|---|
| [TwSphinx54/CauSyncVLA](https://github.com/TwSphinx54/CauSyncVLA) | 공식 코드 (OpenPI 기반 포크, 업데이터 학습·평가) | 공식 (논문 저자 계정으로 보임) | Apache-2.0 (LICENSE, NOTICE, Gemma 약관 별도 파일) | 2026-11-20 | 0 | 생성 2026-11-18. 얕은 복제 `22336bf8d1a4` (2026-11-20 "Link the arXiv paper…"). `repo/`에서 정적 읽기만 |
| [SSSSphinx/CauSyncVLA](https://huggingface.co/SSSSphinx/CauSyncVLA) (HF) | 업데이터 가중치, **LIBERO만 공개** (`libero/*`) | 공식 | 메타데이터 없음 | 2026-11-20 | 다운로드 0 | 받지 않음 (승인 전 가중치 다운로드 금지) |
| [xupan.top/projects/causyncvla](https://xupan.top/projects/causyncvla) | 프로젝트 페이지 (영상, 그림, BibTeX) | 공식 | — | — | — | 정적 HTML 확인. "Under review · 2026"로 표기 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 기반 코드 (π0.5, JAX와 PyTorch 변환) | 상위 저장소 | Apache-2.0 | 2026-08-24 | 14,724 | 받지 않음 |
| [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | 시뮬레이션 벤치마크 (LIBERO-10) | 상위 저장소 | MIT | 2025-03-15 | 2,562 | 받지 않음 |

- 공식 코드 README는 Linux, CUDA GPU, Python 3.11을 전제합니다. `pyproject.toml`은 `torch==2.7.1`을 고정합니다. 논문은 학습을 PyTorch 2.8.0/CUDA 12.8로 적었습니다(D4, `04` 참조).
- 실로봇 업데이터, 대응 전문가, 학습 기록은 저장소에 없습니다(`docs/EXPERIMENTS.md`: "privately held").
- 논문에 적힌 LIBERO-Plus [8]와 기준선 코드(RTC, Latent Bridge, VLA-Rolling)는 조회하지 않았습니다.

## 5. 참고

- 논문 참고문헌 번호는 논문 본문 기준입니다. 저자가 "[4]"로 쓴 RTC와 "[13]"으로 쓴 VLA-Rolling 등은 1번 표의 번호와 같습니다.

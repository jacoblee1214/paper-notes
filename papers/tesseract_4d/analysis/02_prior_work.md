# 02. 선행 연구, 학회 확인, 공식 자료

## 1. 학회 확인 (venue)

**결론: ICCV 2025로 확인합니다.** 근거는 공식 proceedings 목록입니다.

| 근거 | 내용 | 확인 |
|---|---|---|
| CVF Open Access ICCV 2025 | `openaccess.thecvf.com/content/ICCV2025/papers/Zhen_Learning_4D_Embodied_World_Models_ICCV_2025_paper.pdf`, HTTP 200, 12.8 MB, 첫 쪽에 "This ICCV paper is the Open Access version" 워터마크 | 2026-11-20 직접 확인 |
| 공식 저장소 README | "[2025-06-25] TesserAct is accepted to ICCV 2025!" (`repo/README.md` 70행) | 저장소 (사용자 저장소 아님, 저자 저장소) |
| GitHub 저장소 설명 | "ICCV 2025 \| TesserAct: Learning 4D Embodied World Models" | gh API 조회 |
| arXiv abs | 댓글과 저널 참조 없음 | 2026-11-20 |
| 프로젝트 페이지 | 학회 표기 없음 (`project.html`에서 ICCV 문자열 없음) | 확인 |

CVF판은 arXiv v1과 본문 수치가 같고 제목만 바뀌었습니다(`01` §6). 따라서 "학회 확인"은 학회판과 arXiv v1의 대응으로 봅니다.

## 2. 선행 연구 계보

| 계보 | 대표 연구 (논문 참고문헌 번호) | TesserAct와의 관계 |
|---|---|---|
| 비디오 확산 기반 계획·정책 | UniPi / Learning universal policies via text-guided video generation [15], Video language planning [14] | 표 3의 UniPi* 베이스라인. 같은 계열이지만 2D 역동역학을 씀 |
| 비디오 세계모델 (2D 픽셀) | Genie [8], Learning interactive real-world simulators [67], Dreamitate [40], RoboDreamer [79], IRASim [80] | 저자가 "2D 픽셀 세계모델"로 묶는 대상. 4D 표현으로 대체하려는 대상 |
| 3D·기하 세계모델 | 3D-VLA [76] (같은 1저자 Haoyu Zhen, 2024), Aether [57] (기하 인식 세계모델, 합성 데이터만) | 가장 가까운 선행. 3D-VLA는 깊이 추정기 ZoeDepth를 쓰고 목표 상태 3D만 예측 |
| 4D 생성 (SDS 기반) | 4D-fy [3], Align your Gaussians [41], DreamGaussian4D [53], Consistent4D [30] | 최적화 비용이 커서 저자가 RGB-DN 비디오로 대체했다고 주장 |
| 단안 깊이 추정 | Marigold [32], RollingDepth [31] (video depth), ZoeDepth [5] | 실세계 라벨을 만든 추정기 (RollingDepth + Temporal Marigold-LCM) |
| 법선 추정·적분 | DSINE [2], StableNormal [70], Bilateral normal integration [9], ECON [66], Durou et al. [16] | 합성 법선 생성(DSINE depth2normal)과 식 (3)의 법선 적분 계열 |
| 비디오 기반 생성 백본 | CogVideoX [69], OpenSora [77], Latent video diffusion [54, 77] | 기반 모델은 CogVideoX-5B (SFT). OpenSora와 CogVideoX는 베이스라인 |
| 광학 흐름 | RAFT [59] | 식 (5)의 흐름 기반 마스크 |
| 점군 생성·정책 | Point-E [45], 4D Point-E (저자 구현 베이스라인), PointNet++ [49] | 4D Point-E 베이스라인 (표 2). PointNet은 역동역학의 점군 인코더 |
| 행동 복제 | BC-Z [27] (Image-BC 구현 기반), RLBench [26] | 표 3 베이스라인과 과제 환경 |
| 새 시점 합성 | Shape of Motion [62], 3D Gaussian splatting [33] | 표 4 베이스라인 |

## 4. GitHub와 코드 자료 (확인 2026-11-20)

| 자료 | 링크 | 라이선스 | 별 / 포크 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| **공식 코드** | https://github.com/UMass-Embodied-AGI/TesserAct | MIT (LICENSE 2025 Haoyu Zhen) | 413 / 20 | 2025-08-04 | `repo/` 얕은 clone (`d2fe4b1`, 2025-08-03 커밋, 2026-11-20 확인). 정적 읽기만. 42개 파일 |
| 프로젝트 페이지 | https://TesserActWorld.github.io | - | - | - | 정적 사이트. 학회 표기 없음. 코드·논문·데모 링크 |
| 가중치 (v0.1e) | https://huggingface.co/anyeZHY/tesseract | MIT (cardData) | 다운로드 0, 좋아요 7 | 2025-05-01 | 받지 않음. `tesseract_v01e_rgbdn_sft`(40k steps), `tesseract_v01e_rgb_lora`(12k steps) |
| 기반 모델 CogVideoX-5B | https://huggingface.co/zai-org/CogVideoX-5b (구 THUDM) | other (CogVideoX 라이선스) | - | 2024-11-23 | 받지 않음. 저장소의 `i2v_depth_normal_*.py`가 이 라이선스를 명시 |
| 깊이 추정기 RollingDepth | GitHub `anyeZHY/rollingdepth` 포크 (저장소 DATA.md) | 미확인 | - | - | 받지 않음 |
| 법선 추정기 Marigold-LCM | `prs-eth/marigold-normals-lcm-v0-1` | 미확인 | - | - | 받지 않음 |
| 데이터셋 | 저자 README: "no clear plan for releasing the whole dataset" | - | - | - | 4D 데이터셋 미공개. Bridge 전처리 스크립트만 있음 |
| RLBench | https://github.com/stepjam/RLBench | 미확인 | - | - | 저장소에서 RLBench 로더는 주석 처리됨 (`04` D3) |

## 5. 원문 PDF 크기와 보관

- arXiv v1 PDF는 `curl`로 받으면 31,577,322 바이트(약 31.6 MB)입니다. abs 페이지 제출 이력의 "26,393 KB"와 맞지 않습니다. 어느 쪽이든 25 MB보다 크므로 `.gitignore`에 넣었습니다. 재다운로드는 README에 있습니다.
- CVF판 PDF(12.8 MB)는 25 MB보다 작아서 `paper/cvf_iccv2025.pdf`로 보관합니다.

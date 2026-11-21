# 02. 선행연구와 관련 저장소

## 1. 학회 표기

- **ICML 2025 포스터.** 근거: icml.cc 공식 가상 사이트 포스터 페이지 `icml.cc/virtual/2025/poster/46026`(표기 "2025 Poster")와 PMLR 권 267 `proceedings.mlr.press/v267/zhou25t.html`(pp. 79115–79135, "Proceedings of the 42nd International Conference on Machine Learning").
- PDF 메타데이터의 `Subject: Proceedings of ICML 2025`는 스타일 파일 잔여물이며, 학회 채택의 근거로 쓰지 않았습니다. 본문에 학회 표기는 없습니다(arXiv 스탬프만 있음).
- arXiv 댓글 필드와 저널 참조는 비어 있습니다(2026-11-21 확인).

## 2. 계보 (논문 §2와 참고문헌 기준)

| 축 | 선행 작업 | DINO-WM와의 관계 | 근거 |
|---|---|---|---|
| 세계 모델 개념 | Ha & Schmidhuber 2018, Sutton 1991, Todorov & Li 2005 | 개념적 뿌리 | §1, §2 |
| 온라인 잠재 세계 모델 (보상 포함) | Dreamer 계열 (Hafner 2019, 2020, 2024 = DreamerV3), TD-MPC2 (Hansen 2024) | 보상과 작업 정보를 쓰므로 DINO-WM의 "작업 무관" 주장과 대비 | §2, §4.2 |
| 이산 토큰 세계 모델 | IRIS (Micheli 2023) | 토큰 단위 자기회귀 대 DINO-WM의 프레임 단위 인과 | §3.1.2, §4.2 |
| 영상 확산 세계 모델 | AVDC (Ko 2023), Genie (Bruce 2024), 행동 조건 영상 모델 (Yang 2023) | 생성 모델 대 잠재 예측. 계획 정확도 비교 | §2, §4.6, 부록 A.6 |
| 픽셀 공간 예측 기반 계획 | Visual foresight (Finn & Levine 2017), Ebert 2018, Wu 2020 | 목표 이미지 기반 계획의 선행 설정 | §3 |
| 사전학습 시각 표현 | R3M (Nair 2022), MVP (Xiao 2022), I-JEPA (Assran 2023), DINO (Caron 2021), DINOv2 (Oquab 2024), V-JEPA (Bardes 2024) | 패치 특징을 세계 모델 관측으로 쓰는 것이 새로움 | §2, §4.4 |
| 오프라인 세계 모델의 보조 정보 | 전문가 시연 (Pathak 2018, Wang 2023), 키포인트 (Ko 2023, Wen 2024), 역모델 (Du 2023), 밀집 보상 (Ding 2024) | DINO-WM은 이 보조 정보 없이 학습 | §2 |

**가까운 대안 (논문이 직접 비교한 것):** IRIS, DreamerV3, TD-MPC2, AVDC (기준선), R3M, ResNet-18, DINO CLS (인코더 교체). 기준선 대부분은 DINO-WM 원 논문 이후 버전이 나왔을 수 있으므로, 여기서는 논문이 쓴 버전만 대조합니다.

**구성 요소별 출처:** DINOv2 인코더(`facebookresearch/dinov2`), ViT 전이 모델(Dosovitskiy 2021), CEM 계획(부록 A.5), 전치 합성곱 디코더(Razavi 2019). 새로운 부분은 "동결 DINOv2 패치 + 프레임 단위 인과 ViT 전이 + 디코더와 분리된 학습"의 조합입니다.

## 4. GitHub, 프로젝트 페이지, 관련 저장소

조회일 2026-11-21. 별 수, 라이선스, 마지막 push는 GitHub API 응답 기준입니다.

| 저장소 | 관계 | 라이선스 | 별 / 포크 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) | **공식 구현** | MIT (LICENSE, Copyright 2025 gaoyuezhou) | 631 / 93 | 2025-03-24 (API의 updated_at 2026-11-18) | 얕은 복제 `repo/` (commit `0a9492f`, "add checkpoints", 2025-03-24). 정적 읽기만 |
| [dino-wm.github.io](https://dino-wm.github.io/) | 프로젝트 페이지 | - | - | - | 학회 표기 없음. 저자 이름 표기, 익명 아님. 논문이 계획 영상을 올린다고 적은 곳 (영상 내용은 확인하지 않음) |
| [OSF 저장소 bmw48](https://osf.io/bmw48/) | 데이터셋과 체크포인트 (README 링크) | 미확인 | - | - | 받지 않음. README는 체크포인트가 PointMaze, PushT, Wall에만 있다고 적음 |
| [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | 관측 인코더 | Apache-2.0 | 13,560 | 2026-11-05 | 사용하지 않음 (정적 참조) |
| [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) | 기준선 | MIT | 1,004 | 2026-10-20 | 기준선 |
| [eloialonso/iris](https://github.com/eloialonso/iris) | 기준선 | GPL-3.0 | 903 | 2024-10-14 | 기준선 |
| [flow-diffusion/AVDC](https://github.com/flow-diffusion/AVDC) | 기준선 (AVDC) | MIT | 266 | 2024-04-25 | 기준선 |

- 공식 저장소 README의 `[[Code]]()` 링크는 비어 있습니다. 코드 링크는 GitHub 저장소로 연결되는 것을 확인했습니다.
- 공식 저장소의 설정(`conf/`)은 환경별 학습 설정을 담고 있으며, 논문 부록과 여러 곳이 다릅니다(`04` 참조).
- 저장소가 "checkpoints" 커밋을 가졌지만 가중치 파일은 포함되어 있지 않습니다. 체크포인트는 OSF에 따로 있습니다(`04` D16).

## 5. 확인하지 않은 것
- IRIS, AVDC의 저장소 라이선스 전문(GitHub API의 SPDX 표기만 사용).
- 기준선 원 논문의 최신 버전. 논문이 쓴 버전과 같은지 확인하지 않았습니다.

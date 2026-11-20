# Motus 선행연구와 관련 자료

## 1. 계보

| 줄기 | 대표 연구 (논문 참고문헌 번호) | Motus와의 관계 |
|---|---|---|
| 시각-언어-행동 모델 (VLA) | OpenVLA [26, 27], π0.5 [8], X-VLA [60], GR00T N1 [6], RDT-1B [31] | 비교 대상. π0.5와 X-VLA는 실험에 직접 나옴 |
| 영상 생성을 정책에 쓰는 흐름 | UniPi [19], Video Prediction Policy [25], Gen2Act [4], VIDAR [21] | Motus의 VGM 전문가 발상과 가까움 |
| 통합 모델 | UWM [64] (단일 확산 백본, 다섯 분포), F1 [32] (VLA+IDM+영상 예측), Unified Video Action [28] | Motus가 직접 겨냥하는 선행연구. UWM은 처음부터 학습한다는 점에서 차이 |
| 혼합 Transformer | Bagel [18] (이해·생성 공유 self-attention), MoT [30] | Tri-model Joint Attention의 구조적 원형 |
| 잠재 행동 | Genie [9], AdaWorld [22], Rybkin 외 [37], UniVLA [11], LAOM (Nikulin 외) [34], Moto [15] | §4.2의 잠재 행동 VAE 발상. 광학 흐름과 DC-AE 조합은 논문 고유 |
| 광학 흐름 표현 | DPFlow [33], VideoJAM [12] | 행동의 공통 표현으로 흐름을 쓰는 근거 |
| 베이스 모델 | Wan 2.2 5B (VGM), Qwen3-VL-2B (VLM), DC-AE [13] | 모두 기성 모델 (Table 11) |
| 확산 스케줄러 | UniDiffuser (논문 직접 인용 없음, 같은 발상), rectified flow | 독립 타임스텝 학습 방식의 근거 |

확인 방법: 위 표는 논문 §2.1, §2.2, §4.1, §4.2, 참고문헌 목록에서 옮겼습니다. UniDiffuser는 원문에 인용이 없어서 "같은 발상"이라고만 씁니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 원문 위치 |
|---|---|---|
| 다섯 분포 정의 | 자체 정리 (UWM [64]의 다섯 가지 형태를 확장) | §1, §3 |
| Tri-model Joint Attention | 자체 설계, MoT [30] 영감 | §4.1 |
| UniDiffuser형 시간 샘플링 | UniDiffuser 계열 (원문은 "UniDiffuser-like"라고만 씀) | §4.1, Supp. §7 |
| 광학 흐름 → RGB → DC-AE → 14차원 | DPFlow [33], DC-AE [13], 자체 조합 | §4.2 |
| 라벨 없는 데이터 90% + 약한 지도 10% | AnyPos [39]의 작업 무관 데이터 수집 방식 | §4.2 |
| 데이터 피라미드 | 자체 정리, 데이터셋은 [1, 14, 24, 31, 48] | §4.3, Table 12 |
| Action-Dense Video-Sparse | 자체 설계 | §4.1, Fig. 2 |

## 4. 코드와 저장소 표

| 항목 | 주소 | 라이선스 | 상태 (2026-11-20 확인) | 비고 |
|---|---|---|---|---|
| 공식 코드 (프로젝트 페이지 "Code" 버튼) | `github.com/thu-ml/Motus` | Apache-2.0 (GitHub API `license`, `LICENSE` 파일) | ★1321, 포크 84, 최근 push 2026-10-09 (API `pushed_at`), 열린 이슈 35 | 얕은 복제 `repo/` (커밋 `88f6fe3`, 2026-10-10 PR #49 병합). **정적 읽기만** |
| 프로젝트 페이지 하단 "GitHub Repository" | `github.com/motus-robotics/Motus` | 확인 불가 | **404** (GitHub API `Not Found`) | 공개되지 않았거나 잘못된 링크. 정식 코드는 위 항목 |
| 체크포인트 (Stage 1·2·3) | `huggingface.co/motus-robotics/Motus_Wan2_2_5B_pretrain`, `.../Motus`, `.../Motus_robotwin2` | 확인 안 함 | 다운로드하지 않음 (가중치 약 수십 GB, 승인 전 금지) | README §Model Checkpoints |
| 프로젝트 페이지 | `motus-robotics.github.io/motus` | — | 200 | 인용 블록은 `@misc` (학회 표기 없음) |

- 학회 표기: 논문 arXiv 페이지에는 Comments·Journal ref가 없습니다. 프로젝트 페이지 BibTeX도 `@misc`입니다. 학회 확인은 CVPR 2026 공식 virtual 페이지와 CVF Open Access 링크로 했습니다 (`README.md` 참고).



## 5. 위치 판단


# 02. 선행연구와 자료 지도: ManipTrans

## 1. 학회 확인

| 항목 | 확인 결과 | 출처 |
|---|---|---|
| CVPR 2025 채택 | **확인.** CVF Open Access 목록에 "Li_ManipTrans_Efficient_Dexterous_Bimanual_Manipulation_Transfer_via_Residual_Learning_CVPR_2025", pp. 6991–7003 | openaccess.thecvf.com/content/CVPR2025/html/Li_ManipTrans_… (2026-11-22 조회) |
| 포스터 | CVPR 2025 virtual 포스터 페이지 존재 (poster/32548) | cvpr.thecvf.com/virtual/2025/poster/32548 (검색 결과로 확인, 직접 열지 않음) |
| arXiv 댓글 | "Accepted to CVPR 2025" | arxiv.org/abs/2503.21860 (2026-11-22 조회) |
| 저장소 README | "Accepted to CVPR 2025" | `repo/README.md` 5행 |

→ 공식 목록(CVF)으로 확인되었으므로 `venue: "CVPR 2025"`, `venue_note`에 CVF Open Access 목록을 씁니다.

## 2. 선행 계보와 구성 요소별 출처

이 논문은 새 개념보다 **기존 부품을 두 단계로 배치한 것**입니다. 부품마다 출처를 적습니다.

| 구성 요소 | ManipTrans의 쓰임 | 출처 (논문 참고문헌 번호) |
|---|---|---|
| PPO | 두 단계 모두 학습 알고리즘 | [99] Schulman et al. (논문 참조 §3.1) |
| 참조 상태 초기화(RSI)와 조기 종료 | 손 모방 단계, 잔차 단계 | [88] DeepMimic, [89] AMP (§3.2) |
| 커리큘럼 학습 | 손가락·물체 임계 감소 | [8] Bengio et al. (§3.2, §4.2) |
| 잔차 정책 | $a = a^I + \Delta a^R$ | [53] Johannink, [98] Schoettler, [106] Silver (§3.3) |
| 준물리 완화 | 중력·마찰·임계 완화 | [72] QuasiSim, [84] Pang & Tedrake, [85] Pang et al. (§3.3) |
| BPS 형상 표현 | 물체 형상 관측 | [91] Prokudin et al. (§3.3) |
| 볼록 껍질 충돌체 | 물체의 시뮬레이션용 충돌 모델 | [116] Wei et al. (§3.3) |
| MANO 손 모델 | 사람 손 표현, 관절 정의 | [96] Romero et al. (§3.1, 부록 D.1) |
| 관절 손 (Arti-MANO, 22 DoF) | 확장 실험 | [27] D-Grasp (§4.4) |
| 키포인트 정렬 리타겟팅 | Retarget + Residual 기준선 | [94] AnyTeleop (§4.3) |
| 잔차 리타겟팅 | 관련 방법 | [139] DexH2R (§2, 부록 C) |
| Isaac Gym | 시뮬레이터 | [79] Makoviychuk et al. (§4.2) |
| ManifoldPlus | 물체 메쉬 수밀화 | [47] (부록 E) |
| 정책 학습 기준선 | IBC, BET, Diffusion Policy | [33], [101], [25] (§4.7) |
| 데이터: FAVOR | 원격 조작 기반 물체 재배치 | [62] (§3.4) |
| 데이터: OakInk-V2 | 광학 MoCap 기반 양손 작업 | [134] (§3.4) |
| 데이터: GRAB, ARCTIC | 정성 확장 | [107], [32] (§4.1, §4.4) |
| Inspire 손 | 주 플랫폼 | [3] (§3.4) |

**가까운 대안.** 같은 목표(사람 손 궤적 → 로봇 손 궤적)를 다른 방식으로 푼 방법은 세 부류입니다.

- **최적화·준물리 계열:** QuasiSim [72] (정량 비교 없음, 공식 파이프라인 미공개, 저자는 "약 40 h"라고 씀).
- **RL 계열:** RL-Only (PPO로 처음부터 학습, [27] 방식), 리타겟 + 잔차 (DexH2R [139] 방식). 둘 다 논문이 직접 재구현했습니다(§4.3).
- **텔레오퍼레이션 계열:** [26], [44], [45] 등. 실시간 데이터 수집이라 이 논문과 목적이 다릅니다(§2).

## 4. GitHub와 자료 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 별 / 포크 | 마지막 활동 | 비고 |
|---|---|---|---|---|---|
| **공식 코드** | https://github.com/ManipTrans/ManipTrans | GPL-3.0 | 350 / 37 | 생성 2025-03-27, `pushed_at` 2025-10-10 | `repo/` 얕은 clone (HEAD `a3d08cf`, 2025-06-18, "update bibtex"). 정적 읽기만 |
| 프로젝트 페이지 | https://maniptrans.github.io | - | - | - | 정적 HTML 43 KB. 본문 내용은 확인하지 않음 (논문과 README에 같은 링크) |
| 데이터셋 DexManipNet | https://huggingface.co/datasets/LiKailin/DexManipNet | GPL-3.0 (HF 메타데이터) | - | 2025-04-16 | 파일 2개(tar.gz), **총 8.31 GB** (FAVOR 5.24 GB, OakInk-V2 3.07 GB). **받지 않음** |
| 체크포인트 (imitator) | https://huggingface.co/LiKailin/ManipTrans | GPL-3.0 (HF 메타데이터) | - | - | `imitator_ckp/` 24개 파일, **총 0.31 GB**. **받지 않음** |
| QuasiSim (비교 대상) | https://github.com/Meowuu7/QuasiSim | MIT | 80 | 2024-08-10 | 논문 주석 1이 가리키는 저장소. 이 분석에서 열지 않음 |
| Isaac Gym | NVIDIA (Preview 4) | 저장소 설치 안내 | - | - | Linux + NVIDIA GPU 필요, 맥 불가 |
| MANO | https://mano.is.tue.mpg.de | 미확인 (공식 사이트 라이선스 확인 필요) | - | - | 등록 후 다운로드 (저장소 README 안내) |
| SMPL-X, OakInk-V2 | 공식 사이트 (README 안내) | 미확인 | - | - | 등록 후 다운로드 |

공식 저장소 설치 요구 사항: Python 3.8, torch 1.13.1 + cu117, IsaacGym Preview 4, pytorch3d 0.7.3(CUDA 빌드), numpy 1.23.5 (README "Installation"). 맥에서는 CUDA 빌드와 IsaacGym 둘 다 불가입니다.

## 5. 저자 관계

저자 Kailin Li는 DexManipNet의 원 데이터 중 하나인 FAVOR [62]와 OakInk-V2 [134]의 공동 저자입니다(§3.4). 데이터셋과 방법이 같은 연구 그룹에서 나왔다는 뜻이며, 데이터 선택과 평가 세트 구성이 이 그룹의 기존 데이터에 맞춰져 있을 가능성이 있습니다. 독립 검증은 아닙니다.

# 02. 선행연구와 관련 저장소

## 1. 선행연구 계보 (구성 요소별 출처)

LiMA는 새 부품보다 이미 있는 부품을 "느린 Dreamer → 브리지 경계 → 빠른 Refiner" 순서로 이은 구성입니다. 각 부품의 출처는 논문 인용 번호로 적습니다.

| 구성 요소 | 출처 | LiMA에서의 쓰임 |
|---|---|---|
| 통합 WAM 틀 | Cosmos-Policy [17], Cosmos-Predict2 [56] | Dreamer 초기화(2B), 잠재 텐서 규약(슬롯 배치, C = 16, 28 × 28), 학습률 스케줄 형태. 부록 A.2가 "Cosmos-Policy의 잠재 텐서 규약을 따른다"고 적음 |
| 영상 VAE·텍스트 인코더 | Wan2.1 VAE [54], T5-XXL [55] | 다중 시점 영상 토큰화, 언어 임베딩(얼림) |
| DiT와 조건 변조 | DiT [53], FiLM·AdaLN [57] | 언어 변조(Dreamer), 상상 시각 사전의 변조(식 2) |
| 브리지 사후분포 | I²SB [52] | 식 3, 부록 A.1의 μ_t, Σ_t. 아래 표 참고 |
| 이중 시스템·비동기 | RDP [41], DIVA류 상상 협응 [42], Hume [43], CoT-VLA [44], UniVLA [45], RationalVLA [46], DexVLA [47], Last_0 [48], Fast-in-Slow [49], Galaxea G0 [50], HiRT [51] | §2.2가 동기식(System 2 임베딩이 System 1을 조건화)과 비동기식(두 모듈의 갱신 주기 분리)로 나누고, 비동기 설계가 System 2 출력을 정적 외부 조건으로 쓴다고 평가 |
| WAM의 두 갈래 | IDM형 VPP [16], Video2Act [15]; 통합형 Cosmos-Policy [17], LingBot-VA [18], DreamZero [19], RynnVLA-002 [20] | 비교 대상. VPP와 Cosmos-Policy만 실험에 포함 |
| 데이터 수집·로봇 | Metis [9](같은 저자 그룹), CordViP [1] | VIVE + MetaGlove 수집 파이프라인("Following [9]") |
| 기준선 | GR00T N1.6 [21], InternVLA-A1 [22], VPP [16], Cosmos-Policy [17] | 표 1 |

**I²SB 원 논문과의 대조 (원문 `arXiv:2302.05872`을 받아 읽음).**

| 항목 | I²SB 원 논문 | LiMA |
|---|---|---|
| 경계 쌍 | X_0 ~ p_A(깨끗한 데이터), X_1 ~ p_B(·\|X_0)(같은 영상을 열화한 값). Algorithm 1 | X_0 = Z_out^(0)(정답 실행 잠재), X_1 = Z_int(Dreamer 예측). X_1은 같은 장면의 열화가 아니라 다른 모델의 예측 |
| 사후분포 | q(X_t \| X_0, X_1) = N(μ_t, Σ_t), 같은 형태(식 11) | 식 3, 부록 A.1과 같은 형태 |
| 학습 목적 | ε(X_t, t; θ)를 (X_t − X_0)/σ_t에 회귀(식 12) | Ẑ^(0)를 직접 회귀(식 4): x_0 예측 |
| 샘플링 시작 | Algorithm 2: X_N ~ p_B(X_N), 즉 열화 입력 그대로(잡음 없음), DDPM 사후 갱신 | 식 5: N(Z_int, σ_1² I), σ_1 값은 논문에 없음 |
| 스텝 수 | NFE 2–10에서도 성능이 유지된다고 보고(본문 실험 절) | 10스텝(표 3(b)) |

**가장 가까운 대안.**
- *Cosmos-Policy* [17]: 같은 Cosmos-Predict2 2B 위에 영상·행동·가치를 한 잠재 시퀀스로 합칩니다. LiMA의 직접 기준선이며 구조의 출발점입니다. Cosmos-Policy 공개 코드의 기본 학습 설정은 `lr = 1e-4`, 첫 주기 30K 스텝 뒤 `f_min = [0.3, 0.06]`으로 감쇠하는 형태입니다(정적 읽기). LiMA 부록 A.2의 학습률 서술(피크 1e-4, 0.3으로 감쇠한 뒤 0.06으로 고정)과 모양이 같습니다.
- *VPP* [16]: 영상 확산 모델의 예측 표현을 행동 DiT에 조건으로 주는 IDM형입니다.
- *Fast-WAM* [38]: 테스트 때 미래 생성을 없애도 성능이 유지된다는 반대 방향의 접근입니다. LiMA는 인용만 하고 비교하지 않았습니다. "상상(미래 시각 의도)이 필요한가"는 표 2의 w/o Future Visual Intent(양 과제 최저)가 LiMA 구조 안에서 답하는 질문이고, Fast-WAM의 결론과 직접 겨루지는 않습니다.
- *RDP* [41], *Fast-in-Slow* [49], *HiRT* [51], *Galaxea G0* [50], *Last_0* [48]: 비동기 이중 시스템의 선행 설계입니다. LiMA는 이들이 느린 쪽 출력을 "정적 외부 조건"으로 쓴다고 평가하지만(§2.2) 실험에서 이들과 비교하지 않았습니다. 이 서술은 원 논문들과 대조하지 않았습니다.
- *DreamTac* [40]: 같은 저자 그룹의 촉각 WAM입니다(§5 한계에서 촉각을 다음 단계로 제시).

## 2. 코드와 저장소 (GitHub 표)

2026-11-25에 `gh api`와 프로젝트 페이지로 확인했습니다.

| 항목 | 위치 | 상태·라이선스 | 비고 |
|---|---|---|---|
| 공식 코드 | 없음 | 프로젝트 페이지의 코드 칸이 "Code Coming soon" | 코드 대조(D 항목)는 하지 못하고 `04_claim_verification.md`에서 주장 C 항목으로 대조 |
| 프로젝트 페이지 저장소 | github.com/ccdcs/LiMA_repo | 정적 사이트(Academic Project Page Template, Nerfies 기반, 템플릿 CC BY-SA 4.0). 라이선스 파일 없음, 별 0, 마지막 푸시 2026-11-17, 커밋 `6fea7d4`, GitHub 크기 약 68 MB(영상·이미지) | 모델·학습·평가 코드 없음. 복제하지 않음(미디어만 있음) |
| 프로젝트 페이지 | ccdcs.github.io/LiMA_repo | 공개 | 개요 영상, 초록, 표 1과 같은 수치의 표, 과제·일반화 영상, BibTeX(`journal = arXiv preprint`), "CoRL 2026" 표기 |
| 논문 | arXiv:2609.28431 (v1) | CC BY 4.0 (abs 페이지 "view license" 링크) | PDF 약 9.5 MB, 20쪽 |
| Cosmos-Policy | github.com/NVlabs/cosmos-policy | Apache-2.0, 별 900, 마지막 푸시 2026-01-23 | 가장 가까운 기준선, 잠재 텐서 규약의 출처 [17] |
| Cosmos-Predict2 | github.com/nvidia-cosmos/cosmos-predict2 | Apache-2.0, 별 793, 마지막 푸시 2025-10-29 | Dreamer 초기화 [56] |
| I²SB | github.com/NVlabs/I2SB | GitHub API는 NOASSERTION, 별 414, 마지막 푸시 2023-04-13 | 브리지 사후분포 [52] |
| GR00T N1.6 | github.com/NVIDIA/Isaac-GR00T | Apache-2.0, 별 8416, 마지막 푸시 2026-11-17 (릴리스 `n1.6-release` 등) | 기준선 [21] |
| InternVLA-A1 | github.com/Sage-PhysicalAI/InternVLA-A-series | NOASSERTION, 별 575, 마지막 푸시 2026-09-14 | 기준선 [22] |
| VPP | github.com/roboterax/video-prediction-policy | MIT, 별 419, 마지막 푸시 2025-05-17 | 기준선 [16] |
| Wan2.1 | github.com/Wan-Video/Wan2.1 | Apache-2.0, 별 17487, 마지막 푸시 2026-03-05 | VAE [54] |
| Fast-WAM | github.com/yuantianyuan01/FastWAM | NOASSERTION, 별 1672, 마지막 푸시 2026-08-20 | 비교 대상 [38] |
| CordViP | github.com/xuanxuanzzzii/cordvip | MIT, 별 56, 마지막 푸시 2025-11-26 | 같은 그룹의 이전 연구 [1] |

- 체크포인트, 시연 데이터(6과제 × 100), 하드웨어 설계 공개 여부: 논문과 프로젝트 페이지 어디에도 링크가 없습니다. 같은 그룹의 Metis [9] 저장소는 검색으로 찾지 못했습니다.
- 공식 코드가 없어서 `repo/`를 만들지 않았습니다.

## 4. 학회와 버전

- **학회 확인.** (1) 원문 1쪽 하단에 CoRL 2026 표기. (2) corl.org 공식 채택 목록 페이지를 2026-11-25에 직접 열어 같은 제목이 포스터 번호 126으로 있는 것을 확인(OpenReview forum `FsgXevFLkn`). (3) 별도로 수집해 둔 CoRL 2026 채택 목록 사본(2026-11-14 수집)에서도 공식 목록 항목으로 매칭되고 세션은 Poster post-p3(11/11 10:30–12:00). (4) 프로젝트 페이지 배지 "CoRL 2026". arXiv comments에는 학회 표기가 없습니다.
- **버전.** v1만 있습니다(2026-11-25 abs 페이지). 따라서 버전 간 차이 비교는 없습니다.

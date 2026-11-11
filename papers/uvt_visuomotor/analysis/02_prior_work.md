# 02. 선행연구, 가까운 대안, 공식 자료

## 1. 계보

| 줄기 | 대표 연구 | UVT와의 관계 |
|---|---|---|
| 행동 토큰화 | RT-2, FAST, VQ-VLA (논문 [1–3]) | UVT는 토큰 대신 연속 32차원 잠재를 씀 |
| 연속 행동 헤드 | OpenVLA-OFT류 회귀 헤드(논문 [4]), π0 플로우 헤드(논문 [5]), π0.5(논문 [6]) | 논문은 이 헤드들의 출력 목표만 바꿈 |
| 경량 VLA | VLA-Adapter(논문 [7], Qwen2.5-0.5B 계열) | 주 기준선. 공식 코드 MIT |
| 표현 보존 미세 조정 | Knowledge Insulating(논문 [8], Driess 외), 표현 앵커(논문 [9], Dalal 외 2026, Jain 공동저자) | 목표가 아니라 정규화 방향. [9]은 UVT 저자 연구실의 연구 |
| 잠재 행동 모델(LAM) | LAPA(논문 [10]), UniVLA(논문 [11], Apache-2.0), Villa-X(논문 [12]), MVP-LAM(논문 [13]), ViPRA(논문 [14]) | UVT의 z_t 공급원. 논문은 주로 사전학습에 쓰인다고 봄 |
| 정밀도 가중 결합 | PoE(Hinton 2002), 다중 모달 VAE(Wu & Goodman 2018, 논문 [30]) | MVAE 구조의 원형 |
| 벤치마크 | LIBERO(논문 [15], MIT), LIBERO-Plus(논문 [16]) | 평가 |

## 2. 가까운 대안 (같은 문제를 다른 방법으로 푼 것)

1. **보조 헤드 방식:** 행동 헤드 옆에 LAM 코드 예측 헤드를 붙이는 방법. 논문 Table III의 Villa-X와 UniVLA 헤드가 이에 해당합니다. 10k에서 78.8 → 83.2 또는 86.4로 올랐고, 100k에서는 기준과 같거나 낮습니다. UVT(95.0, 98.6)보다 낮습니다. 저자는 "독립 헤드는 상호작용을 강제하지 않는다"고 해석합니다(§IV-E). 이 비교는 **공동 목표의 효과**와 **보조 손실의 효과**를 나누는 증거입니다. 그러나 행동 전용 32차원 대조가 빠져 있어 완전하지 않습니다(`04` §3).
2. **LAM 코드를 행동 대신 쓰는 방법(사전학습 LAM 정책):** LAPA, UniVLA 등. 행동 감독을 없애는 점이 UVT와 다릅니다. UVT는 행동 감독을 유지합니다.
3. **행동 헤드를 바꾸는 방법:** 흐름 매칭을 1단계로 줄이는 IMLE-VLA(`imle_vla`)나 곡률 기반 가속(`adavla`)은 행동 생성 쪽을 건드립니다. UVT는 목표 공간을 바꿉니다. 둘은 결합 가능합니다.
4. **촉각 잠재를 토큰으로 붙이는 방법:** ResTacVLA(`restacvla`)는 촉각 잔차를 VQ 토큰으로 넣습니다. UVT와 같은 "잠재 토큰을 VLA에 붙이기"이지만, 목표가 아니라 입력 측 토큰입니다.

## 3. 이 저장소의 관련 분석 (읽기만 함)

| 폴더 | 논문 | 관련 이유 |
|---|---|---|
| `motion_latent_action/` | Motion-Focused Latent Action (arXiv:2606.18955, IROS 2026) | 사람 영상에서 잠재 행동을 학습해 로봇 VLA에 쓰는 축. UVT의 LAM 공급원과 같은 질문(무엇을 잠재로 볼지)을 다룸 |
| `dreamx/` | DreamX (arXiv:2611.05377) | 영상 확산 모델의 latent-action 토큰을 사람·로봇 공통 인터페이스로 씀. 동역학 토큰의 대안 공급원 |
| `imle_vla/` | IMLE-VLA (arXiv:2609.10915, IROS 2026) | VLA 행동 헤드를 1단계 생성기로 바꾸는 방법. 목표 대 헤드의 축이 다름. 보고 방식(검정, 실물 n=20 등)의 비교 기준 |
| `adavla/` | AdaVLA (arXiv:2608.29208, IROS 2026) | 흐름 VLA 적분을 가속. 행동 생성 쪽, UVT와 직교 |
| `pacevla/` | PaceVLA (arXiv:2610.25961) | 국면별 시간 해상도 디코더. 행동 쪽 구조 변경 |
| `thinkworld/` | ThinkWorld (arXiv:2610.37666) | VLM이 손목 움직임 토큰을 생성하고 미래 특징을 예측하는 보조 목표. 동역학을 보조 감독으로 쓰는 점이 UVT와 같음. 코드 미공개 |
| `openwam/` | OpenWAM (arXiv:2609.07398) | 세계-행동 모델의 설계 선택 비교. 보조 예측이 행동 성능에 주는 효과를 다룸 |
| `temporal_policy/` | Temporal Policy (arXiv:2607.29482, IROS 2026) | 과거 행동으로 행동 생성을 초기화. 시간 정보 활용의 다른 방식 |
| `restacvla/` | ResTacVLA (arXiv:2607.03387) | 잠재 잔차를 VQ 토큰으로 VLA에 넣음. 입력 측 잠재 |

**요지:** UVT는 "보조 감독으로서의 동역학 신호"를 목표 공간에서 다룬 점이 특이합니다. ThinkWorld와 OpenWAM은 보조 예측을 쓰지만 목표 공간을 합치지 않습니다. motion_latent_action과 dreamx는 잠재 행동 공급원입니다. 따라서 UVT의 주장을 검증하려면 "보조 예측 대 공동 목표"와 "잠재 공급원 교체" 두 축을 나눠야 합니다.

## 4. GitHub와 공식 자료 (2026-11-11 확인, `gh api`)

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | 별 | 최근 커밋 (해시, 날짜) |
|---|---|---|---|---|---|
| `DeFisch/UVT` | 논문 공식 구현(MVAE, 파이프라인, VLA 학습과 LIBERO 평가) | 공식(프로젝트 페이지 Code 링크) | **없음**(LICENSE 파일 없음, pyproject는 MIT 분류자 표기) | 3 | `4c46246` (2026-03-11) |
| `OpenHelix-Team/VLA-Adapter` | 기준 VLA(회귀 헤드) | 원저자 저장소로 보임(논문 [7]) | MIT | 2,345 | `23fa0c9` (2026-03-19) |
| `OpenDriveLab/UniVLA` | UniVLA LAM(공개 코드의 LAM 공급원) | 원저자 저장소로 보임(논문 [11]) | Apache-2.0 | 1,145 | `0ab9e9d` (2025-11-19) |
| `microsoft/villa-x` | Villa-X LAM(Table III 헤드) | 원저자 저장소로 보임(논문 [12]) | MIT | 211 | `9b44899` (2026-09-16) |
| `Physical-Intelligence/openpi` | π0.5 원본 구현 | 공식 | Apache-2.0 | 14,604 | `215abfb` (2026-08-24) |
| `Lifelong-Robot-Learning/LIBERO` | 벤치마크 | 공식 | MIT | 2,528 | `8f1084e` (2025-03-15) |
| `sylvestf/LIBERO-plus` | 견고성 벤치마크(LIBERO-Plus) | 공식(논문 [16]) | **없음** | 494 | `4976dc3` (2026-01-21) |

- **UVT 저장소 확인:** 프로젝트 페이지 HTML에서 `github.com/DeFisch/UVT` 링크를 찾았습니다. 커밋 3개, 태그와 릴리스 없음. 논문 실험과 어느 커밋이 맞는지는 확인할 수 없습니다(`04` D24).
- **체크포인트:** UVT 체크포인트 링크는 없습니다. 기반 모델(Prismatic Qwen2.5-0.5B, 약 3 GB)과 UniVLA LAM(약 200 MB)만 README에 링크되어 있습니다.
- **π0.5 + UVT:** 공개 구현이 없습니다. openpi는 원본이고, UVT를 붙인 코드는 없습니다.
- **Hugging Face:** 이 분석에서 확인하지 않았습니다.

## 5. 저자와 연구 맥락

- 저자 Zhenyang Feng과 Unnat Jain은 모두 UC Irvine 소속입니다(논문 1쪽). 논문 참고문헌에 Unnat Jain의 다른 연구(논문 [9] Dalal 외, [25] Patel 외, [22], [23])가 많습니다. 같은 집단의 표현 앵커 연구와 함께 읽으면 맥락이 잡힙니다.
- 본문 감사 문장에 "IROS 리뷰어와 AC"가 있습니다. IROS 2026 채택과 일치합니다.

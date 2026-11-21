# 02. 선행연구와 관련 분석

## 1. 계보

- **Dreamer 계열**: Dreamer(잠재 상상 학습), DreamerV2(이산 잠재), DreamerV3(고정 하이퍼파라미터로 다양한 영역에 적용, symexp twohot 보상 헤드). Dreamer 4는 V3의 상상 학습 루프(보상·가치 헤드, λ-return, twohot)를 가져오고, 세계모델 자체를 영상 생성 트랜스포머로 바꿉니다(§3, §5).
- **영상 생성 world model**: 확산 포싱(Diffusion Forcing, 식 2–4의 기반), shortcut models(원문 [21], 식 3–4), flow matching(식 1). Dreamer 4는 두 기법을 합친 shortcut forcing을 씁니다(§2).
- **Minecraft 에이전트**: VPT(계약자 2.5K 시간 + YouTube 주석 270K 시간, 온라인 RL 194K 시간으로 다이아몬드), Dreamer 3(온라인 1.4K 시간으로 다이아몬드). Dreamer 4는 오프라인 2.5K 시간만으로 다이아몬드를 시도합니다(Table 3).
- **Minecraft world model**: Oasis, Lucid-v1, MineWorld(§4.2, Table 1). Oasis(small)는 공개 가중치가 있으나 본 작업에서 받지 않았습니다.
- **효율 구조**: 공간·시간 분리 어텐션, GQA, 등록 토큰(Table 2에서 각각 어떤 효과를 냈는지 보고).

## 2. 구성 요소별 출처 (원문 인용 기준)

| 구성 요소 | 원문 출처 | 이 논문의 변형 |
|---|---|---|
| 흐름 정합 | 원문 [18, 19] | 식 (1) 그대로 |
| shortcut models | 원문 [21] | 식 (3)을 x-space로 옮김, 식 (7) |
| diffusion forcing | 원문 [14] | 시퀀스의 프레임마다 다른 τ |
| MTP | 원문 [27] | 길이 8, 정책·보상 헤드 |
| PMPO | 원문 [29] | 반대 방향 KL, β = 0.3 |
| symexp twohot | Dreamer 3 [1] | 보상·가치 헤드 |
| MAE 토크나이저 | 원문 [23, 24] | p ~ U(0, 0.9) |
| RMSNorm, RoPE, SwiGLU, QK-Norm, soft cap | 원문 [32–37] | 표준 구성 |
| GQA, 등록 토큰 | 원문 [25, 40] | 동적 모델에 GQA |
| 시간 어텐션 4층마다 | 원문 [39] | 정확히 같은 비율 |

## 4. GitHub와 코드 자료 (확인 2026-11-21)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드** | 없음 | - | - | 프로젝트 페이지에 코드 링크 없음. 저자 GitHub에 dreamer4 저장소 없음 |
| 프로젝트 페이지 | https://danijar.com/project/dreamer4/ | - | - | 영상, arXiv, Twitter 링크. 코드 없음 |
| 비공식 구현 1 | https://github.com/nicklashansen/dreamer4 | MIT | 421★, 2026-07-09 | "Unofficial implementation", PyTorch |
| 비공식 구현 2 | https://github.com/lucidrains/dreamer4 | MIT | 221★, 2026-11-13 | 커뮤니티 구현. 공식 아님 |
| 비공식 구현 3 | https://github.com/edwhu/dreamer4-jax | 없음 | 110★, 2026-07-24 | JAX 구현. 라이선스 미기재 |

비공식 구현은 clone하거나 실행하지 않았습니다. 공식 구현이 없으므로 이 저장소의 검증은 논문 자체의 주장과 표에 한정됩니다(`04_claim_verification.md`).

## 5. 버전과 학회

- arXiv v1 하나. 학회 채택 표기 없음(`README.md`, `01_paper_analysis.md` §6).

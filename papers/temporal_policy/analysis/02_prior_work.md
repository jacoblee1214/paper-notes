# 02. 선행연구와 관계

## 1. 계보

```
확산 정책 (Diffusion Policy, Chi et al., IJRR 2025 [3])   — 가우시안 잡음 원천, DDPM 100단계
  └─ 조건부 확산 (Reuss et al., RSS 2023 [8])              — 목표 조건, classifier-free guidance
플로우 매칭 (Lipman et al., ICLR 2023 [4]) / 정류 플로우 (Liu et al., ICLR 2023 [9])
  └─ 확률적 보간 (Albergo, Boffi, Vanden-Eijnden, JMLR 2025 [6])  — 원천과 목표 임의 결합, 확률 과정 일반화
       └─ 확률적 보간 예측 (Chen et al., ICML 2024 [7])       — 점질량 원천, 점→분포 수송, 해석적 점수
            └─ Temporal Policy (Miller & Jagersand, IROS 2026)  — 원천을 로봇 상태 이력으로 교체
```

논문의 핵심 기여는 "원천의 선택"입니다. 보간·드리프트·손실 틀은 [6], [7]에서 가져옵니다. 저자들이 새로 낸 것은 로봇 LfD에서 이력을 원천으로 쓰는 것, 그리고 그 원천에서의 전달 비용 측정입니다(Sec. II-C, III).

## 2. 가까운 대안

### 2.1 논문이 인용한 것 (Sec. II)

| 방법 | 원천 | 논문의 평가 | 차이 |
|---|---|---|---|
| Diffusion Policy [3] | 가우시안 | 기준선 | 100 NFE, 255M |
| CFM [4] | 가우시안 | 기준선, 10 NFE | 255M, 같은 학습 설정 |
| Streaming Flow Policy [24] | 이전 행동 | "목표가 시점별 주변분포라서 행동 청크의 결합 분포를 못 보고 행동을 섞는다" | 시점별 주변 대 청크 전체 |
| VITA [25] | 잠재 시각 표현 | 디코더 학습에 ODE 역전파 필요 | 원천이 시각 잠재 |
| Trajectory Flow Matching [23] | 관측 이력 | 이중 네트워크로 확산 과정을 매개변수화 | 논문은 같은 원천을 쓰면서 단일 네트워크 |
| Schrödinger bridge / OT-CFM [15]–[20] | 최적 결합 | 로봇에 적용하기 어렵다고 판단 | 결합을 미니배치 OT로 맞춤 |
| Consistency models [10], ManiFlow [14] | — | "직교적(orthogonal)"이라고 씀 | NFE 축소를 증류로 |

### 2.2 논문이 인용하지 않은 가까운 연구 (이 문서 작성자의 판단)

- **ACT의 시간 앙상블링**(Zhao et al., RSS 2023 [27]). 행동을 여러 시점에서 예측해 지수 가중으로 평균합니다. 이 저장소에서 `nestdex/`와 `beyond_implicit_force/`가 다룹니다. TP의 "겹침 d < H"도 인접 청크를 이어 주는 역할에서는 비슷하지만, 앙상블은 **출력 평균**이고 TP는 **생성 원천**입니다. 이 차이 때문에 TP는 샘플 다양성에 영향을 줍니다(`03` §6).
- **Real-Time Chunking**(Physical Intelligence). 이전 청크를 인페인팅으로 이어 붙입니다(`nestdex/analysis/02_prior_work.md` 기재). TP의 원천은 "상태 이력"이고 RTC의 조건은 "이전에 실행한 청크"라서 생성 시작점과 정보가 다릅니다. 원문은 이 문서에서 읽지 않았습니다.
- **가우시안 원천을 쓰는 로봇 플로우의 NFE 축소 연구**(예: `adavla/`, AdaVLA). 곡률로 적분 단계를 줄입니다. 원천은 그대로이고 샘플러만 바꿉니다. TP는 원천을 바꾸고 샘플러는 Euler입니다(`03` §8).
- **정보 없는 원천 대신 평균 예측** (BC 회귀, MSE 정책). TP의 1단계 추론은 정확히 `E[x1 | 이력]`을 내는 회귀입니다(`03` §7). 논문은 이 기준선을 실험에 넣지 않았습니다.

## 3. 이 저장소의 다른 논문과의 관계

| 폴더 | 논문의 방법 | 이 논문과 관계 | 대비 |
|---|---|---|---|
| `chronospolicy/` (arXiv:2610.13929) | 관측 캡처 시각과 행동 실행 시각을 RoPE로 넣고, 학습 때 센서 주기와 지연을 무작위화 | **"시간 정보를 정책에 넣는 다른 방법"**으로 가장 가깝습니다. Chronos는 입력에 시각 좌표를 넣고, TP는 생성 원천에 이력을 넣습니다 | Chronos는 시각 불일치 보정, TP는 전달 거리 단축. 둘 다 이력·시각을 활용하지만 목표가 다름 |
| `adavla/` (arXiv:2608.29208) | 곡률 기준으로 흐름 적분 단계 수를 줄임 | **흐름 단계 수**. TP는 단계 수 10을 주장하고 AdaVLA는 평균 NFE를 역산했습니다 | AdaVLA는 원천 고정 + 샘플러 적응, TP는 원천 변경 + 고정 Euler |
| `openwam/` (arXiv:2609.07398) | 영상 prior + 행동 생성, 잡음 평면과 timestep shift 분석 | **원천·잡음 선택**. OpenWAM 분석은 shift 5 때문에 잡음 쪽 샘플이 25배 조밀하다고 봤습니다(`openwam/analysis/03`) | OpenWAM은 잡음 원천을 유지. 원천을 바꾼 TP는 같은 축(원천의 위치가 학습 신호를 어떻게 바꾸나)에서 다른 해법 |
| `pacevla/` (arXiv:2610.25961) | 시연보다 빠른 행동을 시간 양자화(q = 2)로 생성 | **시간 간격 d**. TP의 d = 14는 원천과 목표 사이 시간 간격이라서, 간격을 바꾸면 전달 비용이 바뀝니다 | PaceVLA는 실행 속도, TP는 학습 쌍의 간격 |
| `beyond_implicit_force/` (arXiv:2607.14578) | ACT 기반 힘 인지 (leader-follower δ, 전류 토크) | 같은 ACT 계열의 **시간 앙상블링**을 공유 | TP는 원천 변경, 이 논문은 입력 특징 변경 |
| `nestdex/` (arXiv:2608.13362) | 다지 손 텔레옵, 시간 앙상블링 Eq. 2의 가중 exp(−m i) | 시간 앙상블링 toy와 비교 근거 | TP는 앙상블 없음 |

IMLE-VLA(arXiv:2609.10915)는 같은 주간에 다른 작업으로 분석 중이라 이 문서에서 인용하지 않았습니다.

## 4. GitHub (2026-11-10 확인)

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | 별 | 최근 커밋 |
|---|---|---|---|---|---|
| `dmiller12/TemporalPolicy` | Temporal Policy 구현(시뮬 학습·평가, 실물 제어 없음) | 공식(논문과 프로젝트 페이지가 링크) | MIT | 2 | `e1e7ee5` (2026-08-18, "add bib") |
| `real-stanford/diffusion_policy` | 기준선 Diffusion Policy와 기반 코드 베이스(논문 감사의 말 명시) | 기준선 원저자 | MIT | 4,721 | `5ba07ac` (2024-12-24) |
| `ARISE-Initiative/robomimic` | 벤치마크(Lift, Can, Square, Transport, Tool Hang) | 벤치마크 원저자 | MIT | 1,602 | `d309eae` (2026-08-09) |
| 프로젝트 페이지 `dmiller12.github.io/TemporalPolicyWebpage/` | 논문, 학위논문, 슬라이드 링크 | 공식 (저자) | 사이트 HTML에 CC BY-SA 4.0 링크가 있으나 템플릿 출처 표기와 함께 있어 사이트 글의 재사용 조건인지 불분명 | — | 날짜 표기 없음(2026-11-10 조회) |
| 학위논문 `DM_thesis_latest.pdf` | 같은 저자 석사 학위논문(79쪽) | 공식 보조 자료 | © Dylan Miller, 2026(표기) | — | 2026-07-24 생성 |

- `gh search repos`로 "temporal policy", "history-initialized", "stochastic interpolant robot"을 검색해 다른 구현은 찾지 못했습니다(2026-11-10).
- 저장소 클론은 `repo/TemporalPolicy/`(얕은 클론, git 제외)에 있고 `.gitignore`에 넣었습니다.
- 학습 데이터는 Robomimic 전체 256 GB(README)이고 체크포인트 `data/0550-test_mean_score=0.969.ckpt`는 README 예시로만 나오며 받지 않았습니다.

## 5. 참고 문헌으로만 쓴 것

- Chen et al., "Probabilistic Forecasting with Stochastic Interpolants and Föllmer Processes", ICML 2024 [7]. 식 4–6의 점수와 추론 SDE 형식이 여기서 옵니다.
- Albergo et al., JMLR 2025 [6]. 보간과 확률 과정의 일반 틀.
- Lipman et al., ICLR 2023 [4]: 조건부 플로우 매칭. 잡음 원천 CFM의 기준.

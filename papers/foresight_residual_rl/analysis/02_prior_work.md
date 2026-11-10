# 02. 선행연구와 관계

## 1. 계보

- **계층 강화학습의 완성 함수:** Dietterich의 MAXQ [17]은 하위 과제 종료 후의 기대 반환(완성 함수)을 학습한다. 논문은 foresight 가치가 "선형 과제 그래프에 제한된 MAXQ 완성 함수"라고 말한다(Sec. II).
- **재보상 하위 과제:** Sutton 외 [18]은 하위 과제의 누적 신호에 부모 과제 보상과 종료 보너스를 넣는다. 논문은 이를 같은 계열로 본다.
- **보상 기계:** Toro Icarte 외 [19]는 단계 사이의 보상 구조를 명시한다.
- **스킬 연결과 초기 집합 정렬:** T-STAR [12](종료 상태 적대 정규화), SCaR [13](양방향 정규화), Sequential Dexterity [14](전이 가능성 함수), Bagaria 외 [15,16](초기 집합 분류기), Lee 외 [23](전이 정책), Meehan 외 [24](옵션 적응). 논문은 이들을 "분포 겹침이라는 대리 목적"으로 묶는다(Sec. II).
- **VLA 잔차 RL:** ResiP [7](확산 정책의 시점별 잔차), ResFiT [20](오프-정책 잔차 미세조정), PLD [21](π0에 probe-learn-distill). 논문은 이들이 개별 과제에만 작동한다고 본다.
- **옵션과 반-MDP:** Sutton 외 [25](옵션 경계의 V_Ω), option-critic [26].
- **시연 학습 기반 단계 연결:** MimicGen [10], Plan-SeqLearn [11].

## 2. 가까운 대안 (이 문서 작성자의 판단)

| 연구 | 관련 이유 | 이 논문과의 차이 |
|---|---|---|
| MAXQ 완성 함수 [17] | 같은 양(하류 기대 반환)을 다룸 | TD로 함께 학습하므로 비정상성이 생기고, 고정 VLA 기반에서는 쓰기 어려움. 이 논문은 몬테카를로 오프라인 추정 후 예측기로 증류 |
| 전역 희소 보상으로 단계 학습 (온라인 MC) | 저자들도 Sec. V-B에서 "불편 추정"이라고 씀 | 분산이 크고 시뮬레이션 비용이 큼. 이 논문의 곱셈 보상은 온라인 MC의 기대값을 예측기로 대체한 것으로 볼 수 있음(03 §3) |
| 잠재 기반 보상 형성 (PBRS, Ng 외 1999) | 단계 경계 가치를 보상에 넣는 가장 표준적인 방법. 논문은 인용하지 않음 | 덧셈 형태이며 최적 정책을 보존함. 곱셈 형태는 보존하지 않음(03 §2) |
| 하류 critic 재사용 | 논문 Sec. V-B가 명시적으로 기각 | 하류 정책의 초기 상태 분포에서 학습된 값이라 상류 종료 상태에 맞지 않음 (논문 주장, 검증 안 함) |
| 초기 집합 정렬 (T-STAR, SCaR 등) | 같은 문제(핸드오프 품질) | 분포 겹침 대리 목적. 성공 확률을 직접 보지 않음 |

## 3. 이 저장소의 다른 논문과의 관계

| 폴더 | 관계 | 비고 |
|---|---|---|
| `thinkworld/` (arXiv:2610.37666) | 세계 모델의 미래 예측 일치를 보상으로 GRPO 학습 | 보상이 모델 자신의 예측에 기대는 구조. 이 논문의 예측기 점수도 학습에 쓰인 같은 예측기로 평가된다는 점이 공통 위험(Goodhart). 그 연구에서는 RL 이득이 작고 설계에 민감하다는 결론을 얻었음 |
| `dexpolicy/` (arXiv:2610.00360) | 학습 중 행동 잡음 σ 스케줄(PPO, GRPO, FPO 적용) | 잔차나 보상 형태보다는 RL 미세조정 설계. 학습 안정성 비교의 관점 |
| `insertion_wm/` (arXiv:2609.28258, 분석 완료) | 잠재 세계 모델로 삽입 과제. 미지 객체 제로샷 56% | 삽입 과제의 다른 경로(모델 기반 계획). 이 논문은 모델 없이 단계 종료 상태의 가치만 예측 |
| `pacevla/` (arXiv:2610.25961) | 행동 청크의 실행 속도 | 보상이 아니라 실행 시간 축. 관련은 약함 |
| `openwam/` (arXiv:2609.07398), `agentic_wam/` (arXiv:2610.30136) | 세계 행동 모델(WAM) | 예측 대상이 영상. 이 논문의 예측기는 단일 이미지에서 성공 확률을 본다 |
| `chronospolicy/` (arXiv:2610.13929) | 관측·행동 시각 조건 | 단계 경계 시간 정의와 연결될 수 있으나 직접 관계는 약함 |

## 4. 저자의 선행 연구 (참고문헌 [28])

- [28] Liu, Zhang, Chang, Boularias (2025), "Failure forecasting boosts robustness of sim2real rhythmic insertion policies." 같은 저자 그룹의 전작으로 보인다. 논문 본문은 [28]을 "상태 기반 RL 삽입 전문가"의 출처로 쓴다(Sec. VI-B). 이 논문에서 실패 예측의 역할은 확인하지 못했다. 원문은 이 세션에서 받지 않았다.

## 5. GitHub와 프로젝트 페이지 (2026-11-10 확인)

프로젝트 페이지 `jaysparrow.github.io/foresight-residual-rl`는 HTTP 404다(GitHub Pages 미활성).

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | 별 | 최근 커밋 |
|---|---|---|---|---|---|
| `JaySparrow/foresight-residual-rl` | 저자 계정의 저장소. 설명에 논문 제목과 "(IROS 2026)". 내용은 **Academic Project Page Template 그대로**(README와 index.html의 PAPER_TITLE, BIBTEX_CITATION_HERE, YOUR_GITHUB_USERNAME 등 플레이스홀더가 남음). 코드·데이터·학습 스크립트 없음 | 저자 계정이지만 코드는 없음 | 없음(license null) | 2 | `f5f1345` (2026-07-17, "Initial commit", 단일 커밋) |
| `JaySparrow/IsaacGymScrew` | 다른 저자 Azulay 외(arXiv:2411.06408)의 IsaacGymInsertion 포크. Robotiq 3지 그리퍼로 교체. README는 "Visuotactile-Based Learning for Insertion" | 이 논문의 코드 아님. 같은 Isaac Gym 삽입 환경의 선행 포크 | LICENSE.txt(README는 Apache-2.0 표기, GitHub 자동 판별 실패) | 0 | `def15c9` (2024-12-28) |
| `osheraz/IsaacGymInsertion` | 위 포크의 원저장소(Azulay 외) | 공식(원 저자), 이 논문과 무관 | 자동 판별 NOASSERTION, README는 Apache-2.0 | 24 | `b0e6b07` (2025-05-20) |
| `JaySparrow/JaySparrow.github.io` | 저자 개인 사이트 소스 | 무관 | — | — | `c208a3a` |

- `gh search repos "foresight residual"`: 위 첫 번째 저장소만 나옴. `"residual RL VLA foresight"`: 결과 없음. `"failure forecasting rhythmic insertion"`: 결과 없음([28] 공개 코드 없음).
- 공식 코드 clone 없음(공개된 코드가 없어 `repo/`를 만들지 않음). 그래서 `04_code_verification.md`가 아니라 `04_claim_verification.md`를 씀.
- 프로젝트 페이지 HTML과 원저장소 README는 보관하지 않음(`.scratch/`에만 임시 보관, 커밋 대상 아님).

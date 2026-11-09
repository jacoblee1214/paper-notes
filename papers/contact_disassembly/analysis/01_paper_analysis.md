# 01. 논문 분석: CONTACT (arXiv:2603.08560v2)

*CONTACT: CONtact-aware TACTile Learning for Robotic Disassembly*. Saka, Hu, Desai, Zhang (공동 1저자), Zhang, Luu, Prince, Zheng, She (교신). Purdue University, Texas A&M University. IROS 2026 채택(abs 주석 "Accepted to IROS 2026"). 원문 8쪽.

## 1. 한 문단 요약

로봇 분해(disassembly)는 접촉이 많고, 힘에 따라 상태가 바뀌는 작업이다. 저자들은 시각 정책(Diffusion Policy 기반)에 촉각을 더하는 세 설정을 **같은 학습 틀**에서 비교한다: Vision Only, Vision + 촉각 RGB 영상(TacRGB), Vision + 촉각 힘장(TacFF, GelSight 광흐름·깊이에서 만든 10×14 격자의 법선·전단 힘). 시뮬(IsaacGym + TacSL) 5과제 S1–S5, 실물 5과제 R1–R5(R4·R5는 변형 부품)에서 TacFF가 10과제 모두 칸 최댓값이다. TacRGB와 TacFF를 함께 넣으면 둘 중 하나만 쓸 때보다 나빠진다(Table V). 이 결과로 "구조화된 힘 표현이 고차원 촉각 영상보다 낫다"고 결론짓는다.

## 2. 문제 설정과 기여 (Sec. I)

- 질문 두 개: (1) 언제 촉각이 시각보다 확실히 나은가, (2) 어떤 촉각 표현이 가장 효과적인가 (Sec. I, p.1).
- 기여 세 개: 과제군 제시(rigid + deformable), 통일된 조건 비교, "표현 선택이 중요하고 단순 결합은 희석한다"(Sec. I 기여 3).
- 주장 강도: 초록과 결론은 "consistently", "critical", "stabilizing signal"처럼 강하다. 실물 표본은 과제당 20회다(Sec. III-C). 이 차이를 `03` §3–§5에서 정량으로 다룬다.

## 3. 방법 (Sec. II)

### 3.1 과제 (Sec. II-A, Fig. 2)
- 시뮬 S1–S5: 헐거운 소켓(S1), 빡빡한 소켓(S2), 뚜껑이 있는 플러그(S3), 평평한 바브(S4), 뾰족한 바브(S5). 모두 강체. S4·S5는 수직 당김으로는 실패하고 먼저 옆으로 밀어야 한다(Sec. II-A.1).
- 실물 R1–R5: R1·R2는 S1·S2와 같은 형상, R3는 S3와 비슷하지만 실물이 약간 작다(Fig. 2 설명). R4는 누르는 변형 탭, R5는 탄성 클립.
- 시뮬 평가에서 위치를 ±5 cm, 자세를 ±10° 흔든다(Sec. III-A.1).

### 3.2 관측과 촉각 표현 (Sec. II-B, II-C)
- 입력: 정면 카메라, 손목 카메라(둘 다 240×320), GelSight R1.5 촉각(TacRGB 240×160 자르기), TacFF 10×14×3(법선, 전단x, 전단y), 엔드이펙터 상태 7차원(위치 3 + 쿼터니언 4). 10 Hz 동기, [-1, 1] 정규화(Sec. III-A).
- 인코더: 모달리티마다 ResNet-18, 2스텝 이력을 특징 차원으로 잇고 엔드이펙터 이력을 붙인다(Sec. II-C, 식 z = Concat(...)).
- 정책: 1D U-Net 확산 모델, 수평 H = 16, 실행 8스텝, 샘플링 DDIM 10스텝(Sec. II-C 끝). 행동은 5차원(dx, dy, dz, dθz, g), yaw만 회전한다(Sec. II-C).
- 학습: 지도 모방학습, 과제당 시연 50개(실물), 400 epoch(Sec. III-A.2).

### 3.3 평가 (Sec. III-A, III-B, III-C)
- 시뮬: 마지막 10 체크포인트(Table I 설명은 "final 10 checkpoints", 본문은 "final 10 training epochs")의 평균, 체크포인트마다 초기조건 50개, 시드 3개 → "1500 rollouts"(Sec. III-B). 본문 1500은 "independent"라는 말이 v1에만 있었다(`02` §5 v1→v2 표).
- 실물: 과제당 20회(Sec. III-C), 실행 시간은 성공 시행만 평균(Table III).

## 4. 결과 (Sec. III)

| 표 | 내용 | 값 요약 |
|---|---|---|
| Table I | 시뮬 성공률 | V / TacRGB / TacFF 평균 60.5 / 61.2 / 65.0 (Fig. 1 헤드라인과 일치) |
| Table II | 실물 성공률 | 평균 52.0 / 37.0 / 68.0. TacFF가 R1–R5 모두 최고(95, 70, 45, 75, 55) |
| Table III | 실물 실행 시간(성공 시행만) | TacFF가 R3·R5에서 길다(67.6 s, 58.0 s). 저자는 "탐색 수정 때문"이라고 해석 |
| Table IV | 조명 교란(R1, R5, 20회) | R5 Dim: V 0%, TacRGB 40%, TacFF 55% |
| Table V | 결합(R3, R5) | TacRGB+TacFF: R3 0%, R5 20%. TacFF 단독은 45%, 55% |

핵심 해석 세 개(논문 원문 근거):
- **TacFF의 이득은 빡빡한 과제·변형 과제에서 크다**(Sec. III-B.1–2, III-C.1): S2 36.9 → 44.3, R4 70 → 75 (TacRGB는 5).
- **TacRGB는 실물에서 평균적으로 해롭다**(Table II): 평균 52.0 → 37.0. R4에서 70 → 5. 논문 본문은 이를 "일관된 이득이 없다"로 완곡하게 쓴다(Sec. III-C.1).
- **결합은 나쁘다**(Table V, Sec. III-D.2): "naive concatenation... 희석".

## 5. 한계와 저자가 밝힌 것 (Sec. III, IV)

- 저자는 "구조화된 융합 (modality-aware attention, 모달리티별 인코더)"이 필요할 수 있다고 쓴다(Sec. III-D.2). 실험은 하지 않았다.
- 실물에서 R1 어두운 조명의 하락폭(55–80 pp)은 본문에서 "noticeable decrease"라고만 썼다(Sec. III-D.1). `03` §4에서 수치로 확인한다.
- 시뮬 TacFF는 시뮬레이터 접촉력 그대로 얻는다(`04` D13). 실물 TacFF는 광흐름·깊이 복원을 거친다(Sec. II-B). 두 힘장의 정보량이 같다는 근거는 논문에 없다.

## 6. 읽으면서 본 빠진 정보

- 학습률, 배치, 옵티마이저, 증강, ResNet 사전학습 여부, 정규화 통계, 시드 값 3개의 구체적 숫자, 시뮬 시연 수(실물 50만 명시), 실물 학습 코드(`04` §2 D12).
- 힘장 계산 방법(광흐름 알고리즘, 깊이 복원 방법, 보정), 실물 TacFF의 단위.
- 표 I의 "bold" 표시는 텍스트로 확인할 수 없다(추출 텍스트에 없음).

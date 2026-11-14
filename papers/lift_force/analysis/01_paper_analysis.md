# 01. 논문 분석: Never Too Late for Force (LIFT, arXiv:2607.14236)

*Never Too Late for Force: Accelerating VLA Post-Training with Reactive Force Injection*. Yi Wang, Wendi Chen, Zimo Wen 외 8인 (SJTU, SII, SUSTech, Zhiyuan College, Noematrix). CoRL 2026 Spotlight (arXiv 댓글 "Accepted to CoRL 2026 (Spotlight)", 공식 저장소 README "CoRL 2026 Spotlight (Top 10%)"). 분석 대상은 **arXiv v3 (2026-10-12)** 입니다. 원문은 `paper/paper.pdf`(18쪽, 5.3 MB)와 `paper/paper.txt`입니다. 버전 차이는 §7에 적었습니다.

## 1. 한 문단 요약

사전학습 VLA(π0.5)는 시각 정보로 행동 청크를 한 번에 만들기 때문에 접촉 순간에 힘이 바뀌어도 청크가 끝날 때까지 반응하지 못합니다. LIFT는 기존 action expert 옆에 **반응 expert**를 붙입니다. 이 expert는 기존 가중치를 복사해 초기화하고, 기준 expert의 토큰을 보는 **shifted causal attention**으로 행동을 하나씩 디코딩합니다. 6D 말단 힘은 **GRU 힘 메모리**로 요약한 뒤 **0으로 초기화한 교차어텐션**으로 반응 스트림에만 더합니다. 그래서 학습 시작 시점의 출력은 원래 정책과 같습니다. 후학습은 **온라인 DAgger**로 합니다. 사람이 고친 온라인 교정 데이터와 기존 시각 데이터를 1:1로 섞어 씁니다. 세 과제(수건 접기, 책 꽂기, 하노이 탑)에서 시각 전용 DAgger보다 피크 점수와 최종 점수가 높다고 보고합니다.

## 2. 문제 설정 (Section 3)

- 1단계: 손으로 든 수집기(iPhone, 힘 센서 없음)로 모은 시각 전용 데이터 $D_v$로 π0.5를 맞춥니다.
- 2단계: Flexiv Rizon 4S(6D 힘 센서)에서 사람이 개입한 교정 데이터 $D_f$를 모아 후학습합니다.
- 목표는 "힘을 늦게(post-training 후반에) 넣되 일반 능력을 버리지 않는 것"입니다.
- 세 가지 목표: **O1** 새 힘 메모리로 청크 안에서 반응하기, **O2** 사전학습 출력 보존, **O3** 두 데이터 원천을 함께 쓰기.

## 3. 방법 (Section 4, Appendix A–B)

### 3.1 O1: 반응 expert와 힘 주입

- **O1.1 반응 action expert.** 기존 π0.5 action expert 옆에 두 번째 action 디노이징 expert를 둡니다. 기존 expert는 청크 안에서 완전 어텐션(모든 $a_{0:H-1}$)을 봅니다. 반응 토큰 $r_i$는 "앞선 반응 토큰 + 뒤쪽 기준 토큰"만 봅니다(Figure 3, ③).
- **O1.2 힘 메모리와 교차어텐션.** 6D 힘 청크 $F_{t:t+H}$를 단층 causal GRU와 선형층으로 $m_{t:t+H}$(폭 $d_m=1024$)로 바꿉니다. 식 (3)에서 쿼리 $q_i$는 $j \le i-L$인 힘 토큰만 보고 $L=3$입니다(Table 1). 이 지연 정렬 마스크는 추론 지연 뒤에도 "그때 쓸 수 있는 힘"만 쓰게 합니다.
- **O1.3 캐시된 시각-언어 prefix.** 청크 시작에서 prefix KV를 한 번 계산하고, 청크 안에서는 최신 힘 이력만 다시 인코딩해 action expert를 다시 평가합니다(Algorithm 1). 느린 VLM 계산을 매 스텝 반복하지 않습니다.

### 3.2 O2: 초기화 동일성

- **O2.1 복사와 shifted causal attention.** 반응 expert의 가중치는 기존 expert의 복사본입니다. $r_{0:i}$의 표현이 $a_{0:i}$와 같아지도록 마스크를 정하면, 같은 노이즈에서 $r_i$는 $a_i$와 같은 문맥을 봅니다(Figure 3 설명). 이 주장은 `analysis/toy_init_equiv.py`로 확인했습니다(`03` §3).
- **O2.2 교차어텐션 출력 0 초기화.** 힘 경로의 출력 투영을 0으로 두므로 0 스텝의 잔차는 정확히 0입니다(Section 4.2). 학습으로 0이 아닌 잔차가 생길 때까지 힘은 행동 출력을 바꾸지 못합니다.

### 3.3 O3: 이종 데이터

- **식 (1).** 기존 스트림과 반응 스트림의 흐름 매칭 손실을 같은 순전파에서 더합니다. 노이즈 $\varepsilon$, $\varepsilon^r$은 독립입니다.
- **O3.2 선택적 힘 마스킹.** 시각 전용 배치는 힘 자리표시자를 쓰고, 힘 메모리를 마스크해서 힘 인코더로 기울기가 흐르지 않게 합니다.
- **O3.3 1:1 균등 샘플링.** RLPD의 대칭 샘플링을 따릅니다. 주요 실험은 고정 1:1입니다(Appendix K.1은 1:2, 0:1과 비교합니다).

### 3.4 파이프라인 (Section 4.4, Appendix C–E)

- 로봇은 10 Hz 위치 제어로 돌고, 6D 힘은 1000 Hz 이상으로 측정한 뒤 10 Hz로 내려 씁니다(Appendix D).
- 온라인 데이터는 개입 프레임만 씁니다(Appendix C, E). 체크포인트는 100 스텝마다 추론 서버로 보냅니다.
- 한 온라인 실험은 약 2–3시간이고 20–30 에피소드를 모읍니다(Appendix J).
- 학습 한 번에 8-GPU 노드(A800) 한 대를 씁니다(Appendix G).

## 4. 실험 (Section 5, Appendix I–L)

### 4.1 설정

- 과제 3개: 수건 접기(단계 점수 0–1), 책 꽂기(단계 점수 0–1, 0.5 단위), 하노이 링 끼우기(이진).
- 평가: 체크포인트당 "3 × 10 자율 롤아웃"(n=30), 95% 구간(Section 5.1, Figure 5 범례 "95% CI (n=30)").
- 비교군(v3 기준 6개): π0.5 w/ Online DAgger, LIFT w/o Reactive Force Injection(단일 프레임 힘), LIFT w/ Offline DAgger, π0.5 w/ Offline Handheld Data, Residual Policy(28M 파라미터 힘 반응 Transformer, 동결 π0.5 위에 얹음), LIFT.

### 4.2 본문 수치 (텍스트에 적힌 값)

| 과제 | 지표 | LIFT | π0.5 w/ Online DAgger | LIFT w/o RFI | 출처 |
|---|---|---|---|---|---|
| 수건 접기 | 피크 점수 | 0.825 (#2.8K) | 0.725 (#3.1K) | 0.95 (#2.8K) | 5.2, 5.3 |
| 수건 접기 | 0.65 도달 시 샘플 | 0.65 (#2.3K) | 0 → 0.725 | 0.925 (#2.4K) | 5.2 |
| 책 꽂기 | 피크 점수 | 0.6 (#4.6K) | 0.4 (#5.6K) | 0.4 (#5.0K, #5.3K) | 5.2, 5.3 |
| 하노이 | 피크 점수 | 0.6 (#1.7K) | 0.3 (#1.4K) | 0.5 (#0.7K) | 5.2, 5.3 |
| 하노이 | 최종 점수 | 0.6 (#2.0K) | 0.2 (#1.6K) | 0.1 (#1.7K) | 5.2, 5.3 |

- 온라인 DAgger 대 오프라인: LIFT w/ Offline DAgger는 책 꽂기에서 0이 됩니다(5.5). 그림에서는 수건 약 0.63, 하노이 약 0.23입니다.
- 일반화(5.4, Figure 6, 최종 체크포인트): 수건 원본 0.8 → 다른 데스크톱 0.825, 파란 수건 0.925, 보라 조명 0.85. 책 원본 0.5 → 다른 책 0.35, 다른 데스크톱 0.1, 보라 조명 0.45. 하노이 원본 0.6 → 다른 데스크톱 0.3, 보라 조명 0.5. 본문은 "대부분의 조건에서 큰 저하 없음"이라고 씁니다(§5 `03` 참조).
- 잔차 기준선(5.6, Figure 7): 최종 점수가 수건 약 0.1 이하, 책 약 0.1 이하, 하노이 0입니다. LIFT 곡선은 같은 그림에 없습니다.

### 4.3 부록의 추가 결과

- K.1 (Figure 10): 수건에서 0:1(온라인만)은 1:1과 1:2보다 나쁩니다. 수치 표는 없습니다.
- L (Figure 11): 일반화 조건 정의. 책 꽂기 조건은 "Gray book, Another tablecloth, Purple light"이고, 하노이는 "Another tablecloth, Purple light"입니다. **Figure 6의 "Another desktop", "Another book"과 이름이 맞지 않습니다**(`03` §1).
- J (Figure 9): 학습 스텝 곡선. 수치는 그래프로만 있습니다.

## 5. 한계 (Section 6)

저자가 쓴 한계는 세 가지입니다. 사람 개입이 데이터 효율을 제한하고 운영자 부담이 큽니다. 실험은 단일 팔 플랫폼 한 종류입니다. 다른 힘 센서는 다루지 않았습니다.

이 분석에서 추가로 보이는 한계는 `04_code_verification.md` §2에 있습니다. 주요한 것은 (1) 학습 시드 정보가 없고 CI는 롤아웃 표본만 반영한다는 점, (2) 힘 메모리 길이가 청크 길이(H=10, 1초)로 고정된다는 점, (3) 힘 지연 L=3(0.3초)이 반응 속도의 하한이라는 점입니다.

## 6. 인용과 배경 (Section 2, 참고문헌)

관련 연구는 세 묶음입니다. 힘 기반 정책(ForceMimic, FoAR, ACP, FACTR, CR-DAgger 등), 빠른-느린 반응 정책(RDP, ImplicitRDP, FA-RDP), VLA에 힘을 붙인 연구(ForceVLA, TA-VLA, ForceVLA2, FAVLA)입니다. 후학습 인프라는 SOP와 RoboPocket을 인용합니다. 계보와 출처 표는 `02_prior_work.md`에 있습니다.

## 7. 버전 차이 (arXiv v1 2026-07-15 → v3 2026-10-12)

v1은 16쪽, v3는 18쪽입니다. 다음이 바뀌었습니다. 수치 표현은 같은 곳도 많지만 실험 설명은 다시 쓰였습니다.

| 항목 | v1 | v3 |
|---|---|---|
| 평가 횟수 | "체크포인트당 10회 자율 롤아웃" | "3 × 10 (n=30)", 95% CI (n=30) |
| 비교군 | 5개: "LIFT w/o Online DAgger"(오프라인만) 포함 | 6개: "LIFT w/ Offline DAgger"로 이름 변경, "Residual Policy" 추가 |
| Figure 1 | 캡션: 결과는 "better final performance than vision-only DAgger" | 캡션: "higher peak and final performance", 우측에 "average peak score" 막대(수건·책·하노이) |
| Figure 7 | 실패 사례 그림 | 잔차 기준선 학습 곡선 |
| 관련 연구 | 본문 5절은 요약, 전문은 "Section A" | 관련 연구는 본문 2절, Section A는 지연 정렬 마스크 세부 |
| 초록 | "preserving its general manipulation knowledge" | "preserving its output at initialization" |
| 데이터 공개 | "Our code and data will be publicly available" | "Our code is publicly available"(데이터 언급 없음) |
| 일반화 | "shows no clear drop" | "shows no substantial degradation relative to its in-distribution behavior" (본문) |
| 본문 수치 | 같음 (수건 0.725·0.825·0.925·0.95, 책 0.6·0.4, 하노이 0.6·0.3·0.2) | 같음 |

- 본문 수치(피크, 최종, 샘플 수)는 두 버전에서 같습니다. 그런데 표본 크기는 10에서 30으로 바뀌었습니다. 같은 수치가 두 표본 크기에 모두 맞는지는 원자료가 없어 확인할 수 없습니다(`03` §1).
- 초록의 "ablations show that reactive force memory and online corrective data are both important" 문장은 두 버전에 같이 있습니다.
- v1의 "LIFT w/o Online DAgger" 실험은 v3에서 "LIFT w/ Offline DAgger"로 이름이 바뀌었고 "fixed force-correction buffer"라는 설명이 붙었습니다.
- 참고문헌: Flexiv TDK가 v1 [17]에서 v3 [32]로 옮겼습니다. FA-RDP(arXiv:2607.28596)는 v1 참고문헌에 없고 v3에 추가되었습니다.

## 8. 재확인 메모

- 라이선스: arXiv abs 페이지의 license 링크는 `creativecommons.org/licenses/by-sa/4.0/`입니다(2026-11-14 확인). CC BY가 아닙니다.
- PDF의 저작권 문구: 1쪽에 arXiv 스탬프와 CoRL 표기만 있습니다. 재인쇄 금지 문구는 없습니다. 그래도 라이선스가 BY-SA이므로 공개 게시에서는 논문 그림을 뺍니다.
- 그림 판독: Figure 5와 Figure 7의 값은 그래프에서 읽은 근사값입니다(판독 오차 약 ±0.03). 본문 수치와 대조할 때만 씁니다.

# 01. 논문 분석: NestDex (arXiv:2608.13362)

- 제목: *NestDex: Nested Policy Learning with Copilot Assisted Teleoperation for Dexterous Manipulation*
- 저자: James Zhao†, Jinhe Tang†, Mingyuan Ba, Weiming Zhi\* (University of Sydney 컴퓨터과학부·Australian Centre for Robotics, Vanderbilt University)
- 공개: arXiv v1 2026-08-13(v1만 있음), 9쪽(본문 8쪽 + 참고문헌) · 프로젝트 https://aus.bot/research/nestdex (PAIR Lab)
- 라이선스: **CC BY 4.0**(arXiv abs 페이지). PDF 안에는 이와 다른 권리 표기가 없습니다("IEEE"는 참고문헌에만 나옴).
- 코드: **공개되지 않았습니다.** 공개된 것은 프로젝트 웹사이트 저장소(`cheese-zj/nestdex-site` @ `63f96e6`)뿐이고, 그 안에 Fig. 9–10의 시행별 데이터 파일(`data/inner-policy-effort.json`)이 있습니다(`04` C5).
- 분석 범위: 본문 전체, 프로젝트 페이지 번들과 공개 데이터 파일

---

## 1. Abstract (한 줄 요약과 주장)

**한 줄 요약:** 다지 손 텔레옵에서 사람은 **팔과 1-DoF 클러치만** 조종하고, 손가락은 미리 학습한 **고유수용감각 기반 "손 기술" 정책(inner policy)**이 만든다. VLM이 과제 단계마다 쓸 기술을 고른다. 이렇게 모은 시연으로 **배포 때는 inner policy 없이** 팔과 손을 함께 제어하는 visuomotor 정책(outer policy)을 학습한다.

논문의 기여는 세 가지입니다(Sec. I).

1. **Copilot 보조 시연 수집:** 되돌릴 수 있는 클러치로 기술의 진행도를 조절하고, 단계마다 기술을 자동으로 고른다.
2. **중첩(nested) 수집과 자율 배포:** inner policy는 수집에만 쓰고, outer policy는 그 시연으로 따로 학습해 혼자 실행한다.
3. **실물 수집→자율 평가:** 6개 과제에서 수집 신뢰성·효율, 자율 정책 학습, 압축 손 행동 표현(H-VAE), 온라인 접촉 반응 실행을 평가한다.

---

## 2. Method

### 2.1 전체 구조 (Fig. 2, Sec. IV)

```
[inner 학습]  여러 카메라로 사람 손 3D 재구성 → AnyTeleop식 벡터 리타겟팅(Huber, Eq. 3) → 로봇 손 실행
             기록 (q_t, e_t) = 손 관절 위치 20 + 관절 effort 20 → 기술마다 시연 10개 → 기술마다 inner policy 하나
             proprio Transformer(enc 4, dec 1): 최근 30스텝 (q, e) → 다음 30스텝 관절 위치 chunk (Eq. 4)
                     ▼
[copilot 수집] leader 팔 관절 → follower 팔 관절 (직접 대응)
             leader 클러치 c_t → 진행도 p_t (Eq. 6) → 목표 index s_t (Eq. 7) → 현재 index r_t를 한 칸씩 따라감 (Eq. 8)
             r 증가: inner policy를 최신 proprio로 질의 + temporal ensembling(Eq. 2) / r 감소: 버퍼의 이전 명령 재생
             r = 0까지 되돌리면 VLM(손목 카메라 + 기술 설명 목록)이 다음 기술을 고름
                     ▼
[outer 학습]  과제당 시연 20개 → H-VAE(20 → 10, [128, 64], 과제별) → 행동 = [팔 관절 7, 손 latent 10]
             visuomotor Transformer(enc 4, dec 1) + DINOv3 임베딩, 256×256 RGB → 100스텝 chunk, BC (Eq. 10)
             배포: 매 스텝 질의 + temporal ensembling, 손 latent는 H-VAE decoder로 관절 명령
```

### 2.2 Inner policy (Sec. IV-A)
- **기술 시연:** 보정된 여러 카메라로 손 keypoint를 삼각측량해 단일 시점의 깊이 모호성과 가림을 줄이고, AnyTeleop [4]의 벡터 리타겟팅 잔차를 Huber 벌칙으로 바꿨습니다(Eq. 3).
- **학습:** 기술마다 시연 N = 10개, 모든 유효 sliding window를 씁니다. 입력은 관절 위치와 effort뿐이고, 이미지나 물체 정체는 받지 않습니다(Fig. 3).
- **설정:** AdamW, 20,000 step, batch 256, lr 10⁻⁵ (Sec. V Training setup).

### 2.3 클러치와 기술 선택 (Sec. IV-B, Eq. 5–8)
- T_max = 시연 중 가장 긴 길이(Eq. 5). 클러치 값을 [c_start, c_end]로 정규화해 p_t ∈ [0, 1]을 얻고(Eq. 6), s_t = ⌊p_t (T_max − 1)⌋(Eq. 7).
- 실행 index r_t는 제어 주기마다 **최대 한 칸**씩 s_t를 따라갑니다(Eq. 8). 앞으로 가면 정책을 다시 질의하고, 뒤로 가면 버퍼의 이전 명령을 실행하고 지웁니다. 되돌릴 때는 앙상블 예측을 비우고 버퍼는 남깁니다.
- **VLM 선택:** 시작할 때와 사용자가 클러치를 끝까지 되돌려 r = 0이 될 때만 호출합니다. 그래서 기술 실행 중에는 기술이 바뀌지 않습니다. 어떤 VLM인지는 논문에 없습니다.

### 2.4 Outer policy와 H-VAE (Sec. IV-C)
- H-VAE는 손 관절 명령 20차원만 10차원으로 압축하고, 팔 명령은 관절 공간 그대로 둡니다. 학습 뒤에는 posterior 평균 μ_φ를 목표로 써서 샘플링 잡음을 없앱니다(Eq. 9).
- outer policy는 (v_t, s_t) → 100스텝 chunk를 BC로 학습합니다(Eq. 10). 설정은 AdamW, 50,000 step, batch 8, lr 10⁻⁵입니다.

### 2.5 Temporal ensembling (Sec. III, Eq. 2)
- ACT [27]와 같은 식입니다. 같은 실행 시각에 대한 예측들을 오래된 것부터 i = 0, 1, …로 놓고 w_i = exp(−m i)로 가중 평균합니다. **가장 오래된 예측의 가중치가 가장 큽니다.** m 값은 논문에 없습니다(`03` §3).

---

## 3. Experiments

### 3.1 설정 (Sec. V)
- **하드웨어:** 팔·손 한 쌍마다 leader(7-DoF Piper Nero 팔 + 1-DoF 클러치)와 follower(같은 팔 + 20-DoF WujiHand I + 손목 카메라). inner policy는 100 Hz로 실행합니다. outer policy의 제어 주기는 적혀 있지 않습니다.
- **과제 6개(Table I):** 단일 팔 4개(Tongs Transfer, Bottle Disposal, Dual-Object Transfer, Ingredient and Pot Transfer), 양팔 2개(Toast Preparation, Binder Filing).
- **기준선:** 같은 플랫폼에서 AnyTeleop [4](카메라로 사람 손 자세를 로봇 손에 매핑). 같은 작업자가 두 인터페이스에 익숙해진 뒤 수행했습니다.
- **지표:** 과제 설명의 모든 단계를 끝내야 성공. 성공 1건당 시간 = (성공·실패 시행 전체 시간) / 성공 수. NestDex는 기술마다 시연 10개를 모은 시간을 성공 시연 수로 나눠 더합니다.

### 3.2 Q1: 시연 수집 (Table II)

| 과제 | Copilot 시간(s) / 성공 | AnyTeleop 시간(s) / 성공 |
|---|---|---|
| Tongs Transfer | 44.33 / 20/20 | N/A / 0/20 |
| Bottle Disposal | 41.37 / 20/20 | 88.88 / 10/20 |
| Dual-Object Transfer | 36.19 / 20/20 | 121.63 / 6/20 |
| Ingredient and Pot Transfer | 43.26 / 20/20 | 55.29 / 15/20 |
| Toast Preparation | 327.46 / 20/20 | N/A / 0/20 |
| Binder Filing | 221.80 / 20/20 | N/A / 0/20 |

- **성공률 차이는 Ingredient를 빼면 확실합니다.** Fisher p는 Tongs·Toast·Binder 1.5×10⁻¹¹, Dual-Object 3.3×10⁻⁶, Bottle 4.4×10⁻⁴, Ingredient 0.047입니다(`04` C1).
- **"더 빠르다"는 시행이 빨라서가 아니라 덜 실패해서입니다.** 성공 1건당 시간 × 성공률로 시행 1회당 시간을 구하면 AnyTeleop 44.4 / 36.5 / 41.5 s, Copilot 41.4 / 36.2 / 43.3 s로 0.96–1.07배입니다. 시간 이득은 거의 전부 성공률에서 나옵니다(`03` §1). 실패 시행과 성공 시행의 길이가 같다는 가정이 필요하고, Copilot 쪽은 기술 시연 수집 상각분까지 포함한 값입니다.

### 3.3 Q2·Q3: 자율 정책 (Table III, 단일 팔 4과제, 20 rollout)

| 조건 | Tongs | Bottle | Dual-Object | Ingredient |
|---|---|---|---|---|
| Copilot, no H-VAE | 65% | 60% | 80% | 85% |
| Copilot, H-VAE | **100%** | **75%** | **90%** | **100%** |
| AnyTeleop, no H-VAE | N/A | 40% | 20% | 75% |

- **H-VAE 효과는 합치면 유의하지만 과제별로는 하나만 유의합니다.** 4과제 합산 73/80 대 58/80, Mantel–Haenszel p = 0.0018(OR 4.3)입니다. 과제별로는 Tongs만 유의하고(13/20 → 20/20, Holm p = 0.033), 나머지는 p = 0.23–0.66입니다. 본문의 "consistently improves"는 방향이 일관적이라는 뜻으로만 맞습니다(`04` C3).
- **수집 방식 효과(H-VAE 없이)도 Dual-Object에서만 유의합니다.** 16/20 대 4/20, p = 3.6×10⁻⁴. Bottle 12 대 8(p = 0.34), Ingredient 17 대 15(p = 0.70)입니다. 3과제 합산 MH p = 5×10⁻⁴입니다.
- **빈 칸이 있습니다.** AnyTeleop + H-VAE 조건이 없어서 "H-VAE가 어떤 데이터에서도 돕는가"는 알 수 없습니다. **양팔 2과제는 자율 정책 결과가 없습니다.** Fig. 1과 Fig. 4의 양팔 과제는 수집만 보여 줍니다.

### 3.4 Q4: 온라인 실행 (Fig. 7, 9, 10)
- 같은 inner policy로 물병 잡기를 세 조건에서 10회씩 시험했습니다. 고정 명령 재생 3/10, 닫힌 루프·앙상블 없음 7/10, 닫힌 루프·앙상블 9/10입니다.
- 본문의 p = 0.0198은 고정 재생 대 앙상블의 양측 Fisher 정확검정 값과 정확히 같습니다(0.01977). 다만 세 쌍을 비교했으므로 Holm 보정하면 0.059입니다. 닫힌 루프 두 조건의 차이(7/10 대 9/10)는 p = 0.58로 구분되지 않는다고 저자들도 씁니다(`04` C4).
- 앙상블 없음의 P95 jerk가 2.30배 크다는 값(p = 1.8×10⁻⁴)은 Fig. 7(c)의 0.21 s 창 값과 맞습니다. 창을 0.11, 0.31 s로 바꾸면 1.94, 2.03배이고, 명령 제한 전 기준으로는 2.84–3.62배입니다.
- **공개 데이터로 다시 계산했습니다.** 시행별 유지 노력 중앙값과 정확 Mann–Whitney p(0.117, 0.017, 0.400, 성공끼리 비교 0.0012)가 모두 재현됩니다. 30회를 합치면 **닫힘 뒤 유지 노력 하나로 성공을 AUC 0.91로 가를 수 있습니다**(문턱 0.45에서 90% 정확). 실패 11회 중 9회가 문턱 아래, 즉 대부분 쥐는 힘이 떨어지는 시행입니다(`03` §4).

### 3.5 Q5·Q6: 정성 평가
- **Q5(Fig. 3):** 하나의 잡기 정책이 학습에 쓴 물체 4개에서 서로 다른 손 모양을 만듭니다. 저자들도 "처음 보는 물체로의 일반화가 아니라 학습한 접촉 조건 사이의 적응"이라고 한정합니다. 시연 10개를 20,000 step × 256으로 학습하면 500–3,600 epoch(시연 길이 2–10 s 가정)라서, 사실상 암기 영역입니다(`03` §5).
- **Q6(Fig. 4, 11):** Toast와 Binder에서 선택기가 단계에 맞는 기술을 고르고, 이미 쓴 기술을 다시 씁니다. 선택 정확도나 잘못 고른 횟수는 보고하지 않습니다.

---

## 4. Limitations

### 4.1 저자가 밝힌 한계
Q5는 학습 물체 4개 안에서의 적응이고 새 물체 일반화가 아니라고 밝힙니다(Sec. V-C). 닫힌 루프 두 조건의 성공률 차이는 현재 표본으로 확정할 수 없다고 씁니다. 그 밖의 한계 절은 없습니다.

### 4.2 분석하면서 추가로 발견한 한계와 의문점
1. **효율 이득의 정체는 신뢰성입니다.** 시행 1회당 시간은 두 방식이 거의 같습니다(0.96–1.07배). "작업자 부담을 줄인다"는 주장을 직접 재는 지표(작업 부하 설문, 작업자 여러 명)는 없습니다. 작업자는 한 명입니다.
2. **기준선이 하나뿐입니다.** 가장 가까운 선행 연구인 shared autonomy [12](VR 텔레옵 + 자율 손 VLA)를 인용만 하고 비교하지 않았습니다. 장갑·VR 기반이나 RL 기술 버튼(DexScrew식) 비교도 없습니다(`02` §3).
3. **자율 정책 평가가 단일 팔 4과제뿐입니다.** 양팔 과제의 "collection-to-autonomy"는 시연되지 않았습니다. AnyTeleop + H-VAE 칸도 없습니다.
4. **작은 표본의 과제별 주장이 많습니다.** H-VAE "일관된 개선"은 4과제 중 1과제만 유의하고, Q4는 조건당 10회입니다.
5. **재현에 필요한 값이 빠져 있습니다.** temporal ensembling의 m, VLM 모델 이름, outer policy 제어 주기, 단일 팔 과제마다 쓴 기술 수, H-VAE 복원 오차, 기술 시연 길이(T_max), 추론 지연이 없습니다.
6. **공개 데이터의 조건 이름이 본문과 다릅니다.** 앙상블 조건의 id가 `online_rtc`, 앙상블 없음이 `online_no_rtc`입니다. 본문은 RTC(Real-Time Chunking)가 아니라 ACT식 지수 가중 평균(Eq. 2)을 썼다고 씁니다. 이름만의 문제인지 다른 알고리즘인지는 코드가 없어 확인할 수 없습니다(`04` C6).
7. **Q4 시행은 조건별 블록으로 진행됐습니다.** 공개 데이터의 episode 번호를 보면 조건이 2–4회씩 묶여 번갈아 진행됐고(10–13 고정 재생, 15–18 앙상블 없음, 19–22 앙상블 …), 번호 14, 24, 30, 31, 40은 빠져 있습니다. 무작위 순서가 아니라서 시간에 따른 변화(손 온도, 물병 상태 등)와 조건이 섞일 수 있습니다(`04` C5).
8. **클러치는 기술을 시연 속도보다 빠르게 실행할 수 없습니다.** 실행 index가 제어 주기마다 한 칸만 움직이므로(Eq. 8), 기술 하나에 최소 T_max / 100 s가 걸립니다. 작업자는 기술을 늦추거나 되돌릴 수만 있습니다(`03` §2).

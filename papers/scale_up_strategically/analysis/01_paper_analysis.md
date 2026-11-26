# 01. 논문 분석: Scale Up Strategically

*Scale Up Strategically: Learning Compositional Generalization via Bias-Aware Evaluation and Data Collection for Robotic Manipulation* (Yu Qi, Zhang Ye, Xinyi Xu, Yuxuan Lu, Amitoj Sandhu, Boce Hu, Haojie Huang, Jonathan Tremblay, Lawson L.S. Wong; Northeastern University, NVIDIA). arXiv:2607.21582, cs.RO/cs.CV. 분석 대상은 **v1 하나**(2026-07-23, 20쪽)입니다. 2026-11-26 확인 시점에 v2 이후는 없었습니다. CoRL 2026 포스터(corl.org 공식 채택 목록: Poster post-p2, 11/10, 포스터 No. 101).

## 0. 한 문단 요약

로봇 정책이 언어를 "이해"하지 않고 눈에 띄는 단서(예: 색)를 지름길로 쓴다는 관찰을, 지시를 **다섯 인자(factor)**(verb, color, object, size, spatial)로 쪼개어 정량화한다. 두 인자를 훈련 데이터에서 짝지어 상관시킨 뒤(대각선 칸만 훈련), 그 상관을 깨는 평가 지시에서 정책이 어느 인자를 따르는지 VLM 판정으로 센다(FDR, 식 1). 쌍별 FDR을 Copeland 점수로 합쳐 전역 순위 FDH(식 3)를 만들고, 여섯 정책에서 대체로 color ≥ object ≥ spatial ≥ verb ≥ size가 나온다고 보고한다(표 1, 2). 이 순위를 이용해 "덜 접지된 인자 쪽으로 시연을 재배분하는" 수집 전략 V를 제안하고, 시뮬레이션(π0, π0.5, GR00T-N1.7)과 UR5 실로봇(GR00T-N1.7)에서 L과 Random 기준선보다 낫다고 보고한다(그림 4, 표 3).

## 1. 문제 설정 (§3, 부록 A)

- 시뮬레이션: ManiSkill 3의 탁상 다중 과제. Panda 팔, 에이전트 뷰와 손목 뷰 256×256 (부록 A).
- 템플릿: "[Verb] the [Size] [Color] [Object] on the [Spatial Attribute] of the table". 어휘는 verb 6(grasp, lift, push, pull, rotate, slide), color 6, object 6(cube, sphere, cup, car, pyramid, star), size 6(small, large, smaller, larger, smallest, largest), spatial 5(left, right, middle, front, behind). 지시 공간 6×6×6×6×5 = **6,480** (§3, 표 4. 재계산 일치, `03` K1).
- 장면마다 무작위 방해물 최대 3개.

## 2. 평가: FDR과 FDH (§4, 부록 B)

### 2.1 인자쌍 상관 실험 (§4.1, 그림 3)

- 쌍 (f1, f2)마다 정책 하나를 미세조정한다. 훈련 지시는 대각선(각 f1 값이 f2 값 하나와만 짝지어짐)이고 나머지 인자는 무작위다. 예: color–object는 red+cube, yellow+sphere, … 6쌍을 "Grasp/Lift" 두 동사로 만든 12개 지시(부록 B). 훈련 시연 **300개**.
- 평가: 대각선이 아닌(OOD) 칸에서 롤아웃 **400개**를 균일 추출.
- 장면 구성 두 가지(그림 3). (a) verb 대 객체 중심 인자: 같은 색 물체 두 개(지시된 대상과, 그 동사가 훈련에서 짝지어진 물체). (b) 객체 중심 인자 둘: 둘을 동시에 만족하는 물체가 없는 불가능 지시(예: "grasp the blue cup"을 파란 별과 빨간 컵이 있는 장면에).
- 판정: Gemini-2.5-Flash가 에이전트 뷰 영상을 보고 success / overfit-f1 / overfit-f2로 분류(부록 B.2). 사람 판정과의 일치율 88.7 %(표 5, π0.5, 주석 롤아웃 80개).

**식 1.** FDR(f1, f2) = (N_f1 − N_f2) / (N_f1 + N_f2 + ε). 양수면 f1이 지배. "neither"로 판정된 롤아웃은 N_f1 + N_f2에서 제외(부록 B.2).

### 2.2 FDH (§4.2, 부록 B.3)

**식 3.** C(f_i) = Σ_{j≠i} 1[FDR(f_i, f_j) > τ], τ = 5 %. 이긴 쌍만 +1, 지거나 비기면 0. Copeland 점수 내림차순이 FDH.

### 2.3 결과 (표 1, 2, 6, 7)

| 모델 | FDH (표 2) |
|---|---|
| π0.5 | color > object ≈ spatial > verb > size |
| π0 | color > object > spatial ≈ verb > size |
| OpenVLA-oft | color > object ≈ verb ≈ size > spatial |
| GR00T-N1.7 | color > object > spatial > verb > size |
| XVLA | color ≈ object > spatial > verb > size |
| Genie-Envisioner | color > object > spatial > verb > size |

- Finding 1: 모든 모델에서 color가 최상위. Finding 2: verb와 size가 최하위(OpenVLA-oft 제외). Finding 3: OpenVLA-oft를 빼면 순서가 color ≥ object ≥ spatial ≥ verb ≥ size로 대체로 일치.
- 저자의 해석(§4.2)은 "가능한 설명"으로, color는 눈에 띄는 저수준 시각 단서이고 size는 상대 기하에 의존한다는 것이다. 이 해석을 직접 가르는 실험(예: 색을 흐리게 한 대조)은 없다.
- 부록 B.3: τ = 10 %에서도 "일관"하다고 쓴다(표 7). 재계산 결과는 `03` V1.

## 3. 전략적 데이터 수집 (§5, 부록 C.1–C.2)

- 아이디어(Gao et al. [40]의 환경 인자 조합 일반화에서 착안): 훈련에서 한 인자는 한 값에 고정하고 다른 인자를 전부 훑는 "열"을 수집한다. 고정되는 쪽이 덜 접지된 인자. 여기에 대각선과 부대각선을 더한다. 인자가 셋 이상이면 인자당 열 하나를 추가하고 FDH에서 낮은 쪽부터 고른다(§5, §6).
- 중요한 설계: 설정마다 FDR을 다시 재지 않고 **전역 순서 하나**(color ≥ object ≥ spatial ≥ verb ≥ size)를 재사용한다(§6 "Selection of Under-grounded Factors"). Nf = 2이면 낮은 인자 하나, Nf = 3이면 둘, Nf > 3이면 셋.
- 기준선: **L**(각 인자를 한 값에 고정하고 나머지를 균일 표집, 격자에서 "L"자 모양), **Random**(전체 공간에서 무작위 조합), **Complete**(전수, 상한 참고용, 실로봇 Bun에서만 보고).
- 비용 단위: **factor change f** = 연속한 두 구성 사이의 Hamming 거리의 총합(§6, 부록 C.2). 전략마다 f가 다르면 Random 시연을 몇 개 더해 f를 맞춘다고 쓴다. 시연 예산 n은 총 시연 수.
- 비용 서술(부록 C.2): Complete는 Π k_i, V와 L은 O(Σ k_i), Nf = 2이면 V ≈ 3k, L ≈ 2k.

## 4. 실험 (§6, 부록 C)

### 4.1 시뮬레이션 (§6.1, 그림 4, 그림 10)

- 정책 π0, π0.5, GR00T-N1.7을 공개 체크포인트에서 "각 모델의 기본 학습 설정으로 수렴까지" 미세조정(부록 C.3). 시연은 ManiSkill 모션 플래너로 생성한 스크립트 시연(원격조작이 아님).
- 설정 12개(그림 4): 쌍별 8개(Color-Object, Verb-Object, Color-Verb, Verb-Size, Color-Spatial, Spatial-Size, Verb-Spatial, Spatial-Object), Verb-Color-Object 둘(π0.5, π0), All-Factor 둘(π0.5, GR00T). 쌍별은 n = 200과 f = 6/12/18, Verb-Color-Object는 f = 15/20/30, All-Factor는 f = 50과 n = 200/400/800.
- 평가: 체크포인트당 시드 3개, 시드당 롤아웃 200개, 점은 시드별, 선은 평균(그림 4 캡션).
- 본문 주장: V가 대다수 설정에서 L과 Random 이상. 큰 이득은 color-object +10점(최고 기준선 대비), color-spatial +7점, spatial-size(GR00T) +14점. 예외는 Verb-Size(π0.5)에서 V 51 % 대 Random 68 %(f = 18). All-Factor에서 V는 n = 400으로 기준선 n = 800을 넘는다고 쓴다(§6.1).
- 부록 C.4: size가 들어간 쌍에서는 "V의 이점이 사라진다"(L이 V와 같거나 낫다). size는 전략과 무관하게 가장 접지하기 어렵다고 해석.

### 4.2 실로봇 (§6.2, 표 3, 그림 7–9, 부록 C.5)

- UR5, 맞춤 그리퍼, RealSense D455 에이전트 뷰, FLIR 손목 카메라, GELLO 시연. 정책은 GR00T-N1.7 하나.
- 과제 세 개, 각 4×4 = 16개 지시 격자. **Bun**(color × spatial, 찜기 바구니에 찐빵 넣기), **Pizza**(spatial × object, 소스 올리기), **Cup**(verb × color, 컵에 대한 기술).
- f = 12 고정, n = 100과 200. 체크포인트당 롤아웃 **48개**(지시당 3개). 성공 조건은 "두 번의 시도 안에 과제 완료"(부록 C.5).

| 방법 | Bun(100) | Bun(200) | Pizza(100) | Cup(100) |
|---|---|---|---|---|
| L | 11/48 (22.9 %) | 18/48 (37.5 %) | 17/48 (35.4 %) | 18/48 (37.5 %) |
| Random | 18/48 (37.5 %) | 23/48 (47.9 %) | 24/48 (50.0 %) | 23/48 (47.9 %) |
| V | 26/48 (54.2 %) | 31/48 (64.6 %) | 30/48 (62.5 %) | 32/48 (66.7 %) |

- 본문 주장: V가 L보다 평균 +28.7점, Random보다 평균 +16.7점. Bun에서 V(n = 100) 54.2 %가 L과 Random(n = 200)을 넘으므로 "절반의 시연"으로 더 높은 성공. 그림 9: n = 100에서 V가 Complete를 크게 앞서고(부분 커버리지가 전체 커버리지보다 나을 수 있음), 그림 8: 칸별 성공 수.
- 그림 9의 Complete는 n = 200에서 약 88 %(그림 판독 값)로 V(64.6 %)보다 높다. 본문은 n = 100 비교만 서술한다.

## 5. 한계 (§7)와 추가로 보이는 점

저자가 쓴 한계: 통제된 탁상 설정, 더 복잡한 장면과 다단계 과제로의 일반화는 열린 문제, V는 최적 전략이 아니라 한 사례.

읽으면서 보이는 추가 점(근거는 `03`, `04`):

1. FDR은 인자 사이의 상대적 우세만 재고, "어느 정도 접지되었는가"의 절대 수준은 재지 않는다. 판정은 VLM 영상 판독이고 검증은 π0.5 한 모델의 80개이다(표 5).
2. V 이득은 같은 f에서 훈련 칸 수가 더 많은 지지집합과, 평가 격자에 훈련 칸이 섞이는 것과 함께 읽어야 한다(`04` D3, `03` V8).
3. 전역 순서를 재사용하는 설계는 모델별 FDH 차이(특히 OpenVLA-oft)를 수집 전략에 반영하지 않는다. 수집 실험은 FDH가 가장 일관된 π0.5, π0, GR00T에서만 했다.
4. 실로봇은 정책 하나, 학습 시드 하나(추정), 시행 48개이다. 같은 열 안 V 대 Random 차이는 개별로는 Fisher p = 0.10–0.30이다(`03` V4).

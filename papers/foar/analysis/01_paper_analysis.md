# 01. 논문 분석: FoAR (arXiv:2411.15753)

분석 대상은 **arXiv v2 (2025-05-02)**입니다. 원문은 `paper/paper.pdf`(9쪽, `pdftotext` 결과는 분석 중 임시 파일로만 썼습니다). 인용 표기는 "§"(절), "식"(수식 번호), "표 I–V", "Alg. 1", "App. A"입니다.

## 0. 메타와 버전

| 항목 | 내용 |
|---|---|
| 제목 | FoAR: Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation |
| 저자 | Zihao He*, Hongjie Fang*, Jingjing Chen, Hao-Shu Fang†, Cewu Lu† (SJTU; \* 동등 기여, † 교신) |
| arXiv | 2411.15753, cs.RO. v1 2024-11-24 (2,028 KB), v2 2025-05-02 (2,097 KB). 분석 대상은 v2 |
| 학회·저널 | IEEE Robotics and Automation Letters, vol. 10, no. 6, pp. 5625–5632 (2025년 6월). DOI 10.1109/LRA.2025.3560871 (Crossref 확인). arXiv 댓글 "Accepted to Robotics and Automation Letters" |
| 라이선스 | arXiv abs 페이지의 "view license" 링크는 `http://creativecommons.org/licenses/by-nc-sa/4.0/`를 가리킵니다 (CC BY-NC-SA 4.0). PDF 본문에서 ©, copyright, All rights reserved 문구는 찾지 못했습니다 |
| 프로젝트 | https://tonyfang.net/FoAR/ (영상, 코드 링크, BibTeX 포함. 프로젝트 BibTeX는 "arXiv preprint"로만 적음) |
| 공식 코드 | https://github.com/Alan-Heoooh/FoAR (분석은 `repo/` 정적 읽기, §4 참조) |

### 0.1 v1과 v2의 차이

v1과 v2는 방법(식, 알고리즘, 하이퍼파라미터)이 같고, 실험 표의 구성과 캡션이 바뀌었습니다. 주장의 뼈대는 같습니다.

| 항목 | v1 (2024-11-24) | v2 (2025-05-02) |
|---|---|---|
| 기준선 수 | 본문 "five baseline methods, including RISE and three ablation variants" (목록은 넷) | "six baseline methods": ACT [52], Diffusion Policy [10], RISE, 세 절제 변형 |
| 표 I의 Wiping (General) 열 | RISE, RISE (force-token), RISE (force-concat), FoAR (3D-cls)의 WG 칸이 "-"이고, FoAR만 0.850 | 여섯 방법 모두 WG 값을 채움 |
| 표 I 캡션 | "For the Wiping (General) task, we only select the best-performing ablation variant on the Wiping task, RISE (force-token)"이라고 쓰였으나 표에는 다른 방법도 있음 | 캡션 단순화 |
| 반응 제어 절제 | 표 III: Wiping (General)에서 "FoAR wo. Reactive Control" 0.650 (grasp 100, wipe 80) 대 "w. Reactive Control" 0.850 (100, 100) | 표 III: Wiping에서 F/T 주파수(2, 10, 100 Hz)와 예측기·반응 제어 유무의 다섯 행. 완전 모델 0.875 (100, 100) |
| 부록 | 표 V (F/T 인코더: MLP 대 Transformer) 있음 | 같음 (값도 같음. 표 I와 불일치는 `03_analysis.md` K7) |
| 그림 | Fig. 2(구조도) 없음 | Fig. 2(구조도)와 Fig. 1 설명 보강 |

v1 표 III의 "0.650 / 80%" 행은 v2에서 "100Hz, predictor only" 행으로 옮겨졌고, 완전 모델 0.850(WG)은 v2에서 0.875(Wiping)로 바뀌었습니다. 같은 셀이 과제 이름이 다른 두 실험에 걸쳐 있습니다(`03_analysis.md` K8).

## 1. 문제와 주장

접촉이 많은 조작(닦기, 껍질 벗기기, 썰기)은 접촉 상태를 시각만으로 구별하기 어렵습니다. 논문은 Fig. 1을 근거로 접촉 전후의 RGB나 점군 차이가 작다고 봅니다(§III-A). 기존 힘·토크 융합은 전 구간에 힘 데이터를 섞어, 접촉이 없는 구간에서 센서 잡음이 정책을 흔든다고 주장합니다(§I, 두 번째 문단).

핵심 주장은 세 가지입니다(§I, §V).

1. **미래 접촉 확률 ϕ로 힘 특징의 비중을 정하는 융합**이 접촉 구간의 정밀도를 높이고, 비접촉 구간의 성능을 해치지 않습니다(§III-B, 식 (융합)).
2. **반응 제어**(ϕ와 현재 힘으로 행동 청크를 보정)가 단순 위치 제어만으로 접촉 과제를 잘 수행하게 합니다(§III-C, Alg. 1).
3. 외란(재작성, 이동)에서도 성능이 유지됩니다(§IV-E).

## 2. 방법

### 2.1 구성

입력은 점군 p_t(RGB-D를 점군으로 바꾼 것, 복셀 5 mm, App. A), 최근 To = 200개(100 Hz, 약 2 s)의 힘·토크 f_{t−To:t}, 현재 RGB I_t입니다. 출력은 행동 청크 a_{t:t+Ta}(Ta = 20, 10 Hz)와 미래 접촉 확률 ϕ(t) ∈ [0, 1]입니다(§III-A, Fig. 2).

| 구성 요소 | 내용 | 출처 |
|---|---|---|
| 점군 인코더 | RISE의 희소 3D 인코더 (MinkowskiEngine, 얕은 ResNet), 512차원 점 토큰, 4 인코딩 + 1 디코딩 Transformer (d_model 512, d_ff 2048) | §III-B, App. A |
| 힘 인코더 | 3층 MLP (64, 128, 512)로 토큰화 후 정현파 위치 부호화의 Transformer, 읽기 토큰 h^f_t | §III-B, App. A |
| 미래 접촉 예측기 | ResNet18(RGB) + 두 층 MLP(힘, 128→512 라고 적음), 이어 붙여 선형층 → 시그모이드 ϕ | §III-B, App. A |
| 융합 | h_t = [h^s_t ; ϕ(t)·h^f_t + (1 − ϕ(t))·h*], h*는 학습 가능한 임베딩 | §III-B 식 (융합) |
| 행동 헤드 | CNN 기반 확산 헤드 [10], 학습 100 단계, 추론 DDIM 20 단계 | App. A |

### 2.2 손실과 라벨

L = L_action + α L_predictor, α = 0.1 (§III-B, §IV-A, App. A). L_action은 확산 노이즈 예측의 L2, L_predictor는 BCE입니다. 접촉 라벨은 시연의 힘·토크가 δ_demo를 넘는지로 자동 생성하며, 시간 창은 [t−2 s, t+2 s]입니다(App. A). δ_demo의 값은 논문에 없습니다("may differ across tasks").

### 2.3 반응 제어 (Alg. 1)

- ϕ < δϕ (= 0.9)이면 비접촉 버퍼에 행동 청크를 넣습니다.
- ϕ ≥ δϕ이면 접촉 버퍼를 쓰고, 현재 힘 |f| < δf (8 N)이고 토크 < δt (5 N·m)일 때만 행동을 보정합니다. 보정은 최근 Tf = 5 단계 평균 위치의 방향으로 ε = 0.006 m만큼 민 것입니다(Alg. 1 줄 12–14).
- 두 버퍼는 따로 시간 앙상블 [52]합니다(§III-C).
- 제어 명령은 end-effector 위치 제어입니다. 컴플라이언스, 어드미턴스, 하이브리드 힘·위치 제어는 쓰지 않는다고 명시합니다(App. A, Deployment). 한계 절(§V)에서도 같은 점을 적습니다.

## 3. 실험 설정

| 항목 | 내용 | 출처 |
|---|---|---|
| 로봇 | Flexiv Rizon, Dahuan AG-95 그리퍼, OptoForce F/T 센서(플랜지와 그리퍼 사이), RealSense D435 | §IV-A |
| 작업 영역 | 45 × 60 × 40 cm | §IV-A |
| 작업 | Wiping (화이트보드 고정), Wiping (General) (임의 방향), Peeling (오이), Chopping (피망) | §IV-A, Fig. 3 |
| 시연 | 닦기·껍질 50회, 썰기 40회, 햅틱 원격 조작 | §IV-A |
| 평가 | 닦기·껍질 각 20회 (모든 방법), 썰기 10회 (FoAR와 RISE만). 외란 실험의 시행 수는 적혀 있지 않음 | §IV-A, 표 IV |
| 기준선 | ACT, Diffusion Policy, RISE, RISE (force-token) [힘 토큰을 RISE 트랜스포머에 추가], RISE (force-concat), FoAR (3D-cls) | §IV-A |
| 학습 | 2 × A100, 배치 240, lr 3e-4, 워밍업 2000, 코사인 스케줄 | App. A |
| 측정 지표 | ASR(행동 성공률), 닦기 점수 (완전 1, 부분 0.5, 없음 0), 껍질 점수 (벗긴 비율을 시연 평균 0.778로 나눔), 썰기 (조각 수, 정규화 길이의 평균과 SD) | §IV-A |

## 4. 결과 요약 (논문이 보고한 값)

| 표 | 내용 | 주요 값 (FoAR 대 최강 기준선) |
|---|---|---|
| 표 I | 닦기, 닦기 (일반), 껍질 (점수와 ASR) | 0.875 (대 RISE (force-token) 0.575), 0.850 (대 RISE (force-concat) 0.675), 0.756 (대 RISE (force-concat) 0.524) |
| 표 II | 썰기 (10회) | 조각 수 3.9 ± 0.9 대 RISE 1.8 ± 0.6. 정규화 길이 평균 0.353 대 0.727, SD 0.094 대 0.411. 배치 성공 7/10 대 3/10 |
| 표 III | 닦기의 설계 선택 (F/T 주파수, 예측기, 반응 제어) | 완전 모델 0.875, 예측기만 0.650, 반응만 0.650, 2 Hz 0.625, 10 Hz 0.800 |
| 표 IV | 닦기 (일반)의 외란 (원래, 재작성, 이동, 둘 다) | FoAR 0.800–0.850 (ASR 100%), RISE (force-token)은 원래 0.600, 둘 다 0.600 (ASR 100/100) |
| 표 V (App. B) | 껍질, F/T 인코더 변형 | MLP 0.426, Transformer 0.588 (표 I의 0.756과 다름, `03_analysis.md` K7) |

본문 해석은 표 I와 II를 근거로 "기준선보다 낫다"입니다. 표 III와 IV에 대한 해석은 아래 §6과 `03_analysis.md`에서 검사합니다.

## 5. 한계 (논문 자체)

논문 §V가 적은 한계는 셋입니다. 정적 힘·토크 임계값이 복잡한 환경에서 깨질 수 있습니다. 위치 제어는 적응성이 낮습니다. 양팔, 휴머노이드, 손가락 로봇으로 확장하는 일은 미래 과제입니다.

논문이 적지 않은 한계는 다음과 같습니다.

- **시행 수가 작습니다.** 과제당 20회, 썰기 10회입니다. 본문의 "significantly"는 검정 없이 쓰였습니다(`03_analysis.md` §1).
- **기준선의 튜닝과 공정성 정보가 없습니다.** RISE 기준선의 학습 설정과 ACT, DP의 입력 형식이 적혀 있지 않습니다.
- **접촉 라벨의 δ_demo와 시간 창이 과제마다 다릅니다.** 라벨 품질이 예측기 성능을 직접 정합니다(§2.2). 값이 적혀 있지 않아 재현에 필요한 정보가 빠져 있습니다.
- **위치 제어가 실제로 "단순"한지는 구현 세부에 달려 있습니다.** 공식 코드의 로봇 인터페이스는 Flexiv의 임피던스 계열 모드를 씁니다(`04_code_verification.md` D10).
- **데이터, 체크포인트, 기준선 코드, 썰기와 외란 실험 코드가 공개되어 있지 않습니다**(`04_code_verification.md` §1).

## 6. 주장과 근거의 간격 (요약)

| 주장 | 근거 | 간격 |
|---|---|---|
| "모든 기준선을 유의하게 능가" (§IV-B, 표 I) | 점수 평균만 있고 개별 시행 점수는 없음 | ACT(세 과제), FoAR (3D-cls)(두 과제), Diffusion Policy(닦기 두 과제)에는 Holm 보정 후에도 유의하지만, RISE, RISE (force-token), RISE (force-concat) 대비 점수 차이는 세 과제 모두 Holm 보정 후 유의하지 않음 (최소 p_Holm = 0.064, `03_analysis.md` §1.2) |
| "RISE (force-token)은 복합 외란에서 성능이 떨어진다" (§IV-E) | 표 IV | 표 IV에서 같은 방법의 "둘 다" 점수는 원래와 같고(0.600), 성공률은 100/100으로 오히려 높음 (`03_analysis.md` K6) |
| "반응 제어가 필수다" (§IV-D, 표 III) | 완전 모델 0.875 대 예측기만 0.650 | 차이 0.225, 검정 p ≈ 0.10 (`03_analysis.md` §1.4) |
| "접촉 검출 변형은 접촉이 늦거나 실패한다" (§IV-D) | 본문만 설명 | 표 III에는 검출 변형 행이 없음 (`03_analysis.md` K5) |
| "썰기에서 두 배 이상 조각" (§IV-C) | 표 II | 3.9/1.8 = 2.17. 정확히 맞음. 다만 n = 10 |

## 7. 해석 메모

- 방법의 핵심은 "힘 특징을 ϕ로 가중하고, 비접촉에서는 학습된 중립 임베딩 h*로 대체"하는 한 줄입니다(§III-B 식 (융합)). 실제 구현(§4)에서도 그대로 확인됩니다.
- 접촉 예측기가 ϕ를 틀리면 두 경로가 모두 틀어집니다. 즉 융합 가중과 반응 제어가 같은 신호에 기댑니다. 논문은 이 공동 실패를 실험하지 않았습니다.
- 임계값 0.9는 낮은 재현율과 높은 정밀도를 노린 선택으로 보입니다. 논문은 δϕ 민감도를 보이지 않았습니다.

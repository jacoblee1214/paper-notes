# 01. 논문 분석: Evaluating Real-World Robot Manipulation Policies in Simulation

## 0. 기본 정보와 버전

- **제목:** *Evaluating Real-World Robot Manipulation Policies in Simulation* (SimplerEnv / SIMPLER)
- **저자:** Xuanlin Li\*, Kyle Hsu\*, Jiayuan Gu\*, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, Ted Xiao (UC San Diego, Stanford, UC Berkeley, Google DeepMind)
- **arXiv:** 2405.05941, cs.RO (cs.CV, cs.LG 교차). **v1만 존재** (2024-05-09, 8,759 KB). abs 페이지 "Submission history"와 API 모두 v1만 보고합니다(2026-11-22 확인). 분석 대상은 v1입니다.
- **학회 판본:** *Proceedings of The 8th Conference on Robot Learning* (CoRL 2024, 뮌헨, 2024-11-06~09), PMLR 270:3705–3728, 2025-01-12 공개. PMLR 페이지(`proceedings.mlr.press/v270/li25c.html`)의 BibTeX와 목록에서 확인했습니다. 이 판본은 **24쪽**으로 arXiv v1(**22쪽**)보다 깁니다(§0.1 참조).
- **라이선스:** arXiv abs 페이지의 "view license" 링크는 `creativecommons.org/licenses/by/4.0/`(CC BY 4.0)입니다(2026-11-22 확인). 원문 PDF 본문에서 ©, copyright, All rights reserved, 회사 고지는 찾지 못했습니다(`paper/paper.txt` 검색). 그림 게시 판단은 `04_code_verification.md` §1에 있습니다.
- **공식 코드:** `github.com/simpler-env/SimplerEnv` (MIT, 별 1,188, 마지막 push 2026-09-30, 분석 시점 HEAD `eddb5690`). 서브모듈 `ManiSkill2_real2sim` (Apache-2.0, 커밋 `ef7a4d4`, 2024-10-18). 자세한 내용은 `04_code_verification.md`.
- **프로젝트 페이지:** `simpler-env.github.io` (논문, 영상, 코드, Colab 링크). 페이지에는 학회 표기가 없습니다.

### 0.1 학회 판본과 arXiv v1의 차이 (확인한 범위)

PMLR 판본(24쪽)을 내려받아 표 값과 키워드를 대조했습니다(임시 폴더에만 두었고 이 저장소에는 넣지 않았습니다).

- **같은 것:** 표 IV의 실로봇·시뮬 값(예: Pick Coke Can 평균 RT-1 Converged 0.853, Octo-Base 실로봇 0.293), 표 V의 Octo 과제 값(에그플랜트 0.400, 0.233, 0.433), 절제 실험(표 III, 표 VI–VIII)의 MMRV·r 값 대부분, 부록 B의 과제 설명(24 시도 문구 포함).
- **다른 것:** PMLR 판본 그림 4 상단에 **"Google Robot MMRV = 0.049, r = 0.929"** 가 있습니다. arXiv v1에는 같은 값이 없고 과제별 MMRV(그림 6)만 있습니다. 네 과제 평균으로 다시 계산하면 MMRV 0.026, r 0.973이라 이 값을 재현하지 못했습니다(집계 방식이 다를 수 있음, 미해결). 그림 번호와 부록 번호도 바뀌었습니다(부록 표 "Table 6"로 표기).
- **이 분석의 한계:** PMLR 판본 전체를 표 단위로 대조하지는 않았습니다. 이 분석의 수치 재계산은 arXiv v1 기준입니다.

## 1. 한 줄 요약

시뮬레이터에서 **정밀한 디지털 트윈 없이도** 실로봇 정책의 상대 순위를 맞추는 평가 환경을 만들 수 있다고 주장합니다. 제어 격차는 PD 파라미터의 오프라인 시스템 식별(SysID)로, 시각 격차는 실사 배경 합성(green screening)과 텍스처 매칭(Visual Matching)으로 줄이고, 텍스처를 무작위화하는 Variant Aggregation도 비교합니다. Google Robot(RT 계열)과 WidowX + Bridge(Octo, RT-1-X) 두 설정에서 실로봇과 시뮬의 순위 상관(MMRV, Pearson r)을 보고합니다.

## 2. 구조와 방법 (§ 번호는 PMLR 판본이 아니라 arXiv v1 기준)

| 절 | 내용 | 핵심 |
|---|---|---|
| I | 서론 | 실로봇 평가는 비싸고 재현이 어렵다. 목표는 1:1 재현이 아니라 상대 순위의 상관이다 |
| II | 관련 연구 | 내비게이션 sim2real 평가(Kadian 외 [32], Deitke 외 [14]), 시뮬 벤치마크, 도메인 무작위화·적응 |
| III | 문제 정의와 지표 | 식 (1)–(2) MMRV, Pearson r. Pearson의 두 한계(선형성, 범위 민감성)를 지적 |
| IV-A | 제어 격차 | 6D 말단 궤적 손실 식 (3)–(5), 간단한 어닐링으로 PD(강성·감쇠) 최적화 3라운드 |
| IV-B | 시각 격차 | 그린스크린 식 (배경 합성), 텍스처 매칭(객체·로봇), Variant Aggregation(무작위 변형) |
| V | SIMPLER 구성 | Google Robot 과제 4개군(픽, 가까이 옮기기, 서랍, 서랍 안에 사과), WidowX 과제 4개 |
| VI-A | 실험 설정 | RT-1 (Begin/15%/Converged), RT-1-X, RT-2-X, Octo-Base/Small. 시뮬은 Octo 3시드, Google 로봇 팔 색 4종 평균 |
| VI-B | 상관 결과 | 그림 6–7, 표 I, IV, V |
| VI-C | 분포 이동 | 배경·조명·방해물·테이블·카메라 자세 5축. RT-1 두 체크포인트. 팔 텍스처 예측 |
| VI-D | 절제 | SysID(표 II), 시각 매칭 조합(표 III), 물리 파라미터 민감도(표 X), Isaac Sim(그림 10, 표 XI) |
| VII | 결론, 한계 | 강체 조작에 한정 |

### 2.1 지표 (식 (1)–(2))

- **순위 위반:** RankViolation(i, j) = |R_i − R_j| · 1[(R_{S,i} < R_{S,j}) ≠ (R_i < R_j)]
- **MMRV:** 정책별 최대 위반을 평균. 위반 크기로 가중하므로, 실로봇 차이가 작은 쌍의 순위 뒤집힘은 작게 셉니다.
- 저자 주장: Pearson은 선형 적합만 보고 범위에 민감하므로 MMRV를 함께 씁니다(§III-B).

### 2.2 제어 격차 (§IV-A)

- 식 (3)–(5): 위치 손실 L_transl과 회전 손실 L_rot(arcsin 형태)를 더한 L_sysid. 실로봇 데모를 개루프로 재생하고 말단 궤적을 비교합니다.
- 그림 4: SysID 전 Google Robot이 코카콜라 캔을 놓치는 장면과 SysID 후 성공하는 장면을 보여 줍니다(정성적).
- 공식 코드에 같은 손실과 어닐링 도구(`tools/sysid/sysid.py`)가 있고, Google Robot 기본 PD 값이 SysID 결과로 들어 있습니다(`04` §2 D3).

### 2.3 시각 격차 (§IV-B)

- 그린스크린: I′ = M ⊙ I_sim + (1 − M) ⊙ I_real. 실로봇 첫 프레임에서 로봇과 물체를 지운 배경을 쓰고, 시뮬의 전경 마스크를 덧댑니다.
- 텍스처 매칭: 객체 텍스처를 실사 이미지에서 투영하고, 로봇 메쉬 색은 실사 영상에서 여러 색을 골라 평균합니다.
- Variant Aggregation: 배경·조명·방해물·테이블 질감을 무작위화한 변형 여럿을 평균합니다.

### 2.4 평가 프로토콜 (부록 B)

- Google Pick Coke Can: 3 방향 × 25 위치 = 75. Move Near: 60. 서랍 열기/닫기: 27+27. 서랍 안에 사과: 27.
- WidowX: 과제당 24 (부록 B 문구). **단 표 V의 에그플랜트 실로봇 값은 1/30 격자에 맞습니다**(`03` §1.3).
- 시뮬 시도 수: Google Pick은 25 × 3 × 4 색 = 300. Bridge는 24 에피소드 × Octo 시드 3 = 72.

## 3. 주장과 근거

| 주장 | 근거 | 상태 |
|---|---|---|
| SIMPLER(Visual Matching)가 실로봇 순위와 강하게 맞는다 (MMRV·r) | 표 IV, 그림 6. Google 평균 MMRV 0.056, r 0.92 (표 I) | 재계산으로 대부분 재현(`03` §1). 다만 6 정책 표본이라 r의 신뢰구간이 넓다 (`03` §2) |
| 검증 MSE는 순위 지표로 나쁘다 | 표 I: MMRV 0.375, r 0.31 | 값은 맞다. 비교 대상이 학습 중 MSE 하나라 일반화에는 한계 |
| 시각 매칭이 Variant Aggregation보다 낫다 | 표 I: 0.056 대 0.143 | 맞다. 다만 그린스크린과 텍스처를 **둘 다** 쓸 때만 효과가 있다(표 III) |
| SysID가 제어 손실과 MMRV를 줄인다 | 표 II: 제어 손실 0.131 대 0.267 / 0.432, MMRV 0.031 대 0.070 / 0.100 | 수치는 표 안에서 맞다. 두 설정이 공식 코드의 WorseControl 1·2와 같은지는 확인하지 못했다(`04` D3, 추정) |
| 분포 이동에 대한 민감도를 재현한다 | 표 VI, 그림 8: 카메라 자세 −0.47 대 −0.46 | 주된 방향은 맞다. 세부 값에는 불일치가 있다(`03` §3) |
| 물리 파라미터 오차에 강건하다 | 표 X: MMRV 0.055 대부분, 0.105 (마찰 0.15) | 맞다. 다만 마찰 0.15에서 MMRV가 뚜렷이 높다 |
| Isaac Sim에서도 재현된다 | 표 XI, 그림 10: MMRV 0.058, r 0.919 | 논문 내부 표로만 확인. Isaac 구현은 공식 저장소에 없다 (`04` §2) |
| 약 1500 평가 에피소드 | 그림 1 캡션 | 부록 B로 세면 실로봇 Google 1,296 + WidowX 306 = 1,602이다 (`03` §4). 근사 표현 |
| 7배 속도 향상 (4090 GPU, 3.5k 스텝/s) | §V | 측정 조건(시뮬 주파수, 해상도) 외 재측정은 안 했다. 이 맥에서 같은 측정은 불가 |

## 4. 제한 (논문 스스로 밝힌 것과 빠진 것)

- **강체 조작에 한정.** 천, 액체, 유연 물체는 없습니다(§VII 제한).
- **과제와 로봇 수가 적다.** 두 설정, 정책 6개(Google), 3개(WidowX). MMRV와 r은 정책 수에 크게 좌우됩니다(`03` §2).
- **실로봇 시도 수가 적다.** 대부분 24–75회입니다. 정책 간 실로봇 차이 중 많은 것이 잡음 수준입니다(`03` §1).
- **객체 질량과 마찰은 근사.** GPT-4나 검색으로 밀도를 정하고 마찰은 물성 추정입니다(§V, 부록 C). 민감도 실험(표 X)은 있지만 범위가 좁습니다.
- **수동 조정이 많다.** 로봇·카메라 자세와 텍스처 매칭은 사람이 맞춥니다(§V). 과제마다 노력이 다르고 보고되지 않습니다.
- **Octo 시드 평균** (Bridge 스크립트는 0, 2, 4 세 시드를 쓰지만 Google 스크립트에는 반복이 보이지 않습니다. `04` D6).
- **그린스크린 한계:** 로봇 팔이 움직이며 가리는 배경은 첫 프레임 기준이라 정확하지 않을 수 있습니다(§IV-B 기술).

## 5. 이 분석이 본 것 요약 (세부는 `03`, `04`)

1. 표 IV의 MMRV와 Pearson 값 36개(Visual Matching 18개, Variant Aggregation 18개)는 재계산과 대부분 맞습니다(±0.001). 예외는 Visual Matching Close 열의 r 하나(재계산 0.771, 인쇄 0.768)입니다. 실로봇 값은 모두 k/n 격자에 맞고, 시뮬 값은 한 셀(RT-1-X Close 0.891)만 격자에서 벗어납니다.
2. 실로봇 잡음을 넣으면 Pick Coke Can 평균 MMRV 0.031의 95% 구간은 0.000–0.082입니다. 상위 세 정책의 실로봇 차이는 p = 0.30–0.45입니다.
3. 에그플랜트 실로봇 값과 Table V 상관 계수의 표기가 서로 맞지 않습니다. 공식 저장소의 REAL_PERF 값과도 다릅니다.
4. 표 VI의 MMRV 일부는 인쇄된 입력으로 재계산하면 다른 값이 나옵니다.
5. 공식 저장소는 MIT이고 평가 지표 구현이 논문 식과 같습니다. 다만 값은 `metrics.py`에 하드코딩되어 원자료(실로봇 로그)는 없습니다.

## 6. 참고 문헌 표기 (논문 내부 번호)

- 정책: RT-1 [6], RT-2 [5], RT-1-X·Open X-Embodiment [11], Octo [50], RT-2-X는 [5] 계열.
- 데이터: Bridge V2 [66], Objaverse [15], CoACD [67].
- 시뮬레이터: SAPIEN [68], Isaac Sim [49], ManiSkill2 (저장소 기준).
- 지표 관련: Pearson [51], Spearman [61], 시뮬레이티드 어닐링 [53].
- 분포 이동 기준: Xie 외 [69] (논문이 실로봇 테이블 질감 실험의 근거로 인용).

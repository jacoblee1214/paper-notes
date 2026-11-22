# 04. 공식 코드 대조와 구현 가능성

- **대상:** `github.com/simpler-env/SimplerEnv` 커밋 `eddb5690749a1e34267807c0f7b5eede323918d7` (2026-09-30, "pin transformers<5"). 서브모듈 `ManiSkill2_real2sim` 커밋 `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682` (2024-10-18).
- **방법:** 얕은 복제(`repo/`)를 **정적으로만** 읽었습니다. 설치, 실행, import, 체크포인트·데이터 다운로드는 하지 않았습니다. 복제본은 입력 데이터로 취급했습니다.
- **서브모듈 범위:** 희소 체크아웃으로 `mani_skill2_real2sim/` 코드와 LICENSE, README만 받았습니다(49 MB, 자산 디렉터리 `data/` 제외).

## 1. 라이선스와 공개 범위

| 대상 | 라이선스 (근거) | 공개 시 판단 |
|---|---|---|
| 논문 (arXiv v1) | CC BY 4.0 (abs 페이지 "view license" → `creativecommons.org/licenses/by/4.0/`, 2026-11-22) | 글과 재계산은 인용과 함께 가능 |
| 논문 PDF 본문 | ©, copyright, All rights reserved, 회사 고지 없음 (`paper/paper.txt` 검색 결과 없음). 첫 쪽에 CC 표시도 없음 | 조건 충족 |
| 그림 (논문 전체) | 그림에는 실로봇 사진, 실제 배경 이미지(그린스크린 입력), 데이터셋 프레임이 많습니다 (그림 4, 5, 9, 1의 실사 장면). 패널별 검토는 하지 않았습니다 | **게시 불가.** 공개본과 블로그에는 논문 그림을 넣지 않고 우리 그래프만 씁니다 |
| 공식 코드 SimplerEnv | MIT (LICENSE 파일, GitHub API 모두 확인) | 저장소 링크 가능. 코드는 복사하지 않음 |
| 서브모듈 ManiSkill2_real2sim | Apache-2.0 (LICENSE 파일) | 링크 가능 |
| 프로젝트 페이지 | 하단 링크 `creativecommons.org/licenses/by-sa/4.0/` (템플릿 표기로 보임). 페이지 콘텐츠 라이선스로 확정하지 않음 | 링크만 |
| Colab, 영상 데이터셋 | 미확인 | 받거나 재배포하지 않음 |

**그림 게시 판정: 불가.** 논문 본문은 CC BY 4.0이지만 주요 그림 대부분이 실로봇 사진이나 실제 배경을 포함하고, 패널 단위 검토를 하지 않았습니다. 공개물은 우리 그래프만 사용합니다.

## 2. 논문 ↔ 코드 대조 (D 항목)

| ID | 논문 | 공식 코드 | 판정 |
|---|---|---|---|
| D1 | Algorithm 1: Google 로봇 시뮬 주파수 H_sim = 501 (구현 값), 제어 주파수 3 Hz | `simpler_env/evaluation/argparse.py` 기본 `--sim-freq 513`. 스크립트 대부분 513, **Octo Google 스크립트 하나만 501** (`scripts/octo_pick_coke_can_visual_matching.sh`) | 불일치. 같은 과제에서 정책마다 시뮬 주파수가 다릅니다(RT-1 513, Octo 501). 513은 3의 배수라 조건은 통과. 결과 영향은 측정하지 않음 |
| D2 | Algorithm 2: Bridge 시뮬 500 Hz, 제어 5 Hz | 스크립트 `--control-freq 5 --sim-freq 500` | 일치 |
| D3 | §IV-A: PD(강성·감쇠) SysID, 시뮬레이티드 어닐링 **3라운드**, 손실 식 (3)–(5) | `tools/sysid/sysid.py`는 어닐링 **한 번**(`sa.minimize`, step_max 2000)을 호출하고 라운드 축소 로직이 보이지 않습니다. Google 기본 강성·감쇠(`ManiSkill2_real2sim/.../google_robot/defaults.py`: 1700.0, 1737.047…, 979.975…)는 `sysid.py`의 주석 처리된 초기값과 **같은 숫자**입니다. 즉 공개 기본값은 SysID 결과입니다. 제어기 한계 1.5 / 2.0 / 50 (팔), 1.0 / 7.0 / 50 (그리퍼)는 Algorithm 1과 **일치** | 수치 설정은 일치. 3라운드 절차는 공개 도구에 없음 (미확인). 손실은 평균 번역 오차와 회전 오차의 합으로 보이나, 식 (4)의 정규화(√2 · arcsin 형태)는 확인하지 않음 |
| D4 | Algorithm 1 제어기 (Ruckig 시간 최적 궤적) | `agents/utils.py`의 `parameterize_path`가 Ruckig를 씀. `pd_ee_pose.py` 기본값에 jerk 50 | 일치 |
| D5 | Google 로봇 팔 색 4종 평균 | 스크립트 `urdf_version` 4개 (None, recolor 1, recolor 2, cabinet recolor 1) | 일치 |
| D6 | Octo 3시드 평균 | Bridge 스크립트 `--octo-init-rng 0 2 4` (세 시드). **Google Octo 스크립트에는 시드 반복이 없음**. Octo 추론 코드는 `seed`를 서버와 맞추는 루프를 가짐 | Bridge 일치. Google 쪽은 확인 못 함 |
| D7 | 부록 B: Bridge 과제 24 시도 | `--obj-episode-range 0 24` → `range(0, 24)` = 24 에피소드 (`maniskill2_evaluator.py`) | 일치. 그러나 **공식 `metrics.py`의 에그플랜트 실로봇 값(Octo-Base 0.250, Octo-Small 0.400)이 표 V(0.233, 0.433)와 다름** (`03` V2). 표 V의 에그플랜트는 1/30 격자 |
| D8 | 식 (1): 순위 위반에서 `<` 비교 | `mean_maximum_rank_violation`은 `>` 비교 | 구현 차이. 동률 때만 다름. 표 IV에서 동률은 2쌍, 두 규칙의 MMRV 값은 모든 열에서 같음 (재계산). 영향 없음 |
| D9 | 식 Pearson r | `pearson_correlation`은 표준 공식에 1e-8 보정 | 일치 |
| D10 | 표 IV, V의 실로봇·시뮬 원자료 | `metrics.py`에 **값을 하드코딩**. 원자료(에피소드 로그, 실로봇 영상 목록)는 저장소에 없음 | 재현 한계. 평가 로그가 없어 시도 단위 검정 불가 (`03` §5) |
| D11 | Isaac Sim 구현 (§VI-D, 표 XI) | 저장소 본체와 서브모듈 어디에도 Isaac 코드가 없음 (문자열 검색 0건). README는 GPU 병렬 Bridge를 `maniskill3` 브랜치로 안내 | Isaac 재현은 공개되지 않음 |
| D12 | 인용 정보 | README의 BibTeX는 `journal = arXiv preprint`만 적고 CoRL 2024를 적지 않음 (설명 줄에는 "CoRL 2024") | 인용 정보 불일치. 학회 판본은 PMLR 270 |
| D13 | 환경 버전 | 서브모듈은 2024-10-18 커밋, 논문은 2024-05. 저장소 README는 `numpy<2`, `ManiSkill2_real2sim` 설치를 요구 | 논문 결과를 똑같이 재현하려면 커밋 고정이 필요 |
| D14 | 로봇 설정 이름 | `GoogleRobotStaticBaseWorseControl1/2/3`, `TwiceFingerFriction`, `ManualTunedIntrinsic`이 있음 | 표 II의 Setting 1·2, 표 X의 마찰 설정과 대응 가능성이 있으나 **매핑은 확인 못 함** (추정) |

## 3. 재실행에 필요한 것 (설치하지 않고 정리만 함)

- **시뮬레이터:** SAPIEN 기반 ManiSkill2 real2sim. 저장소 README는 **NVIDIA GPU와 CUDA 11.8–12.x**를 요구합니다("full installation" 기준). 최소 설치도 Vulkan 렌더링을 씁니다.
- **정책:** RT-1, RT-1-X, RT-2-X는 TensorFlow·JAX 체크포인트, Octo는 JAX. README는 `tensorflow[and-cuda]==2.15.1`, `jax[cuda11_pip]==0.4.20`을 씁니다.
- **자산:** 서브모듈의 `data/`(텍스처, 실사 인페인팅 이미지, URDF 메쉬)와 `ManiSkill2_real2sim` 설치가 필요합니다. 크기는 서브모듈 전체 417 MB(희소 체크아웃 전)로 확인했습니다.
- **체크포인트:** 이 분석에서 받지 않았습니다(요청 제약). 크기는 저장소에 없고 README 링크로만 확인.

## 4. 구현 가능성 판정

**판정: 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서는 공식 SimplerEnv 평가를 재현할 수 없습니다. 표 재계산과 통계 검사는 완료했습니다.**

| 항목 | 맥에서 | 근거 |
|---|---|---|
| 표 재계산 (MMRV, r, 검정, 격자 검사) | **가능, 완료** | `analysis/recompute.py` |
| 그래프 (재계산 기반) | **가능, 완료** | `analysis/plot_figures.py` |
| 공식 지표 코드를 CPU에서 돌리기 (`metrics.py`) | **가능** (numpy만). 실행은 하지 않음 | 정적 확인으로 충분 |
| SAPIEN·ManiSkill2 환경 생성과 랜덤 행동 롤아웃 | **조건부/불가**. README는 NVIDIA GPU를 명시. Vulkan과 SAPIEN의 macOS 설치는 확인하지 않음 | README 69–70행 |
| RT-1 / Octo 추론 + 시뮬 평가 (표 IV, V 재현) | **불가**. CUDA 추론 전제, 체크포인트 다운로드 필요 | README 277, 288행 |
| 실로봇 평가 재현 | **불가**. 실로봇 필요 | — |
| Isaac Sim 재현 (표 XI) | **불가**. 코드 미공개, NVIDIA GPU 전제 | D11 |
| SysID 재실행 (표 II) | **불가~조건부**. 환경과 궤적 데이터(서브모듈 data 필요) 필요, CPU 병렬 18프로세스 전제 | `sysid.py` 143행 이하 |

**노력 추정 (측정 전):** 맥에서 환경을 띄우는 것만 해도 설치 검증에 반나절에서 하루가 들 것으로 봅니다. 정책 추론까지 하려면 CUDA 머신이 필요해 **맥 단독으로는 완료되지 않습니다.** 리눅스 GPU 머신에서 Google Pick Coke Can 한 조건(정책 1개, 25 × 3 위치)을 돌리는 데는 수 시간 정도로 추정합니다(미측정).

## 5. 확인하지 못한 것

- D3의 3라운드 어닐링이 공개 코드 밖(다른 스크립트나 서브모듈 외부)에 있는지 확인하지 못했습니다.
- D6의 Google Octo 시드 처리, D14의 WorseControl과 표 II의 대응을 확인하지 못했습니다.
- 공식 `metrics.py`의 에그플랜트 Octo 값(0.250, 0.400)이 어디서 왔는지 확인하지 못했습니다. 저장소 이력은 얕은 복제라 볼 수 없습니다.
- 표 X의 마찰·질량 실험을 재현할 수 있는 스크립트는 저장소에서 찾지 못했습니다.

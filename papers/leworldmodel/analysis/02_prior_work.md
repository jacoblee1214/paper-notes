# 02. 선행연구와 관련 자료

## 1. 계보

LeWM은 두 줄기의 교차점에 있습니다. 하나는 **JEPA 표현 학습**(I-JEPA, V-JEPA, 붕괴 방지 기법)이고, 다른 하나는 **잠재 공간 세계모델과 계획**(PlaNet·Dreamer 계열, DINO-WM, PLDM, TD-MPC2)입니다. 논문 §2의 분류를 따르면 다음과 같습니다.

| 줄기 | 대표 | 붕괴 방지 방식 | LeWM과의 차이 |
|---|---|---|---|
| 표현 JEPA | I-JEPA [12], V-JEPA 1·2 [13, 14] | EMA + 정지 그래디언트 | LeWM은 둘 다 쓰지 않음 (논문 §2) |
| 사전학습 인코더 세계모델 | DINO-WM [18] (DINOv2 동결) | 동결 인코더 | LeWM은 픽셀에서 끝까지 학습. DINO-WM은 proprio를 추가할 수 있음 |
| end-to-end JEPA | PLDM [21, 22] | VICReg 계열 7항 손실 | LeWM은 2항(예측 + SIGReg). 튜닝 6개 → 1개 |
| 가우시안 정규화 | **LeJEPA [25]** (Balestriero, LeCun, 2025) | SIGReg, 정규성 검정 | LeWM의 SIGReg 출처. LeJEPA는 표현 학습 논문이고 세계모델 실험이 없음 |
| 보상 기반 잠재 세계모델 | Dreamer [27–29], TD-MPC2 [32] | 재구성 또는 보상 신호 | LeWM은 재구성과 보상이 없음 |
| 생성형 세계모델 | IRIS, DIAMOND, DreamerV4, Genie [3, 4, 6, 9] | 픽셀 재구성 | 픽셀 공간에서 예측 |

**계획 쪽:** 논문은 CEM을 Rubinstein & Kroese [40]에서 가져옵니다. 논문의 CEM 설정(샘플 300, 엘리트 30, 반복 30)은 DINO-WM [18]을 따릅니다. 잠재 직선화 가설 [43–45]은 LeWM이 부록 H에서 측정하는 대상입니다.

**VoE 계열:** 발달심리의 위반 기대 패러다임 [46]을 물리 학습 벤치마크 [47, 48]에 맞춘 것입니다. 이 논문의 VoE는 세계모델의 예측 오차를 놀람으로 씁니다.

**근거 확인:** 위 계보는 논문 §2와 참고문헌 [1–51]에서 확인했습니다. 논문 밖의 인용 관계는 확인하지 않았습니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| ViT-Tiny 인코더, [CLS] 임베딩 | Dosovitskiy [34] | 공식 코드는 HF `vit_hf(size="tiny", patch 14)` |
| 투영 MLP + BatchNorm | 논문 §3.1 (BN 근거 [35]) | 공식 코드는 은닉 2048의 2층 MLP |
| AdaLN-zero 행동 주입 | Peebles & Xie [37] (DiT) | `module.py`의 `ConditionalBlock` |
| SIGReg | LeJEPA [25], Epps–Pulley [38], Cramér–Wold [39] | 투영 1024, 매듭 17 |
| CEM 계획 | Rubinstein & Kroese [40] | 논문 부록 B |
| 자기회귀 예측, 인과 마스크 | 논문 §3.1 | 공식 `ARPredictor` |
| 기준선 GCIQL, GCIVL | Kostrikov [41], Park [49] (OGBench) | DINOv2 패치 임베딩 입력 |
| 측정 환경 | OGBench [49], DMC [51], TwoRoom [22] (Sobal) | PushT는 DINO-WM [18] 데이터 |

## 4. GitHub, 프로젝트 페이지, 공개 자료 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 별 / 포크 | 마지막 push (API) | 최신 커밋 | 비고 |
|---|---|---|---|---|---|---|
| **공식 코드 (LeWM)** | https://github.com/lucas-maes/le-wm | MIT | 4656 / 696 | 2026-05-26 | `8edfeb3` (2026-05-22, "fix data config") | `repo/`에 얕은 clone(`--depth 1`). 정적 읽기만 |
| 프로젝트 페이지 | https://le-wm.github.io | - | - | - | - | 논문 요약, BibTeX는 `@article`(학회 표기 없음) |
| 기반 라이브러리 | https://github.com/galilai-group/stable-worldmodel | MIT | 2316 / 295 | 2026-10-02 | 확인 안 함 (정적 파일 목록만) | 환경, 계획, 기준선 학습 스크립트. 기준선 설정 `scripts/train/pldm.yaml`, `lewm.yaml`만 API로 읽음 |
| 체크포인트 (HF) | https://huggingface.co/quentinll/lewm-pusht 외 `lewm-cube`, `lewm-tworooms`, `lewm-reacher` | MIT (모델 README) | - | 2026-04-16 | - | `weights.pt` 약 72 MB × 4개. 받지 않음 |
| 데이터 (HF) | 같은 컬렉션의 `lewm-pusht`(13.1 GB), `lewm-cube`(46.2 GB), `lewm-tworooms`(3.4 GB), `lewm-reacher`(23.8 GB) | 미확인 | - | 2026-03-27 | - | 압축 파일 크기. 받지 않음 |
| 기준선 체크포인트 (Drive) | README의 Google Drive 폴더 | 미확인 | - | - | - | PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop. 받지 않음 |

**README 내용 중 확인한 사실:**
- 공개 체크포인트 표에는 DINO-WM(proprio 포함)이 Two-Room과 Push-T에만 있고, DINO-WM-noprop은 네 환경 모두 있습니다. 논문 그림 6의 회색 막대(DINO-WM+prop)가 Two-Room과 Push-T에만 있는 것과 맞습니다.
- 공개 체크포인트 표에 **LeJEPA** 기준선이 있지만 논문 본문 결과에는 LeJEPA 수치가 없습니다(`04` D14).
- README는 데이터 파일을 `.h5`라고 쓰는데, 훈련 설정은 Push-T를 `pusht_expert_train.lance`로 부릅니다(`04` D11). HF 데이터 파일은 `pusht_expert_train.h5.zst`입니다.
- 평가 스크립트 `eval.py`는 `model.to("cuda")`를 씁니다. MuJoCo는 `MUJOCO_GL=egl`로 설정합니다(`04` §4).

**이 분석에서 하지 않은 것:** 공식 코드 실행, 설치, 가중치와 데이터 다운로드, 기준선 코드 정적 읽기(stable-worldmodel 본문).

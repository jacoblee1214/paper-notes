# 04. 공식 코드 대조와 구현 가능성

대상: `GaTech-RL2/EgoEngine-CoRL2026`, 커밋 `b1363c2` (2026-10-16). `repo/`에 얕은 복제를 했고, **정적으로만 읽었습니다.** 실행, 설치, import, 가중치와 데이터 다운로드는 하지 않았습니다. 코드 안의 내용은 외부 데이터로 취급했습니다. 코드의 설정 파일과 문서에 있는 값은 "릴리스가 주장하는 값"으로 인용합니다. 동작은 확인하지 않았습니다.

## 1. 저장소 구성 (정적 확인)

| 경로 | 내용 | 논문 대응 |
|---|---|---|
| `egoengine/` | CLI (`list`, `validate`, `demo`, `action`, `render`, `composite`, `export`, `run`), HDF5 내보내기 | §3 전체 연결 |
| `action/examples/run_mjwp_mixed.py` (1354줄) | Replay → MPC → RL 모드 전환 러너 | §3.2.2, 부록 C.1 |
| `spider/mode_switching.py` (151줄) | chunk 20, 2-chunk 대비 1-chunk 커밋 | 부록 C.1 |
| `spider/simulators/mjwp.py` | 물체 추적 종료 규칙, 보상 가중치 | 부록 C.2–C.3 |
| `spider/rl/` | goal-conditioned 잔차 PPO 관측과 행동 | §3.2.2, 부록 C |
| `visual/compositor.py` (431줄), `egoengine/visual.py`, `egoengine/robot_visual.py` | 로봇 렌더링, 가림 마스크, 합성 | §3.3 |
| `action/configs/*.json` (30개) | 사례별 설정 (legacy 러너용) | 부록 C.2 |
| `examples/prepared/` | TACO 3 (`brush`, `cut`, `scrape`), Aria 3 (`mustard_1..3`) | §4 일부 |
| `docs/*.md` | 입력 규격, 모드 전환, RL 레시피, 재현 방법 | §3 |
| `UPSTREAMS.json`, `THIRD_PARTY_NOTICES.md` | 상류 저장소와 커밋 | — |

**릴리스 범위 (README와 `docs/REPRODUCIBILITY.md`의 명시):**
- "FoundationPose and early perception/reconstruction are outside this release." 디지털 트윈의 인식 단계(FoundationStereo, SAM2, FoundationPose, AprilTag 보정)는 없습니다.
- 사람 제거 inpainting(Inpaint-Anything v2), 배경 제거 모델 학습은 없습니다. 사전 처리된 배경 영상(`removed_w_mask_5.mp4`)을 입력으로 받습니다.
- 정책 학습(HPT, flow matching 디코더)과 실로봇 실행 코드는 없습니다. `grep`에서 HPT와 flow matching 용어가 저장소에 없습니다.
- H2S2R(RL 기준선) 코드는 없습니다. `visual/retargeting/kinematics/rb_loader.py`에 H2S2R 저장소 URDF 경로가 있을 뿐입니다.

## 2. 논문 ↔ 코드 대조 (D1–D18)

판정 기호: **일치** (논문과 같음), **차이** (논문과 다른 값 또는 규칙), **없음** (논문에 있는 것이 릴리스에 없음), **미정** (논문에 없어 코드 값을 쓸 수밖에 없음).

### 2.1 행동 분기

| 번호 | 항목 | 논문 | 코드 (정적 확인) | 판정 |
|---|---|---|---|---|
| D1 | 물체 추적 종료 규칙 | 식 C.4: 가중합 e = λ_R e_R² + λ_p e_p² > C 이면 종료 (단일 경계 C) | `spider/simulators/mjwp.py` `get_terminate`: **위치 노름 > object_pos_threshold 또는 회전 차 노름 > object_rot_threshold**의 OR 규칙 (두 경계를 따로 씀). 보상 오프셋은 sqrt(pos_term² + rot_term²) | **차이.** 종료 영역의 모양이 다릅니다. 논문의 단일 경계 C와 코드의 OR 규칙은 같지 않습니다. |
| D2 | TACO 경계값 | 과제별 회전 경계 0.9 rad (Cut), 1.2 rad (Scrape, Brush), 1.5 rad와 위치 0.12 m (Pour). 나머지는 "물체 움직임에서 정함" (C.2) | `action/configs/taco_rl.json`: 0.05 m, **1.5 rad** (모든 과제 공통). `taco_mpc`, `taco_replay`: 0.05 m, **1 rad**. 과제별 파일 없음 | **차이.** 릴리스 설정에 과제별 값이 없습니다. |
| D3 | 통합 CLI의 기본 경계 | 논문 값을 사용 | `egoengine/cli.py`의 `egoengine action`은 `run_mjwp_mixed`에 **경계 인자를 넘기지 않습니다.** 러너 기본값 0.05 m, 1.0 rad 사용 | **차이.** README의 "한 명령 재현" 경로는 과제별 경계를 쓰지 않습니다. |
| D4 | Aria 실로봇 경계 | 0.08 m, 2.5 rad (C.2, 네 과제 공통) | `aria_mustard_rl_real.json` **0.5 rad**, `aria_drawer_rl_real.json` **0.1 m, 3.5 rad**, `aria_flower_rl_real.json`과 `aria_hammer_rl_real.json` 0.08 m, 2.5 rad | **차이.** 네 과제 중 둘만 논문 값과 같습니다. |
| D5 | Aria 실로봇 잡음·보상·DR 설정 | "네 과제가 공유" (C.2): 위치 잡음 0.045, 회전 잡음 1.0, 관절 잡음 0.8, 기저 위치 0.2, 기저 회전 1.0, 매끄러움 0.8, 쌍 여유 ±0.005, xy ±0.015, 질량 0.8–1.2, DR 8회 | `aria_hammer_rl_real.json`만 이 값과 **정확히 같습니다.** 나머지는 과제별로 다릅니다. 예: Mustard는 회전 잡음 0.5, 관절 잡음 0.4, 기저 위치 0.4, 기저 회전 1.2, 매끄러움 0.5, 쌍 여유 [−0.005, 0.002], xy ±0.01. Flower는 위치 잡음 0.06, 회전 잡음 0.6, 기저 회전 0.2 | **차이.** 논문의 "공유"는 해머 사례에만 맞습니다. |
| D6 | 접촉 보너스 | 모든 과제에 공유 (C.2 "Across all tasks ... contact bonus") | Aria RL 설정은 2.0, `taco_rl.json`은 **0.0**. 통합 러너 기본값도 0.0 | **차이.** TACO 설정에는 접촉 보너스가 없습니다. 통합 CLI 기본 경로에도 없습니다. |
| D7 | 기준선 RL (H2S2R) | 표 2 기준선 H2S2R [30] | 코드 없음. `scripts/benchmark_case.py`의 `pure_rl` 모드는 **CoDesign 잔차 PPO 포트** (`docs/RL_RECIPE.md`). 논문의 H2S2R과 같은 방법인지 확인할 수 없음 | **없음 / 미정.** |
| D8 | 모드 전환 규칙 | 2-chunk 계획, 현재 chunk만 커밋, 실패 시 Replay로 복귀, 20 스텝 chunk (§3.2.2, C.1) | `spider/mode_switching.py`: chunk 20, `lookahead_chunks=2`, 현재 chunk와 경계 상태만 커밋. `run_mjwp_mixed.py` 기본 `window_size=20`, `lookahead_chunks=2`. 실패 부분 궤적은 내보내지 않음 | **일치.** |
| D9 | MPC 설정 | 짧은 구간 샘플링 (C는 기술 안 함) | 기본 샘플 8, 반복 2, 온도 0.1 (`run_mjwp_mixed.py`) | **미정** (논문은 수치를 주지 않음). README의 기본값과 일치. |
| D10 | RL 학습 예산 | 논문 본문에 환경 수, 반복 수 없음 | 통합 러너 기본: 환경 1024, 전체 1500 반복, 창당 300 (README "Defaults ... 1024 RL worlds ... 1500-iteration"). 경계 설정 `aria_*_rl_real.json`: 환경 **4096**, 반복 300 | **미정, 설정이 두 가지.** |
| D11 | RL 세부 (관측, 잡음 범위, 학습률) | PPO만 언급 (§3.2.2) | 관측 70차원, 잔차 범위 0.10 m, 0.30 rad, 0.30 rad, PPO 16 스텝, 5 epoch, lr 3e-4, [256,128] MLP (`docs/RL_RECIPE.md`) | **미정** (논문 부재). 재현하려면 이 값이 필요합니다. |
| D12 | 리타게팅 | 식 1 (MINK, 손끝 5개와 손목) | `visual/retargeting/` 있음. 릴리스 설명: "the released retargeting code contains no per-task trajectory corrections" (README). UPSTREAMS.json은 "Remove task-specific trajectory corrections"를 상류 변경으로 적음 | **차이 가능.** 상류 저장소에 과제별 보정이 있었고 릴리스에서 뺐습니다. 논문 결과에 그 보정이 쓰였는지는 알 수 없습니다. |
| D13 | 보상 형태 | 식 C.3: r = C − 가중 오차 | 코드: C − 가중 오차 형태 (`_termination_reward_offset`, sqrt 기준). 가중치 pos 1.0, rot 0.3 (기본) | **일치** (형태). 가중치와 C 값은 D1–D2 참조. |

### 2.2 시각 분기

| 번호 | 항목 | 논문 | 코드 (정적 확인) | 판정 |
|---|---|---|---|---|
| D14 | 사람 제거 | SAM2 마스크 + Inpaint-Anything v2 (§3.3) | 코드 없음. 입력은 `removed_w_mask_5.mp4` (사전 처리된 배경) | **없음.** |
| D15 | 가림 마스크 (식 3) | 두 번 렌더링 (로봇 투명, 불투명)의 RGB 차이 > 0 | `egoengine/robot_visual.py`: 로봇 팔 색을 투명 (`geom_rgba[arm_ids, 3] = 0`)과 불투명 두 경우로 렌더링하는 흐름 확인. 임계값 0이 그대로 쓰였는지는 정적 읽기로 확정하지 못함 | **대체로 일치 (미확정).** |
| D16 | 합성 (식 4) | M ⊙ R + (1 − M) ⊙ I | `visual/compositor.py`의 레거시 합성기 사용. Aria에는 `edge_blur=1, edge_erode=1, rim_blend=0.05, rim_noise=0.5, fg_noise=0.1`이 추가로 적용 (`egoengine/visual.py`). 논문에 없는 후처리 | **차이 (추가 후처리).** |
| D17 | 디지털 트윈 자동화 | §3.1: FoundationStereo, SAM2, FoundationPose, AprilTag 보정 (부록 A.1–A.3) | 코드 없음. 프로젝트 README는 이 단계를 릴리스에서 뺐다고 적음 | **없음.** |

### 2.3 정책 학습과 공개 범위

| 번호 | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| D18 | 정책 학습 (HPT, flow matching, 10 Euler 스텝) | §3.4, 부록 D | 학습 코드 없음. REPRODUCIBILITY: "Policy checkpoints and learning traces are generated by training and are not downloaded implicitly." 실로봇 정책 코드 없음 | **없음.** |

### 2.4 문서 사이의 차이 (논문 대 프로젝트 페이지 대 릴리스 문서)

| 번호 | 항목 | 내용 | 판정 |
|---|---|---|---|
| D19 | 실로봇 데이터 수집 장비 | 논문 본문: Aria Gen2로 일관. 프로젝트 페이지: "일부 예시는 이전 로봇 구성과 **Aria Gen 1** 안경으로 수집" | **차이 (문서).** 논문 부록이 이 예시를 설명하는지 확인되지 않음. |
| D20 | 효율 측정 장비 | 논문: RTX 4090 한 대 (§4.3). 릴리스 검증: A40 한 대 (`docs/MODE_SWITCHING.md`, gpu_smoke) | 장비가 다름. 측정 결과 재현에는 영향이 있음 |
| D21 | 비용 지표 | 논문 표 2 Cost는 "device-agnostic" 스텝 지표. 릴리스: "Do not recompute Cost ... automatic Cost rankings are disabled. ... No verified simulation-step reduction is claimed by this release." | **차이 (핵심).** 릴리스가 논문의 비용 주장을 확인하지 않습니다. |
| D22 | 데이터 규모 | 논문: TACO 2,500 시퀀스, Aria 200편 (4 과제). 릴리스: TACO 3개, Aria 3개 (mustard만, `visual_inputs/Aria`) | **차이 (범위).** Aria의 drawer, flower, hammer 입력은 릴리스에 없습니다. `action/configs/aria_drawer_*`등의 설정은 있습니다. |
| D23 | TACO 사례 완성도 | 논문 표 2는 TACO 16쌍에서 평가 | 릴리스 README: 세 사례 모두 "lack prepared demonstrator-free backgrounds" (최종 합성 불가). `cut_knife_plate`는 저장된 리타게팅의 로봇 마스크가 210 프레임 모두 비어 있음 (`docs/REPRODUCIBILITY.md`). "Successful action generation for all three bundled TACO interactions has not been established." | **차이 (재현 불가 사례).** |
| D24 | 행동 성공의 안정성 | 논문: 성공률 보고 | 릴리스: "RL outcomes vary with sampling and GPU contact dynamics. A fixed seed does not guarantee identical policies or mode sequences across environments." | **차이 (재현성).** |

### 2.5 일치한 부분 (확인)

- 모드 전환 순서와 커밋 규칙 (D8)이 논문과 같습니다.
- 손목 부동(floating-wrist) EEF와 XHand 조인트를 행동 공간으로 씁니다 (논문 C.5와 일치).
- 일치 수는 적지만, 릴리스의 `docs/validation/RELEASE.json`에 따르면 릴리스 검증은 **시각 분기 렌더링, HDF5 내보내기, 행동 분기 기본 흐름의 소형 실행(GPU 한 대, 5 프레임)**에 한정됩니다. 논문 결과 표의 재현은 릴리스에 포함되지 않았다고 명시합니다.

## 3. 라이선스와 공개 범위

- **논문:** arXiv 비독점 배포 라이선스 1.0 (`nonexclusive-distrib`). **CC BY가 아닙니다.** 따라서 논문 그림을 공개 산출물에 넣지 않습니다. 원문 PDF의 저작권 문구는 `pdftotext` 결과에서 찾지 못했습니다(© 또는 All rights reserved 없음).
- **코드:** 저장소 전체는 "Other"입니다. `LICENSE` 파일에 따르면 `spider/`와 `action/`은 **CC BY-NC 4.0**(SPIDER 유래, 비상업), `visual/`은 MIT(robosuite 유래)입니다. 자산과 데이터는 "원래 조건을 유지"합니다.
- **이 저장소의 공개 범위:** 코드는 복사하지 않았습니다. 설명과 대조 결과만 적었습니다. 공개본에 코드 조각은 넣지 않습니다.
- **절대 경로:** 릴리스 설정과 코드에는 원 저자의 기기 경로(`/home/...`, `/mnt/...`)가 남아 있습니다. 이 문서들은 이 경로를 옮기지 않았습니다.

## 4. 구현 가능성 판정

**판정: 이 맥(Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)에서 논문의 전체 결과는 재현할 수 없습니다. 시각 분기 일부와 데이터 검사는 가능합니다.**

| 단계 | 이 맥에서 | 필요 자원 | 근거 |
|---|---|---|---|
| 시각 분기: 로봇 렌더링과 합성, HDF5 내보내기 (Aria mustard 3 사례) | **조건부 가능.** macOS를 공식 지원하는 시각 환경 (Python 3.10, MuJoCo 3.5, OpenGL, torch 없음). 설치 전에 목록을 보여 주고 승인을 받아야 합니다 | OpenGL, 수 GB 디스크, 분 단위 CPU 렌더링 | README "macOS or Linux with OpenGL" |
| 행동 분기: Replay | 공식 경로는 CUDA 환경 (action 프로파일)이며 CPU 실행은 확인하지 않음 | MuJoCo Warp, Warp CUDA | README: "Use a separate Linux Python 3.12 environment with an NVIDIA CUDA runtime" |
| 행동 분기: MPC와 RL (모드 전환) | **불가 (공식 요구 사항 불충족).** 릴리스가 CUDA를 요구하며 MPS 경로는 확인하지 않았습니다 | NVIDIA GPU (논문은 RTX 4090, 릴리스 검증은 A40) | `docs/REPRODUCIBILITY.md` |
| 디지털 트윈 (인식과 물체 추적) | **불가 (릴리스에 없음)** | FoundationStereo, SAM2, FoundationPose, GPU | D17 |
| 사람 제거 inpainting | **불가 (릴리스에 없음)** | Inpaint-Anything v2 등 GPU 모델 | D14 |
| 정책 학습 (HPT, flow matching) | **불가 (릴리스에 없음)**. 학습 자체도 GPU 필요 | 코드 재구현 + GPU | D18 |
| 실로봇 실행 | **불가** | RB-Y1, XHand, Aria Gen2 | — |
| 표 1 FD 재계산 | 이론상 가능하나 **특징 벡터가 없음** | 원자료(특징) 없음 | §4.1 본문 |
| 표 3, 4 통계 재계산 | **가능, 완료** (`03_analysis.md`) | 보고 값만 필요 | — |

**노력 추정 (측정 전, 논문 수치를 바탕으로 한 추정):**
- 시각 분기 3 사례 렌더링과 합성 확인: 반나절에서 하루 (설치 승인 후).
- 행동 분기 재현 (GPU 한 대): 논문 처리량 2.88 demos/h를 기준으로 Aria 200편이면 약 70시간, TACO는 Aria보다 긴 궤적 (2.39배)이라 더 걸립니다. **측정하지 않은 추정입니다.** 과제별 설정 (D2–D5)을 맞추는 작업이 먼저 필요합니다.
- 디지털 트윈과 정책 학습까지 포함하면 릴리스 밖의 구현이 필요합니다. 수 주에서 수 개월 규모로 추정합니다 (측정 없음).

**결론:** 이 논문의 핵심 주장 (행동 분기의 역할, 비용 절감)은 GPU가 있는 Linux 기기와 과제별 설정 맞추기가 있어야 확인할 수 있습니다. 이 맥에서는 시각 분기 데모와 보고 값 재계산까지가 현실적인 범위입니다.

## 5. 확인하지 못한 것

- 동작 확인: 코드를 실행하지 않았으므로 D15 (두 번 렌더링의 임계값)와 D12 (과제별 보정 사용 여부)는 정적 읽기로만 판단했습니다.
- 상류 저장소 (`Randle-Github/EgoEngine-MPC`, `GaTech-RL2/robosuite`)는 메타데이터만 확인했습니다. 복제하지 않았습니다.
- GitHub 저장소의 이슈와 릴리스 노트는 확인하지 않았습니다.
- 논문 부록 E.1의 "1 demonstration으로도 양의 성공" (Flower)은 그림에서만 나옵니다. 텍스트 추출에 그림 수치가 없습니다.

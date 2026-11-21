# 02. 선행연구 계보, 공식 코드, 이 저장소의 관련 분석 (Genie Envisioner)

## 1. 계보 (논문 §1, §7 기준)

| 계열 | 대표 연구 (논문 인용) | Genie Envisioner와의 관계 |
|---|---|---|
| 범용 영상 세계모델 | Genie (Bruce et al., 2024), Cosmos (Agarwal et al., 2025), Cosmos-Predict2 | GE-Base 백본 후보 (Cosmos2 2B). Genie 3 등은 논문의 비교 대상 |
| 영상 확산 백본 | LTX-Video (HaCohen et al., 2024) | GE-Base 기본 백본 (LTX-Video 2B). GE-Act는 이 변형 위에서 돌아감 |
| 로봇 행동 VLA | π0 (Black et al., 2024), GR00T N1 (Bjorck et al., 2025), UniVLA (Bu et al., 2025b), GO-1 | 실험 비교 대상 (§3.4, §4) |
| 영상 기반 정책 | UniPi 계열, GR-2, Vidar, DreamGen, Cosmos Policy | 같은 흐름. 영상을 만들고 행동을 뽑는 방식 (Fast-WAM 논문도 GE를 이 계열로 인용) |
| 행동 조건 영상 모델 | IRASim, Ctrl-World (행동 조건 영상 생성) | GE-Sim과 같은 목적 (행동 대 영상). 논문은 IRASim 계열을 "단일 시점" 한계로 묶음 |
| 평가용 영상 모델 | VBench (Huang et al., 2024), EWMBench (Yue et al., 2025) | EWMBench는 같은 팀의 이전 연구 (2025-05-16 공개). VBench와 사람 순위 비교 (§6.5) |
| 선행 시리즈 | EnerVerse (Huang et al., 2025), EnerVerse-AC (Jiang et al., 2025) | 논문 감사의 말에 나온 전신 연구 |

## 2. 공식 코드와 관련 저장소 (GitHub 표)

| 저장소 | 관계 | 라이선스 | 마지막 push | 스타 | 상태 |
|---|---|---|---|---|---|
| [AgibotTech/Genie-Envisioner-V1](https://github.com/AgibotTech/Genie-Envisioner-V1) | **공식 구현** (이 논문). 프로젝트 페이지 링크 `github.com/AgibotTech/Genie-Envisioner`가 이 저장소로 리디렉션됩니다 | GitHub API: 없음 (LICENSE 파일 없음). README: 비 라이브러리 코드와 데이터는 **CC BY-NC-SA 4.0**, `models/ltx_models`, `models/cosmos_models`, `models/pipeline`, `web_infer_utils/openpi_client`는 Apache-2.0 (Diffusers, LTX-Video, Cosmos, openpi에서 수정) | 2026-09-10 (`d41358f`, "Update README.md") | 592 | 얕은 복제 완료 (`repo/`, 정적 읽기 전용, 설치·실행 안 함, 커밋 `d41358f`, 2026-09-10) |
| [AgibotTech/GE-Sim-V2](https://github.com/AgibotTech/GE-Sim-V2) | 후속작 GE-Sim 2.0 (2026-05-22 생성). 논문과 다른 버전 | Apache-2.0 | 2026-10-20 | 166 | 메타데이터만 확인. 복제하지 않음 |
| Genie Envisioner - Act v2.0 (ge-act-v2.github.io) | 후속작 GE-Act 2.0 (README 2026-09-10 공지) | 확인 못함 | 확인 못함 | 확인 못함 | 프로젝트 페이지 HTTP 200. 저장소 메타데이터 미확인 |
| Genie Envisioner - Sim v2.0 (ge-sim-v2.github.io) | 후속작 GE-Sim 2.0 프로젝트 페이지 (2026-05-28 공지) | 확인 못함 | 확인 못함 | 확인 못함 | HTTP 200. 메타데이터 미확인 |
| [AgibotTech/EWMBench](https://github.com/AgibotTech/EWMBench) | EWMBench 공식 코드 (논문 §6) | 없음 (API: none) | 2025-06-13 | 132 | 복제하지 않음. 정적 읽기 대상 아님 |
| [Genie-Envisioner 프로젝트 페이지](https://genie-envisioner.github.io/) | 공식 소개 페이지 (SPA, JS 번들에서 링크 확인) | - | - | - | 학회 표기 없음 |

- 가중치: GE-Base 가중치 (Hugging Face `agibot-world/Genie-Envisioner`, 2025-08-14), GE-Sim Cosmos 변형 (ModelScope, 2025-10-22), GE-Act (CALVIN 학습, 2025-12-18). **논문 실로봇 결과를 낸 GE-Act 가중치는 저장소 공지에서 찾지 못했습니다.**
- 서로 다른 버전: 공식 저장소의 README는 Genie Envisioner "v1.0"입니다. 이 분석의 대상은 논문이며, 후속 버전 (GE-Act 2.0, GE-Sim 2.0)은 분석하지 않았습니다.
- 서로 다른 날짜: 논문 v1 (2025-08-07) 이전에는 코드가 없었고, 2025-08-13에 공개되었습니다. 논문 v3 (2025-11-04)의 "will be open-sourced upon publication" 문장은 이미 낡았습니다.

## 4. 구성 요소별 출처 (표)

| 구성 요소 | 출처 | 근거 |
|---|---|---|
| 다시점 교차 자기주의, 시점 임베딩 | 논문 고유 (§2.1) | 공식 코드 `models/ltx_models/transformer_ltx_multiview.py`에서 확인 (`04` D1) |
| 희소 메모리 | 논문 고유 (§2.1) | 데이터 로더 `n_previous: 4`, `previous_pick_mode: random` (`04` D7) |
| T5-XXL 동결 텍스트 인코더 | 기존 (Raffel et al., 2020) | 설정 `T5EncoderModel`, `caption_channels: 4096` |
| 흐름 매칭 디노이징 | 기존 (Lipman et al. 계열) | `FlowMatchEulerDiscreteScheduler` |
| GE-Act 병렬 행동 분기 | 논문 고유 (§3) | 설정 `action_expert: true`, 헤드 16 x 32 (`04` D5) |
| 비동기 1:6 추론 | 논문 고유 (§3.3) | 데이터 로더 `video_temporal_stride = 54 // 9 = 6` (`04` D6) |
| GE-Sim pose2image | 논문 고유 (§5.1) | `utils/get_traj_maps.py` (`04` D9) |
| EWMBench | 같은 팀의 선행 연구 (Yue et al., 2025, arXiv:2505.09694) | 논문 §6, 별도 저장소 |

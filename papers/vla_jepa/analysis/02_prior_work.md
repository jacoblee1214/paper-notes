# 02. 선행연구, 대안, 구성 요소, 이 저장소의 관련 분석 (VLA-JEPA)

## 1. 계보

잠재 행동 사전학습은 게임 영상(ILPO, LAPO, Genie)에서 시작해 로봇으로 넘어왔습니다. 로봇 쪽에서는 LAPA, IGOR, UniVLA, Moto, AdaWorld, CoMo, StaMo가 프레임 전이에서 이산 또는 연속 모션 토큰을 뽑고, 이를 VLA가 예측한 뒤 실제 제어로 옮깁니다. villa-X, XR-1, CLAP, VITA는 사람 영상과 로봇 영상에 공통 코드북을 씁니다. 이 흐름에서 VLA-JEPA의 차이는 **잠재 행동을 픽셀 차이가 아니라 동결 V-JEPA2 특징의 미래 상태 예측으로 정의한다**는 점입니다(`01` §1–2).

미래를 예측하는 VLA(WorldVLA, CoT-VLA, 하위 목표·롤아웃 예측)와 객체 중심 조건 VLA도 비교 대상입니다. 논문은 이들이 대량의 행동 라벨 데이터에 기대므로 사람 영상만으로 사전학습하는 VLA-JEPA와 대비된다고 봅니다(2.1절).

## 2. 가까운 대안 (논문 비교 대상)

| 방법 | 핵심 | 논문 비교 | 비고 |
|---|---|---|---|
| LAPA [79] | 잠재 행동 VQ, 성공 궤적 사용 | LIBERO, SimplerEnv | SimplerEnv WidowX에서 시뮬 학습 데이터 사용(∗) |
| UniVLA [13] | 과제 중심 잠재 행동 | LIBERO, LIBERO-Plus, SimplerEnv | 사람 영상 사용 |
| villa-X [19] | 로봇과 사람의 공통 잠재 행동 | LIBERO, SimplerEnv | 대량 로봇 데이터 |
| OpenVLA-OFT [40] | 미세조정 레시피 | 모든 벤치마크 | LIBERO 평균 최상위 기준선 |
| π0, π0-Fast, π0.5 [9, 61, 34] | 흐름 VLA | 모든 벤치마크, 실로봇 | 실로봇 비교 기준 |
| WorldVLA [16], CoT-VLA [87] | 행동-세계 자기회귀 | LIBERO, LIBERO-Plus | LIBERO-Plus에서 낮음 |
| Moto [20] | 잠재 모션 토큰 | SimplerEnv | |

## 3. 구성 요소별 출처

| 구성 요소 | 출처 (논문 참고문헌 번호) | 메모 |
|---|---|---|
| JEPA 목표, 동결 타깃 인코더 | I-JEPA [1], V-JEPA [4], V-JEPA 2 [2] | 식 (4)의 근거 |
| 세계 상태 인코더 | V-JEPA 2 (`facebookresearch/vjepa2`, MIT) | 동결, 두 뷰를 이어 붙임 |
| VLM 백본 | Qwen3-VL [3], Qwen3 [77], SigLIP-2 [70] | 2B 사용 (A.1) |
| 흐름 매칭 헤드 | 흐름 매칭 (논문 식 7–8). DiT [60] | GR00T N1.5 헤드 계보 (코드 헤더 주석) |
| 사람 영상 데이터 | Something-Something-v2 [31] | 220K 클립 |
| 로봇 데이터 | DROID [38], LIBERO [49], Fractal, BridgeV2 [25] | |
| 평가 벤치마크 | LIBERO [49], LIBERO-Plus [28], SimplerEnv [44] | |

**재사용된 코드 기반.** 저장소 README는 `starVLA`(`github.com/starVLA/starVLA`)를 기반으로 한다고 씁니다. 기반 프레임워크의 라이선스는 "Other"로 표시되고 저장소 디렉터리에 별도 고지가 있습니다. 이번 분석에서는 `THIRD_PARTY_NOTICES.md`를 읽지 않았습니다(`04` §4).

## 4. 공식 코드, 프로젝트 페이지, 관련 저장소 (확인 2026-11-20)

| 자료 | 링크 | 라이선스 | 별 / 포크 | 마지막 활동 | 메모 |
|---|---|---|---|---|---|
| **공식 코드** | https://github.com/ginwind/VLA-JEPA | Apache-2.0 | 638 / 59 | push 2026-09-29 (UTC), 최신 커밋 `ec907778` (2026-09-30 +08:00) | `repo/`에 얕은 clone, 정적 읽기만. 실행과 설치는 하지 않음 |
| 프로젝트 페이지 | https://ginwind.github.io/VLA-JEPA/ | - | - | - | 초록, 링크, BibTeX(`@misc{vlajepa2026}`). 학회 표기 없음 |
| 모델 가중치 | https://huggingface.co/ginwind/VLA-JEPA | 확인 안 함 | - | - | 다운로드하지 않음 (승인 정책) |
| 기반 프레임워크 | https://github.com/starVLA/starVLA | Other | 3965 | push 2026-11-18 | 코드 기반. 라이선스 세부는 확인 안 함 |
| 인코더 | https://github.com/facebookresearch/vjepa2 | MIT | 4902 | push 2026-10-29 | V-JEPA 2 공식 저장소 |
| arXiv | https://arxiv.org/abs/2602.10098 | CC BY 4.0 | - | v2 2026-02-14 | 라이선스 재확인 완료 |

**학회 표기.** GitHub 저장소 설명에 "[ECCV 2026]"이 있습니다. 프로젝트 페이지, arXiv 메타데이터, 논문 본문에는 학회명이 없습니다. 웹 검색으로 ECCV 2026 공식 채택 목록을 찾았지만 해당 논문을 확인하지 못했습니다. 따라서 학회는 **저장소 표기로만 확인됩니다.**

## 6. 이 논문의 위치 (요약)

- **무엇을 새로 했나:** 잠재 행동 사전학습의 예측 공간을 픽셀에서 동결 V-JEPA2 특징으로 옮겼습니다. 사람 영상과 로봇 데이터를 같은 세계모델 목표로 묶었습니다.
- **무엇은 기존 조합인가:** V-JEPA 2 인코더, Qwen3-VL 백본, 흐름 매칭 헤드, 기존 벤치마크. 구성 요소는 대부분 기존 것입니다.
- **가장 가까운 대안과의 차이:** 같은 "사람 영상 잠재 행동" 계열(LAPA, UniVLA, villa-X)과 비교하면 예측 공간과 누수 설계가 다릅니다. 그러나 누수 설계의 실제 효과는 논문에서 정량화되지 않았습니다(`01` §6).

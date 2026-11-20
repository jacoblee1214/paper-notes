# 02. 선행연구와 관련 자료

## 1. 계보

- **영상 모델을 정책 백본으로 쓰는 흐름.** UVA, Video Policy, UWM처럼 영상 예측과 행동 예측을 함께 하는 방법이 있습니다(§2 Related Work). 이 논문은 이들과 달리 사전학습 영상 모델을 그대로 초기화로 쓰고, 행동 전용 모듈을 붙이지 않습니다.
- **단계 학습과 별도 행동 모듈.** Video Policy, Unitree UnifoLM-WMA, Latent Policy Steering 등은 영상 미세조정 뒤 행동 모듈을 따로 학습합니다(§2).
- **세계 모델과 가치 함수를 함께 쓰는 흐름.** Dyna, MBPO, TD-MPC, Dreamer 계열과 FLARE, SAILOR, Latent Policy Steering이 있습니다. 이 논문은 정책, 세계 모델, 가치 함수를 한 구조로 묶는다는 점을 차별점으로 듭니다(§2).
- **VLA 계열 기준선.** π0, π0.5, OpenVLA-OFT, CogVLA, UniVLA, GR00T-N1.5, DP-VLA 등이 표 1–3에 있습니다(§5).

## 2. 가까운 대안 (논문이 직접 비교한 것)

| 방법 | 성격 | 이 논문의 위치 |
|---|---|---|
| UVA (Li 2025a) | 행동과 영상의 통합 모델 | LIBERO-Long 90.0 (표 1), RoboCasa 50.0 (표 2) |
| Video Policy (Liang 2025) | 영상 미세조정 뒤 행동 모듈 | RoboCasa 66.0 (표 2) |
| FLARE (Zheng 2025) | 미래 토큰을 DiT 시퀀스에 추가 | RoboCasa 66.4 (표 2) |
| Latent Policy Steering (Wang 2025) | 세계 모델과 별도 가치 함수 | 관련 연구로만 언급 |
| π0.5, OpenVLA-OFT+ | 대규모 로봇 데이터 VLA | ALOHA에서 직접 비교 (표 3) |

## 3. 이 저장소의 관련 분석

아래 폴더는 `cosmos_policy/`와 비교하거나 연결할 수 있습니다. 폴더의 README와 분석 문서를 읽고 정리했습니다. 각 연결은 해당 폴더의 근거 문서를 가리킵니다.



## 4. 공식 코드와 프로젝트 페이지

| 항목 | 내용 | 확인 |
|---|---|---|
| 공식 코드 | github.com/NVlabs/cosmos-policy | GitHub API 조회 |
| 라이선스 | Apache-2.0 (저장소 LICENSE 파일) | 조회, `repo/LICENSE` |
| 별 / 포크 / 열린 이슈 | 898 / 109 / 18 | GitHub API (2026-11-20 조회) |
| 최근 커밋 | 18a2acc, 2026-01-22 (README 업데이트) | `git log` (shallow clone) |
| 저장소 생성 | 2025-12-16 | GitHub API |
| 가중치 | HuggingFace `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`, `nvidia/Cosmos-Policy-ALOHA-…`, `nvidia/LIBERO-Cosmos-Policy`(데이터셋) | README, LIBERO.md |
| 프로젝트 페이지 | `research.nvidia.com/labs/dir/cosmos-policy/`는 `research.nvidia.com/labs/cosmos-lab/cosmos-policy/`로 영구 이동 | curl 확인 |
| 요약 영상 | youtu.be/V2qdFD9n5BM | 저장소 README |
| 가중치 라이선스 | Hugging Face 모델 카드는 조회하지 않았습니다 | 미확인 |

- 저장소는 `repo/`에 얕은 복제(clone)했습니다. 실행, 설치, import는 하지 않았습니다.
- `repo/`는 `.gitignore`에 들어 있습니다.
- 학회: ICLR 2026. 근거는 `README.md`의 venue 항목을 보십시오.

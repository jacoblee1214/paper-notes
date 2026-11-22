# 02. 선행연구, 계보, 관련 자료

## 1. 계보

- **RoboTwin 1.0 (ECCV Workshop 2024 초판, CVPR 2025 Highlight):** 실제 시연을 시뮬레이션 디지털 트윈으로 옮기는 양팔 벤치마크. 이 논문은 1.0을 "실제 시연과 짝을 이룬 복제본"으로 설명한다 (§5.1). 1.0 논문은 arXiv:2504.13059 (본문에서는 [34]).
- **CVPR 2025 MEIS 챌린지 기술 보고서 (arXiv:2506.23351):** 같은 플랫폼 위의 대회 보고서. 본 논문 [8].
- **RoboTwin 2.0 (본 논문, arXiv:2506.18088):** MLLM 코드 생성 + 도메인 랜덤화 + 객체 라이브러리 + 벤치마크.

## 2. 가까운 대안과 본문 비교 (표 6, 부록 B)

본문 표 6은 여섯 자원을 과제 수, 도메인 랜덤화, 자동 데이터 생성, VLA 학습·평가 지원 여부로 비교한다.

| 자원 | 과제 수 (표 6) | 도메인 랜덤화 | 자동 생성 | VLA 학습·평가 |
|---|---|---|---|---|
| Meta-World | 50 | ✕ | ✓ | ✕ |
| Robosuite | 9 | ✕ | ✕ | ✕ |
| RoboCasa | 25 | ✓ | ✕ | ✕ |
| ManiSkill2 | 20 | ✕ | ✓ | ✕ |
| AutoBio | 16 | ✕ | ✓ | ✓ |
| RoboTwin 1.0 | 14 | ✕ | ✓ | ✓ |
| RoboTwin 2.0 | 50 | ✓ | ✓ | ✓ |

**관찰:** 표 6에서 RoboCasa의 "Domain Randomization ✓, Auto Data ✕"와 Meta-World의 "Auto ✓"는 저자 주장이다. 여기서 "Auto Data Generation"의 정의는 본문에 없다. 비교의 기준이 느슨하다. 또 RoboCasa의 인용이 [50] (Meta-World)로 되어 있다 (표 6의 RoboCasa 행). 인용 오류로 보인다. 본문 §5.1은 RoboCasa를 [35]로 인용하므로 표 6의 인용 번호가 틀렸다.

본문에서 언급된 가까운 자원: SAPIEN [48], ManiSkill2 [16], Meta-World [50], CALVIN [32], LIBERO [30], RoboVerse [15], RoboCasa [35], AgiBot World [4], RoboMIND [47], Open X-Embodiment [36], Bridge [12]. 생성 파이프라인 측면에서는 GenSim2 [20], RoboGen [43], RoboCodeX [33], RoboScript [6]와 비교한다 (§2.1).

## 4. GitHub, 프로젝트 페이지, 관련 저장소 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 별 / 포크 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| **공식 코드 (RoboTwin)** | https://github.com/RoboTwin-Platform/RoboTwin | MIT | 3,073 / 539 | 2026-09-24 (API `pushed_at`) | `repo/`에 얕은 clone, 커밋 `ea8b211` (2026-09-24). 정적 읽기만. README는 "RoboTwin 2.0 (ICML 2026)". 열린 이슈 125 |
| 프로젝트 페이지 | https://robotwin-platform.github.io/ | 명시 없음 | - | - | 정적 HTML. 2.0 항목에 arXiv만 있고 학회 표기 없음 |
| 문서 | https://robotwin-platform.github.io/doc/ | 명시 없음 | - | - | 설치, 과제, 설정 안내 |
| 리더보드 | https://robotwin-platform.github.io/leaderboard | 명시 없음 | - | - | HTTP 200 확인. 내용은 읽지 않음 |
| 데이터셋 (HF) | https://huggingface.co/datasets/TianxingChen/RoboTwin2.0 | 미확인 | - | - | HTTP 200 확인. 다운로드 안 함. 10만+ 궤적 |
| 체크포인트 (HF) | 같은 저장소 `cotrain_ckpt` 경로 (README 언급) | 미확인 | - | - | 받지 않음 |
| RMBench | https://github.com/RoboTwin-Platform/RMBench | MIT | 233 | 2026-09-20 | RoboTwin 2.0 기반 기억 의존 조작 벤치마크 (2026-03-03 공개) |
| XPolicyLab | https://github.com/XPolicyLab/XPolicyLab | Apache-2.0 | 602 | 2026-11-21 | RoboTwin 2.0과 RoboDojo가 공유하는 정책 서빙·평가 스택 (코드에 서브모듈로 포함) |
| RoboDojo | https://github.com/RoboDojo-Benchmark/RoboDojo | MIT | 945 | 2026-11-19 | 공식 저장소 (RoboTwin README에서 언급) |
| 1.0 / 초판 | https://arxiv.org/abs/2504.13059 | - | - | - | CVPR 2025 Highlight. 2024 초판 arXiv:2409.02920 |
| 챌린지 보고서 | https://arxiv.org/abs/2506.23351 | - | - | - | CVPR 2025 MEIS 워크숍 |

- 공식 코드는 **MIT**이다. 코드 자체를 재배포해도 되지만, 이 분석 폴더에는 얕은 clone을 `.gitignore`로 제외했다.
- 에셋 (객체 zip, 배경 질감 zip, 임베디먼트)은 저장소에 없고 `scripts/_download_assets.sh`로 받는다. 이 분석에서는 받지 않았다.

## 5. 버전 관리 메모

- arXiv: v1 (2025-06-22), v2 (2025-08-27). 분석 대상은 v2다. v1과의 차이는 `01` §6.
- 학회 판본 (ICML 2026, PMLR v306 pp. 13673–13699): 이 분석에서 내용을 읽지 않았다. arXiv v2와 같은지 확인하지 않았다.
- 공식 저장소 README는 "ICML 2026"이라고 하지만 프로젝트 페이지의 2.0 항목에는 학회 표기가 없다. 학회 확인은 PMLR과 icml.cc 쪽에서 했다 (`README` 참고).

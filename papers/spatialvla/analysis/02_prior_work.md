# 02. 선행연구, 공식 저장소, 관련 분석

## 1. 공식 코드, 프로젝트 페이지, 관련 저장소

| 저장소 또는 자료 | 관계 | 라이선스 (확인 방법) | 기준 시점 | 상태 |
|---|---|---|---|---|
| [SpatialVLA/SpatialVLA](https://github.com/SpatialVLA/SpatialVLA) | 공식 구현 (HF 기반 모델, 학습 스크립트, 데이터 혼합) | GitHub API `license: null`. LICENSE 파일 없음. `pyproject.toml` 분류자는 "Apache Software License" | 기본 브랜치 main, commit `18fd74b2a633`, 마지막 push 2025-06-23, 별 738 | 얕은 복제 (`repo/`, 약 10 MB), 정적 읽기만 |
| SpatialVLA/SpatialVLA 브랜치 `lerobot` | README에 "개발 중"이라고 적힌 개선판 (LeRobot 데이터 로더, 다중 시점과 상태 입력) | 같은 저장소 | 브랜치 이름만 확인 | 복제하지 않음 |
| [DelinQu/SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA) | 공식 README가 지정한 SimplerEnv 평가 코드 (Google Robot, WidowX) | MIT (API) | 2025-06-23 push, 별 270 | 복제하지 않음 |
| [SpatialVLA/dlimp_custom](https://github.com/SpatialVLA/dlimp_custom) | `dlimp`의 시드 고정 수정판 (requirements에서 git으로 설치) | `null` (API) | 2025-03-16 push, 별 1 | 복제하지 않음 |
| SpatialVLA/rh20t (README 링크) | RH20T RLDS 변환 | 저장소 없음 (GitHub API 404) | — | README 링크 깨짐 |
| [IPEC-COMMUNITY/spatialvla-4b-224-pt](https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt) | 사전학습 체크포인트 (OXE+RH20T, 논문 제로샷 표의 기반) | HF 태그 `mit` (메타데이터) | 마지막 수정 2025-03-16, 다운로드 1,005회 (조회 시점) | **받지 않음.** safetensors 두 조각 합계 약 **8.1 GB** (메타데이터 합산) |
| IPEC-COMMUNITY/spatialvla-4b-mix-224-pt | Fractal과 Bridge 혼합 미세조정본 | HF 태그 `mit` | 2025-03-16 | 받지 않음 |
| [spatialvla.github.io](https://spatialvla.github.io/) | 프로젝트 페이지 (영상, BibTeX, 링크) | — | 200 응답, 제목 일치 | 학회 표기 없음 |

- **학회 확인.** 공식 RSS 2025 목록(`roboticsproceedings.org/rss21`)에서 p011로 찾았습니다. 저장소 README 제목에도 "RSS 2025"가 있습니다.
- **라이선스 불명확.** 저장소는 라이선스 파일이 없고, 체크포인트는 MIT입니다. 코드를 내 저장소에 복사하기 전에 확인해야 합니다. 이 저장소에는 복사하지 않았습니다.

## 2. 선행연구 계보

논문이 자기 자리를 잡는 방식은 세 갈래입니다(§II, 참고문헌).

**(a) 범용 정책과 교차 로봇 학습.** RT-1 [6]과 RT-2 [7]가 자기회귀 토큰 행동을 확립했습니다. Octo [48]은 OXE를 한 모델로 묶었고, RDT [39]는 확산 기반 1.2B 모델입니다. HPT [65]는 로봇별 스템 모듈로 이질성을 흡수합니다. SpatialVLA는 이 계열에 속하면서도 **관측과 행동 모두에 공간 표현**을 넣는다고 주장합니다.

**(b) VLM 기반 VLA와 행동 이산화.** RT-2 [7]는 행동을 256 bin으로 나눕니다. OpenVLA [30]는 같은 방식을 Prismatic VLM에 씁니다. π0 [5]는 흐름 매칭 행동 전문가를 붙입니다. CogACT [31], TraceVLA [71], RoboVLM [32]은 OpenVLA 계열을 확장합니다. SpatialVLA의 **행동 격자**는 256 bin 균일 이산화에 대한 직접 대안이고, 논문 §III-A와 표 IV가 그 비교입니다. 이산화 방식의 다른 대안으로는 FAST (arXiv:2501.09747)의 주파수 영역 토큰화가 있습니다.

**(c) 3D를 VLM에 넣기.** 3D-LLM [24], LLaVA-3D [10], LEO [25], 3D-VLA [69]는 3D 특징을 VLM에 넣습니다. SpatialVLA의 차이는 **깊이를 단안 추정으로 얻고 카메라 보정 없이 에고센트릭 좌표를 만든다**는 점입니다. 깊이 추정 모델은 ZoeDepth [4]입니다. 3D 정책 쪽에서는 DP3 (arXiv:2403.03954)가 점군 입력을 씁니다.

## 4. 같은 논문을 다루는 다른 자료

- 원 논문의 참고문헌 가운데 SpatialVLA의 설계와 직접 맞닿은 것: ZoeDepth [4] (깊이 추정), SigLIP [68] (시각 인코더), PaliGemma 2 [62] (백본), Bridge V2 [64] (WidowX 평가 기준), SimplerEnv [35] (시뮬 평가 기준), LIBERO [36] (미세조정 기준).
- SimplerEnv와 LIBERO는 평가 기준이 공개되어 있어 재계산과 재현 모두에서 기준으로 쓸 수 있습니다.

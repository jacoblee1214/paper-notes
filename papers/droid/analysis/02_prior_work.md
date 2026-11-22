# 02. 선행연구와 관련 저장소

이 문서는 DROID가 어떤 데이터셋 계보에 있는지, 공식 코드와 데이터가 어디에 있는지, 이 저장소의 다른 분석과 어떻게 이어지는지 정리합니다. 번호는 논문 참고문헌 번호(v2)입니다.

## 1. 데이터셋 계보 (표 I과 서론 II절)

| 데이터셋 | 논문 번호 | 규모 (궤적 / 장면) | DROID와의 관계 |
|---|---|---|---|
| RT-1 | [2] | 130k / 2 | 단일 기관 데이터. 표 I 비교 대상 |
| BridgeData V2 | [59] | 60.1k / 24 | 장면 수와 동사 꼬리가 가장 가까운 비교 대상 (IV-a) |
| RH20T | [14] | 13k / 7 | 33개 과제, 카메라 보정과 로봇 공개 |
| RoboSet | [1] | 98.5k / 11 | 인간 30%, 스크립트 70% |
| RoboNet | [8] | 162k / 10 | 스크립트 수집 (여러 로봇) |
| DobbE | [47] | 5.6k / 216 | 도구 기반(비로봇) 수집. 장면 수가 DROID 다음으로 많음 |
| Open X-Embodiment (OXE) | [39] | 1.4M / 311 | 기존 데이터셋의 집계. DROID 정책 실험의 비교군 |

- **차별점 주장:** DROID는 장면 수와 시점 다양성에서 앞선다고 씁니다. 집계 데이터셋인 OXE와 비교하면 장면 수 차이는 약 1.8배입니다(`01` §3, `03` K5).
- **수집 방식:** DROID와 같은 "사람 원격 조작" 계열은 RT-1, Bridge V2, RH20T 등입니다. 도구 기반 수집(DobbE 등 집게형 도구)은 손목 시점에 한정되고 형태 차이가 있다고 논문이 비판합니다(서론 II-b).
- **정책 학습 계보:** 확산 정책 [7], Robomimic [37], Octo [38]. Octo의 OXE 큐레이션 분할을 비교군으로 그대로 씁니다(각주 3).

## 2. 코드와 데이터 (GitHub 표와 데이터 위치)

### 2.1 공식 저장소

| 저장소 | 관계 | 라이선스 | 최신 커밋 (확인일 2026-11-22) | 기타 | 상태 |
|---|---|---|---|---|---|
| [droid-dataset/droid](https://github.com/droid-dataset/droid) | 공식 수집 스택: 로봇 제어(Polymetis), GUI, 보정, 후처리, 평가 GUI | **없음.** LICENSE 파일 없음, pyproject와 setup.py에 라이선스 필드 없음, GitHub API `license: null` | `33ae6a6` (2025-09-15, 기본 브랜치 main) | 별 474, 생성 2024-03-14 | 얕은 복제 `repo/droid_main/`. 정적 읽기만 |
| [droid-dataset/droid_policy_learning](https://github.com/droid-dataset/droid_policy_learning) | 공식 정책 학습과 평가: RLDS 데이터 로더, Robomimic 기반 확산 정책 | MIT (LICENSE 파일 있음) | `9a29c83` (2025-04-22, 기본 브랜치 master) | 별 307, 생성 2024-03-14 | 얕은 복제 `repo/policy_learning/`. 정적 읽기만 |
| [프로젝트 페이지](https://droid-dataset.github.io) | 논문, 데이터 시각화 도구, 문서 링크 | — | 확인 안 함 | 문서는 `droid-dataset.github.io/droid` | 웹 페이지 HTML만 확인 |

- 두 저장소 모두 얕은 복제(`--depth 1`)로 받았습니다. 실행, 설치, import는 하지 않았습니다.
- 프로젝트 페이지 HTML의 링크는 위 두 저장소, Colab 노트북, HF 데이터 페이지(`KarlP/droid`), 그리고 무관한 템플릿 링크(nerfies, eureka)입니다.

### 2.2 데이터와 가중치 (받지 않음)

| 항목 | 위치 | 크기 | 라이선스 | 상태 |
|---|---|---|---|---|
| 전체 RLDS | `gs://gresearch/robotics/droid` | 1.7 TB (정책 학습 README) | 논문은 CC BY 4.0 (본문 I, III) | 받지 않음. 메타데이터만 |
| 예제 RLDS 100 궤적 | `gs://gresearch/robotics/droid_100` | 약 2 GB (README) | 같음 | 받지 않음 |
| 원시 스테레오 HD | `gs://gresearch/robotics/droid_raw` | 확인 안 함 | 확인 안 함 | 받지 않음 |
| 보정 행렬 (36k 장면 등), 언어 주석 | HF `KarlP/droid` | 확인 안 함 | 확인 안 함 | 받지 않음 |
| 정책 체크포인트 | 논문과 저장소에 공개 여부 확인 안 함 | — | — | 받지 않음 |

- 정책 학습 README는 "DROID 전체는 1.7 TB"라고 씁니다. 이 크기는 RLDS 한 벌의 것입니다.
- 공식 저장소의 학습 설정 기본값 자체는 샘플 가중치 `[1]`(DROID 한 개)입니다. 논문의 50/50 혼합은 설정 파일에 없습니다(`04` D7).

### 2.3 의존성

- `policy_learning`은 Octo를 **커밋 `85b83fc19657ab407a7f56558a5384ae56fe453b`**에 고정해 데이터 로딩에 씁니다(README 설치 3번). Octo 분석 폴더와 연결됩니다(§3).
- 설치는 Python 3.10 conda 환경을 전제합니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- 확산 정책 [7]: 16 스텝 예측, 8 스텝 실행, DDIM. DROID 정책은 이 구성을 그대로 씁니다.
- ImageNet 사전학습 ResNet-50 [11]과 고정된 DistilBERT [45]: 인코더 선택은 이 논문의 설정이며 다른 선택지와 비교하지 않았습니다.
- 공동 학습 효과의 선행 근거: OXE [39]와 Octo [38]. DROID는 이 흐름에서 장면 다양성을 더 키운 데이터셋이라는 위치에 있습니다.

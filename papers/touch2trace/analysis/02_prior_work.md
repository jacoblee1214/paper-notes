# 02. 선행연구와 관련 저장소

## 1. 선행연구 계보 (구성 요소별 출처)

| 구성 요소 | 논문이 기대는 출처 | 이 논문에서의 위치 |
|---|---|---|
| 촉각 자기지도 사전학습 | Sparsh [2] (비전 기반 촉각), T3 [10] (여러 카메라 촉각 센서), 자기 피부 사전학습 [11], T-DEX [1] (로봇 놀이 데이터로 촉각 표현), Touch in the Wild [19] | 얼린 ViT-MAE 촉각 인코더(§4.1). 논문은 "압전 어레이용 사전학습"을 자기 몫으로 두지만 구조는 He et al. MAE [31]를 32×32 한 채널 이미지에 그대로 적용 |
| 촉각 모방 학습 파이프라인 | T-DEX [1], See-to-Touch/TAVI [4], MimicTouch [14], PolyTouch [3], DIH-Tele [15], 시각촉각 정책 [16–19], ManiFeel [20] | 모두 삽입, 조립, 강체 위주라고 정리(§2). 촉각을 한 덩어리 입력으로 다룬다는 평가 |
| 케이블, DLO 추적 | She et al. [5] (평행 집게 + GelSight, 케이블 자세와 마찰 피드백), Pecyna et al. [22] (시각촉각 RL), Sun et al. [23] (촉각 모터 sim-to-real), Jin et al. [24] (양팔 케이블 라우팅), 동시기 ViTac-Tracing [25] | 평행 집게와 명시적 상태 추정이 주류라는 대비(§2) |
| 다지 손 DLO | Yu et al. [8] (촉각 기반 3차원 자세 추정으로 따라가기), DexDLO [7] (시뮬레이션 학습), Sun et al. [6] (22개 프리미티브 분류체계, 재생) | 이 논문은 학습된 종단 정책을 실제 손에서 보임(§2) |
| 센서 | FlexiTac [27] (12×32, 2 mm 피치), GelSight [21], DIGIT [26] | TacV5(32×32, 0.7 × 0.92 mm)가 6.2배 조밀(§3.1). 재계산은 `03` V2 |
| 텔레오퍼레이션 | Manus 글러브 [30], AnyTeleop [29], 기하 리타기팅 [28] | 규칙 기반 직접 매핑(§3.2). 일반 리타기팅이 열린 문제라는 인용 |
| 정책 구조 | robomimic의 GPT + GMM [33], Diffusion Policy [34] | 본 정책은 robomimic 구성 요소를 사용(§4.2). DP는 비교 대상(표 4) |
| 촉각 시뮬레이션 | TacSL [13], Factory [12] | 사전학습 데이터의 시뮬 약 200만 프레임에 "TacSL에서 영감을 받은 Kelvin-Voigt 시뮬레이터"(§4.1) |
| 데이터 수집 도구 | UMI [32] | 사전학습용 실제 프레임 약 3만 개를 TacV5를 단 UMI 그리퍼로 수집(§4.1, 그림 6a) |
| 부드러움 지표 | Balasubramanian et al. [35] | 부록 F의 ISJ 지표 |

가까운 대안(논문에 비교가 없는 것): 사전학습 없이 손수 만든 촉각 요약만 쓰는 방식(표 5의 16차원이 이 범주의 한 점), 접촉 상태(미끄러짐 여부)를 분류해 제어에 쓰는 반사형 제어, 시각 촉각 융합 정책(부록 D가 한 점을 제공). 또한 케이블 상태 추정과 모델 기반 제어를 쓰는 접근은 비교하지 않습니다.

## 2. 코드와 저장소 (GitHub 표)

기준일 2026-11-26. 스타 수와 마지막 push는 GitHub API 값입니다. **공식 코드는 공개되어 있지 않습니다.**

| 저장소 | 관계 | 라이선스 | 스타 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [grmmtt/corl26-touch2trace](https://github.com/grmmtt/corl26-touch2trace) | **프로젝트 웹사이트 저장소**(코드 아님). HTML, CSS, 이미지, 영상 4개(약 12 MB) | LICENSE 파일 없음, API license: null | 0 | 2026-11-01 | 커밋 `c1dbf1eb136bbfe3f0167cabbf7c04aa7e4532d9`("Update Touch2Trace comparisons and project resources"). `index.html`과 README만 읽음, 복제하지 않음 |
| 프로젝트 페이지 [grmmtt.github.io/corl26-touch2trace](https://grmmtt.github.io/corl26-touch2trace/) | 논문, OpenReview PDF, 요약 영상, 표 2 재게시. **Code와 Dataset은 "coming soon"** | - | - | - | 저장소 README가 "Code and Dataset remain marked as coming soon"으로 적음 |
| [tjian-ai/corl26-slipsense](https://github.com/tjian-ai/corl26-slipsense) | 같은 저자 그룹의 CoRL 2026 포스터 SlipSense(다중 모달 미끄럼 검출) 프로젝트 사이트. 프로젝트 페이지가 "Related Project"로 연결 | license: null | 0 | 2026-10-01 | 읽지 않음 |
| 논문의 OpenReview PDF `openreview.net/pdf?id=ALlLaCTHKQ` | 학회 제출본 링크(프로젝트 페이지) | - | - | - | 읽지 않음(arXiv v1 사용) |
| [FlexiTac/FlexiTac](https://github.com/FlexiTac/FlexiTac) | 논문이 비교한 저해상도 압전 촉각 센서 [27] | license: null | 192 | 2026-11-18 | 참고 |
| [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) | GPT, GMM 구성 요소의 출처 [33] | MIT | 1,611 | 2026-08-09 | 참고 |
| [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | 비교 정책 DP [34] | MIT | 4,747 | 2024-12-24 | 참고 |
| [real-stanford/universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface) | UMI 그리퍼 [32] | MIT | 1,647 | 2026-05-26 | 참고 |
| [facebookresearch/mae](https://github.com/facebookresearch/mae) | MAE 사전학습 [31] | CC BY-NC 4.0(API: NOASSERTION) | 8,362 | 2024-07-23 | 참고. 논문 인코더가 이 코드를 썼는지는 서술 없음 |
| [facebookresearch/sparsh](https://github.com/facebookresearch/sparsh) | Sparsh [2] | CC BY-NC 4.0(API: NOASSERTION) | 225 | 2025-02-27 | 참고 |
| [alanzjl/t3](https://github.com/alanzjl/t3) | T3 [10] | MIT | 64 | 2024-06-21 | 참고 |
| [tesollodelto/dg5f_ros2](https://github.com/tesollodelto/dg5f_ros2) | DG-5F ROS 2 드라이버(제조사 조직 계정) | BSD-3-Clause | 2 | 2026-10-15 | 하드웨어 접근 참고. 8 자유도 리타기팅은 논문이 설명하지만 코드는 없음 |

- 사전학습 데이터, 시연 데이터, 체크포인트, 시뮬레이터(Kelvin-Voigt), TacV5 설계 파일이 공개되었는지 확인할 수 있는 링크는 논문과 프로젝트 페이지에 없습니다(2026-11-26).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- "첫 체계적 실물 특성화"(초록, 기여 2)와 "첫 실물 종단 학습 손 안 케이블 추적"(§1)은 §2의 인용 범위와의 비교에 기댑니다. 이 분석은 인용된 범위를 넘어선 문헌 검색을 하지 않았고, 이 분석이 확인할 수 있는 것은 **같은 플랫폼에서 네 축을 모두 바꿔 본 논문이 인용 목록에 없다**는 점까지입니다.
- 동시기 연구 ViTac-Tracing [25](arXiv:2603.18784, ICRA 2026)는 집게 기반이고 한쪽 끝이 고정된 설정이라고 논문이 구분합니다. 이 분석은 [25]를 읽지 않았습니다.

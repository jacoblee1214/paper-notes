# 02. 선행연구와 공식 저장소

## 1. 계보: 이 논문이 기대는 것

| 구성 요소 | 출처 (논문 참고문헌 번호) | OpenHLM에서의 역할 |
|---|---|---|
| VLA 백본 π0.5 | Physical Intelligence, π0.5 [22], π0 [15], PaliGemma [23] | 행동 전문가를 34-D로 키워 초기화 (§3.2, 부록 C.2) |
| 흐름 매칭 행동 생성 | Lipman 외 [24] | 10단계 적분 (§3.2) |
| 저수준 전신 추종 | SONIC [13], 범용 모션 트래커 | 관절 기반 명령을 추종 (§3.1) |
| 리타게팅 | GMR [20] | 사람 모션 → G1 관절 (§3.1) |
| 원격조작 장비 | PICO4U [19] | 5점(HMD, 컨트롤러 2, 다리 추적기 2) 모션 캡처 (부록 B.2) |
| 분리 제어 기준선 | GR00T N1.5/N1.6 [7, 18], AMO [6], Ψ0 [8] | 비교 대상 (§3.1, §4) |
| VR 3점 | OmniH2O, SONIC [12, 13] | 인터페이스 비교 (§3.1) |
| 로봇 없는 수집 | UMI [16], HuMI [1], 추적기 [26] | 공동 학습 데이터 (§3.3, 부록 B.3) |
| 1단계 생성 | Drifting model [25] | 절제 (§3.2, 부록 C.2) |
| 동시 연구 | EgoHumanoid [17], TWIST2 [9] | 관련 연구 (§5) |

이 논문의 새로움은 새 알고리즘이 아니라 "설계 선택의 효과를 실로봇에서 한 번에 하나씩 비교해 정리한 것"과 그 결과물의 공개입니다. §3의 각 비교는 선행 연구가 각각 쓴 선택지(분리 제어, 3점, SMPL, π0.5 대 PaliGemma, 1단계 생성)를 같은 하드웨어에 놓은 것입니다.

### 1.1 가까운 대안과 차이

| 대안 | 하는 일 | OpenHLM과의 차이 |
|---|---|---|
| SONIC [13] | 대규모 모션 트래커. VLA는 몇 개 실험으로만 붙임 | OpenHLM은 SONIC을 저수준 제어기로 고정하고 VLA의 행동 공간을 관절 목표로 직접 예측 (§5) |
| TWIST2 [9] | 전신 원격조작 수집, 단일 과제 시각운동 정책 | 다중 과제 언어 조건 VLA로 확장 (§5) |
| GR00T N1.6 [7], Ψ0 [8] | 휴머노이드 데이터를 사전학습에 포함, 분리 제어 | 전신 관절 직접 예측 + 비휴머노이드 로봇 사전학습 (§4) |
| EgoHumanoid [17] | 사람-휴머노이드 동일 과제 공동 학습 | OpenHLM은 새 물체와 지시문 확장이 목적 (§5) |

## 2. 공식 저장소와 자료

| 저장소 | 관계 | 라이선스 | 별 / 마지막 푸시 | 비고 |
|---|---|---|---|---|
| [OpenHLM-project/OpenHLM](https://github.com/OpenHLM-project/OpenHLM) | **공식 코드** (로봇 쪽, HuMI 쪽, VLA 학습·추론 세 모듈) | Apache-2.0 (저장소 LICENSE, NOTICE에 제3자 고지) | 117 / 2026-07-10 | 커밋 `4a8d4cd0ecd8313bb8dbb10d2afac2c2ec9ed1b4`(2026-07-10), 얕은 clone 약 101 MB, 정적 읽기만. 이슈 3건 열림 |
| [프로젝트 페이지](https://openhlm-project.github.io/) | 논문과 같은 주소 | 저장소 [OpenHLM-project.github.io](https://github.com/OpenHLM-project/OpenHLM-project.github.io) 라이선스 없음 | 0 / 2026-10-23 | 12개 과제 영상, 비교표 |
| [HF 체크포인트 OpenHLM/OpenHLM-ckpts](https://huggingface.co/OpenHLM/OpenHLM-ckpts) | 공식 가중치 (6개 디렉터리) | 모델 카드 라이선스 표기 MIT | 좋아요 3 / 2026-10-06 | 합계 약 44.8 GB(API 크기), 받지 않음 |
| [HF 데이터 OpenHLM/OpenHLM-data](https://huggingface.co/datasets/OpenHLM/OpenHLM-data) | 공식 시연 데이터 (LeRobot 형식) | 라이선스 표기 없음 | 다운로드 2,102 / 2026-06-30 | 합계 약 297.5 GB(API 크기), 받지 않음 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 학습 코드의 모체(`openpi4OpenHLM`은 OpenHLM 저장소 안의 수정본) | Apache-2.0 | 14,748 / 2026-08-24 | 받지 않음 |
| [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) | SONIC 제어 스택의 모체(`GR00T-WholeBodyControl4OpenHLM`) | API 표시 NOASSERTION (이중 라이선스: 코드 Apache-2.0, 모델 가중치 NVIDIA Open Model License) | 3,944 / 2026-11-03 | 받지 않음 |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | 기준선 GR00T N1.6 | Apache-2.0 | 8,416 / 2026-11-17 | 받지 않음 |
| [YanjieZe/GMR](https://github.com/YanjieZe/GMR) | 실시간 리타게팅 | MIT | 2,845 / 2026-10-22 | 받지 않음 |
| [real-stanford/universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface) | UMI | MIT | 1,647 / 2026-05-26 | 받지 않음 |
| [amazon-far/TWIST2](https://github.com/amazon-far/TWIST2) | 비교 텔레옵 | MIT | 903 / 2026-09-23 | 받지 않음 |

별 수와 푸시 날짜는 2026-11-25에 GitHub API로 읽었습니다. Ψ0의 공식 저장소는 이번 분석에서 찾아보지 않았습니다.

메모: README의 프로젝트 배지는 `openhlm-corl.github.io`를, 논문 PDF와 HF 모델 카드는 `openhlm-project.github.io`를 가리킵니다. 같은 내용을 가리키는지는 확인하지 않았습니다. 공개 체크포인트 README는 다섯 개를 나열하지만 HF 저장소에는 여섯 번째 `8tasks_8full-teleop`(8과제 기준 조건에 해당)이 더 있습니다.

## 4. 겹치는 주장과 구분점

- "비휴머노이드 로봇 사전학습이 휴머노이드로 전이된다"(§3.2): 부록 C.2는 π0.5의 로봇 데이터가 "주로 정적·바퀴 달린 양팔"이라고 적고, 그림 4의 근거는 π0.5 91, PaliGemma 60, 처음부터 42입니다. 사전학습 데이터의 정확한 구성은 이 논문이 아니라 π0.5 논문에 있습니다.
- "휴머노이드 데이터를 사전학습에 넣는 것만으로는 부족하다"(§4): 비교는 백본, 학습 스텝, 저수준 제어기, 원격조작 방식이 모두 다른 시스템 간 비교입니다. 같은 π0.5 백본에서 휴머노이드 사전학습 유무만 바꾼 실험은 없습니다.
- "HuMI 공동 학습은 새 의미는 주지만 새 동작은 못 준다"(§3.3): 4개 보류 과제, 과제당 5회 롤아웃에 기댄 결론입니다. 동작을 못 얻은 사례는 Pouring 한 과제입니다(`03` V5).

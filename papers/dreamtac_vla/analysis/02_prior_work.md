# 02. 선행연구와 관련 저장소

이 문서는 DreamTacVLA의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드와 공개 자료가 무엇인지를 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| DreamTacVLA의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 행동 청크와 CVAE 트랜스포머 정책 | ACT [31] (Zhao 2023) | 정책 본체. 부록 B.1은 "ACT를 정책 백본으로 사용"이라 쓰고, 3절은 "공유 LLM 백본"이라 씀(`03` K4) |
| 시각 언어 인코더 | CLIP [26] | 3절. 공개 코드는 CLIP ViT-B/16을 쓰고(`04` D16), 부록 B.1은 ResNet-18 [33]이라 씀 |
| 촉각 세계 모델 | V-JEPA 2 [22] (Assran 2025) | EMA 교사와 마스크 잠재 예측(D절, 식 11)을 촉각 이미지 시퀀스에 적용. 동결 ViT-L(300M)에 어댑터를 붙임 |
| 대조 정렬 | InfoNCE(본문에 인용 없음) | 식 1. 촉각 대 시각 영역 정렬은 Binding Touch [16], TVL [17], Sparsh [18]의 표현 정렬 계열과 닿아 있으나 다른 점은 "투영된 센서 위치의 시각 토큰"을 양성으로 쓰는 점 |
| 이미지 기반 촉각 센서 | DIGIT [30], GelSight [43] | 실로봇 센서(DIGIT)와 시뮬 광학 모델(Taxim [29]) |
| 촉각 시뮬레이션 | TacEx [28], Taxim [29], Isaac Sim [27] | 시뮬 촉각 이미지 합성(A.1) |
| 전문가 시연 자동 생성 | cuRobo (본문 인용 없음, A.2에서 이름만 언급) | 충돌 인식 궤적 계획 기반 특권 전문가(표 3) |
| 행동 전 미래 예측으로 행동 조건화 | DreamVLA [23], WorldVLA [24] | 잠재 롤아웃으로 행동을 조건화하는 VLA. 촉각 영역에 옮겼다고 서술(2.3절) |
| 촉각 VLA | Tactile-VLA [4], OmniVTLA [5], MLA [14], FuSe [10] | 저차원 힘 신호 또는 압축 촉각을 쓴다고 구분(1절, 2.2절) |
| 반응형 촉각 정책 | Reactive Diffusion Policy [15] | 고해상도 촉각 + 빠른 수정 경로 |
| 촉각 예측 | ViTacFormer [25], ViTacGen [42] | 자기회귀 시각촉각 예측과 시각→촉각 생성. 행동 조건화 예측은 아님(2.3절, F.4) |
| 공간 구조를 넣는 VLA | SpatialVLA [11], PointVLA [12], Evo-0 [13], TraceVLA [38] | HSA와 같은 "공간 대응"이라는 문제 의식. 접촉 수준 기하는 다루지 않는다고 구분(F.2) |

**가까운 대안과의 차이.** 가장 가까운 것은 (1) 촉각을 VLA에 넣는 Tactile-VLA와 OmniVTLA, (2) 촉각 잠재 표현을 정책에 쓰는 Sparsh 계열, (3) 행동 조건 미래 예측을 정책에 넣는 DreamVLA입니다. DreamTacVLA가 내세우는 새로움은 **센서 투영에 의한 공간 정렬(HSA)** 과 **행동 조건 미래 촉각 잠재(Dream)** 의 결합이고, 두 요소 각각의 절제는 표 1에 있습니다. 다만 본체가 VLM이 아니라 ACT라는 점은 "VLA"라는 이름이 주는 기대와 다릅니다(`03` K4).

## 2. 코드와 저장소 (GitHub 표)

| 저장소 | 관계 | 라이선스 | 별 / 마지막 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|
| [michaelyeah7/learning-to-feel-the-future](https://github.com/michaelyeah7/learning-to-feel-the-future) | 공식 코드 (프로젝트 페이지의 Code 링크, README가 arXiv:2512.23864 명시) | LICENSE 파일은 MIT + Dobot 부분 MIT (GitHub API는 `NOASSERTION`). README는 V-JEPA 가중치가 CC BY-NC 4.0이라고 따로 적음 | 10 / 2026-05-16 | 커밋 `efd1f75ae169fc77273759df79590d81584f9566` (2026-05-16), 얕은 복제 97 MB | 정적 읽기만 (`04`). 실행, 설치, import 안 함 |
| [michaelyeah7/dobot_xtrainer_world_tac](https://github.com/michaelyeah7/dobot_xtrainer_world_tac) | 같은 저자 계정의 이전 개발 저장소로 보임 (파일 구성이 공식 코드와 겹침, 2025-11 이후 변경 없음) | MIT | 0 / 2025-11-04 | 복제 안 함 (파일 목록과 학습 스크립트 한 개만 API로 확인) | 보조 정보 |
| Hugging Face `michaelyeah7/dreamtacvla` | 정책 체크포인트 (과제 3개, 체크포인트 25개) | 모델 카드 머리말 apache-2.0 | — / 2026-05-16 | 31.2 GB (API 파일 크기 합) | **받지 않음** |
| Hugging Face 데이터셋 `michaelyeah7/dobot_xtrainer_tac_1107` | 실로봇 시연 HDF5 (usb 50, gearassemb 50, peginhole 52 에피소드) | 카드 머리말 apache-2.0 | — / 2026-05-16 | 3.9 GB (API 파일 크기 합) | **받지 않음** |
| [프로젝트 페이지](https://michaelyeah7.github.io/learning-to-feel-the-future/) | 논문, 코드, 모델, 데이터셋 링크, BibTeX | — | — | 2026-11-23 확인 | 비디오 칸은 "Drop video / GIF here" 자리표시자 상태 |
| [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | 세계 모델 기반 | MIT (가중치는 별도 약관) | 4913 / 2026-10-29 | — | 외부 의존 |
| [DH-Ng/TacEx](https://github.com/DH-Ng/TacEx) | 촉각 시뮬레이션 (논문 [28]) | API에 표기 없음 | 173 / 2026-11-11 | — | 외부 의존, 공식 코드에는 시뮬 코드 없음 |
| [NVlabs/curobo](https://github.com/NVlabs/curobo) | 시뮬 전문가 (부록 A.2) | Apache-2.0 | 2039 / 2026-11-21 | — | 외부 의존, 공식 코드에 없음 |
| [MarkFzp/act-plus-plus](https://github.com/MarkFzp/act-plus-plus) | 정책 본체의 출처 (`THIRD-PARTY-LICENSES`에 명시) | MIT | 3678 / 2024-05-15 | — | 외부 의존 |

- **공개 범위 요약:** 공식 코드는 **실로봇 학습·추론 코드**입니다. 시뮬레이션 데이터 생성(IsaacSim, TacEx, cuRobo), 세계 모델 사전학습, 기준선(π0, Diffusion Policy) 평가, Tool Stabilization 과제 설정은 저장소에서 찾지 못했습니다(`04` §1, D3, D12).
- **라이선스가 서로 다릅니다.** 코드 MIT, 체크포인트와 데이터 apache-2.0(카드 머리말), V-JEPA 가중치 CC BY-NC 4.0(README 주의 문구). 체크포인트가 V-JEPA 가중치를 포함하는지는 확인하지 않았습니다(받지 않음).
- 논문 초록은 "Code, data, and videos are available at" 프로젝트 페이지라고 씁니다. 코드와 데이터는 위 링크로 확인했고, 비디오는 프로젝트 페이지에서 자리표시자 상태였습니다(2026-11-23).

## 4. 참고

- 번호가 붙은 논문 참고문헌은 `paper/paper.txt`의 References를 따릅니다. 부록 F는 본문 2절의 확장입니다.

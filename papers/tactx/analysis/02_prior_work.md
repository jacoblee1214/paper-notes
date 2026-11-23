# 02. 선행연구와 관련 저장소

TACTX의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드와 데이터가 어디에 있는지 정리합니다. 번호는 논문 참고문헌 번호(논문 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| TACTX의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 센서 간 촉각 표현 정렬(비전 기반 센서 계열) | UniTouch [5], T3 [6], AnyTouch [7], Sparsh [8], Cross-sensor touch generation [9], 센서 불변 표현 [46], AnyTouch 2 [47], Touch-to-touch [48] | 저자들은 이들이 영상과 같은 공통 기판을 공유하는 센서 사이에 머문다고 봅니다(§1, §2). TACTX는 입력 변환 없이 세 가지 변환 방식에 걸쳐 정렬한다고 주장합니다 |
| 공통 입력 형식으로 바꾸는 방식 | UniForce [49], 센서 간 힘 학습 [50], UniTac-NV [51], TransForce [52] | 서로 다른 센서를 하나의 입력 형식으로 사상합니다. TACTX는 센서별 인코더를 쓰고 입력 변환이 없다는 점이 다릅니다(§2) |
| 짝 없이 분포를 맞추는 방식 | TactAlign [53] | 쌍 데이터 없이 정렬. TACTX는 쌍 데이터를 전제로 하며 약한 짝짓기·비짝 정렬을 향후 과제로 둡니다(§6) |
| 대조 학습(NT-Xent, InfoNCE) | SimCLR [54], CPC [19] | 정렬 손실의 기반. 구현은 SimCLR 저장소의 `NT_Xent`를 씁니다(`04` D1) |
| 변분 인코더·재구성·교차 재구성 | 교차 임베딩 계열 [41–45]: Cross-hand latent [41], Latent action diffusion [42], 잠재 공간 정렬 [43], CycleVAE [44], UniVLA [45] | 센서(또는 로봇)마다 인코더–디코더 쌍을 두고 공통 잠재로 매핑하는 설계. TACTX는 이를 촉각에 적용했습니다(§2) |
| 센서 | Daimon DM-Tac W [29](비전), eFlesh [16](자기), FlexiTac [17](저항) | 세 센서 모두 외부 센서입니다. 센서 벤치마크 TaCo [40]에는 저자 일부가 겹칩니다 |
| 하위 정책 | ACT [20], GELLO 원격조작 [55] | 정책은 ACT를 그대로 씁니다. 표현만 바꿉니다 |
| 사전학습 촉각 표현 | Sparsh [8], T3 [6], 촉각 피부 자기지도 [37, 38], UniT [39] | 센서별 대규모 자기지도 표현. TACTX는 처음부터 학습한 작은(16차원) 표현입니다 |

**가까운 대안과의 차이.**
- T3 [6], AnyTouch [7], UniTouch [5]는 비전 기반 촉각의 센서 간 표현을 다룹니다. TACTX와 달리 센서 방식이 다른 조합(자기, 저항)을 다루지 않고, 과제 정책 전이를 직접 평가하는 범위도 좁습니다.
- UniForce [49]와 UniTac-NV [51]는 서로 다른 센서를 공통 입력 형식으로 바꿔 힘이나 표현을 공유합니다. TACTX는 형식을 맞추는 대신 센서별 인코더와 쌍 감독을 씁니다.
- 가장 가까운 설계는 교차 로봇 잠재 정렬 [41–44]입니다. TACTX는 같은 틀을 촉각에 옮긴 것으로 읽을 수 있습니다.

## 2. 코드와 저장소 (GitHub 표)

확인일 2026-11-23 (GitHub API, Hugging Face API).

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|---|
| [tactx-project/TactX](https://github.com/tactx-project/TactX) | 공식 코드. 프로젝트 페이지 Code 링크. 표현 학습, 평가, 잠재 변환, ACT 정책 학습, 합성 데이터 테스트 | LICENSE 파일 없음 (GitHub API license: null). `policy_learning/.../third_party/detr`에 Apache-2.0 사본, SimCLR는 외부 의존(MIT) | 2 | 2026-10-21 | `9381ee0d1912695003945436bf3d2f4f8361e672` (단일 커밋 "Release TactX code") | 정적 읽기만 (`repo/`, 실행·설치·import 안 함) |
| [jnsungp/TactX](https://github.com/jnsungp/TactX) | 같은 코드의 저자 계정본. 설명 "Official codebase of TactX ... (CoRL 2026)". 브랜치 `main`, `refactor/tactx-release-cleanup`. 이력 30개 이상 커밋, 저장소 크기 약 134 MB | LICENSE 없음 | 0 | 2026-10-11 | 정리 전 기준 커밋 `f5d66f82` (정리 브랜치의 base, `maintenance/CLEANUP.md`) | 메타데이터만 확인, clone 안 함 |
| [TactX/TactX](https://huggingface.co/datasets/TactX/TactX) (Hugging Face 데이터셋) | 정책 시연 데이터: 센서 3 × 과제 4, 544 에피소드(훈련 482 / 검증 62), 총 약 130 GB | 카드에 cc-by-4.0 | - | 2026-10-17 | `val_split.json` | 받지 않음 (크기만 확인). **표현 학습용 쌍 데이터(궤적 2,670개)는 이 데이터셋에 없음** |
| [tactx-project/tactx-project.github.io](https://github.com/tactx-project/tactx-project.github.io) | 프로젝트 페이지 소스 | 없음 | 0 | 2026-10-23 | - | 읽기만 |
| [notvenky/eFlesh](https://github.com/notvenky/eFlesh) | 자기 센서 eFlesh [16] 하드웨어·펌웨어 | MIT | 449 | 2025-10-28 | - | 참고 |
| [FlexiTac/FlexiTac](https://github.com/FlexiTac/FlexiTac) | 저항 센서 FlexiTac [17] 하드웨어·펌웨어 | LICENSE 없음 | 190 | 2026-11-18 | - | 참고 |
| [Spijkervet/SimCLR](https://github.com/Spijkervet/SimCLR) | 공식 코드가 import하는 `NT_Xent`, `get_resnet` (커밋 `2738efae` 고정) | MIT | 824 | 2024-05-21 | `simclr/modules/nt_xent.py` 텍스트만 읽음 | 설치 안 함 |
| [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | 하위 정책의 기반 ACT [20] | MIT | 2,352 | 2024-07-23 | - | 참고 |

- 프로젝트 페이지의 링크는 Paper, arXiv, Code(`tactx-project/TactX`), Dataset(Hugging Face), TL;DR입니다.
- 사전 학습된 TACTX 인코더 체크포인트와 정책 체크포인트는 코드 저장소와 데이터셋 어디에도 없습니다(`04` D16).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- 교차 임베딩 정렬(센서별 인코더–디코더 + 교차 재구성)은 교차 로봇 학습에서 이미 쓰이는 틀입니다 [41–44]. 새로운 점은 촉각 세 방식에 대한 적용과 정책 전이 평가입니다.
- NT-Xent의 온도 0.01은 SimCLR 계열 기본값(0.5 또는 0.1)보다 훨씬 낮습니다. 배치 64의 쌍마다 음성 예시가 적을 때 낮은 온도는 양성과 가장 가까운 음성에 큰 가중을 줍니다. 이 설정의 민감도는 논문에 없습니다.

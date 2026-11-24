# 02. 선행연구와 관련 저장소

이 문서는 ControlTac의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드와 관련 저장소가 어떤 상태인지를 정리합니다. 번호는 v3 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| ControlTac의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 비전 기반 촉각 센서와 데이터 부족 | GelSight [2], GelSlim 3.0 [16], DIGIT [17], 9DTact [18], 촉각 로보틱스 전망 [1] | 센서 종류와 센서 간 외형 차이(젤, 조명)가 문제 설정의 출발점입니다(§1, 부록 A) |
| 촉각 시뮬레이션 | TACTO [6], Taxim [7], DiffTactile [14], Taccel [15], 물리 기반 렌더링 [21], ObjectFolder [19, 20] | 논문의 Simulator 기준선(Taxim, 수동 자세 제어와 z 변위→법선력 매핑, §4 Baselines)입니다. 시뮬-실제 격차를 이 논문의 동기로 듭니다(§1) |
| 시뮬→실제 생성 변환 | Jianu 외의 딥 텍스처 생성 네트워크 [56] (arXiv:2112.01807), Higuera의 점자 읽기 확산 모델 [26] | Sim2Real 기준선은 [56]의 생성 방식을 "적응"한 것입니다(§4 Baselines). [26]은 같은 계열의 확산 기반 시뮬→실제 변환으로 본문 §2에서 인용됩니다 |
| 교차 모달 촉각 생성 (Vis2Tac, Text2Tac) | Visuo-Tactile cross-modal [11], Touch and Go [9], TaRF [12], Touching a NeRF [13], Generating visual scenes from touch [27], TextToucher [8], Binding touch to everything [10] | 제어 신호가 없는 "자유 형식" 생성으로 분류하고 그림 2에서 비교합니다. 정량 비교는 TextToucher(Text2Tac)로 분류 한 곳뿐입니다(그림 6(a), 부록 E.6) |
| 교차 센서 생성 | Touch2Touch [28] | 같은 목표(센서 간 변환)를 다루지만 본 논문은 9DTact 한 센서 쌍만 미세조정으로 시험합니다(그림 5) |
| 힘 라벨과 영상 데이터 | FeelAnyForce [51] (CC BY 4.0, HF 데이터셋), AnyTouch2 [54] | 1단계 20,000장과 2단계 7,000장, 힘 추정 틀, 평가 데이터의 출처입니다(부록 B.1, B.4, B.6) |
| 확산 모델 구성 | DDPM [42], DDIM [43], DiT [81], SANA 오토인코더 [47], 잠재 확산 | 생성기와 샘플러(부록 B.1). 공식 코드의 변환기는 SANA 계열 선형 어텐션 구조입니다(`04` D1) |
| 구조적 조건 주입 | ControlNet [48, 49] | 2단계 자세 어댑터의 직접 출처입니다. 본문은 "ControlNet에서 영감을 받았다"고 적습니다(§3.2) |
| 하류 모델 | DINOv2 ViT [84], ViT [83, 85], ACT [62] | 힘·자세 추정기, 분류기, 모방학습 정책입니다(부록 B.4, D, E.12). 본문 §4.3은 ACT를 [62]로, 부록 E.12는 "Action Chunking Transformer"를 [88](ALOHA 2)로 인용합니다 |
| 학습 기반 자세 추정 | PCA 기반 케이블 조작 [60] (She 외), 학습 기반 추정기 [6] | 표 2의 PCA 기준선과 "라벨이 필요한 학습 기반 추정기" 문제 설정 |
| 촉각 표현 학습 (대조 대상) | T3 [22], Sparsh [23], SITR [24], AnyTouch [25] | 생성 데이터를 사전학습에 쓸 수 있다는 맥락으로만 인용합니다(§2). 정량 비교는 없습니다 |

**가까운 대안과의 차이.** 가장 가까운 것은 (a) Taxim 같은 예제 기반 시뮬레이터에 센서 영상을 맞추는 계열과 (b) 시뮬 영상을 생성 모델로 실제 영상에 옮기는 Sim2Real 계열입니다. ControlTac은 (1) 참조 영상 한 장에서 시작하고 (2) 힘과 접촉 마스크를 명시적 조건으로 두며 (3) 별도 시뮬레이터 없이 FeelAnyForce의 힘 라벨이 있는 실제 영상만으로 학습한다는 점이 다릅니다. 이 중 (3)은 학습 데이터가 같은 센서(GelSight Mini)에서 온 쌍 영상이라는 전제를 포함합니다. 새 센서에는 9DTact 실험처럼 소량 미세조정(2,000장)이 필요합니다(그림 5).

## 2. 코드와 저장소 (GitHub 표)

조회일 2026-11-24. 별 수와 마지막 푸시는 GitHub API 값입니다.

| 저장소 | 관계 | 라이선스 | 별 / 마지막 푸시 | 상태 |
|---|---|---|---|---|
| [DongyuLUO/ControlTac](https://github.com/DongyuLUO/ControlTac) | 공식 코드 (main: 학습과 추론, gh-pages: 프로젝트 사이트) | **없음** (API license: null, LICENSE 파일 없음. README와 THIRD_PARTY_NOTICES는 "출처 확인 후 라이선스 선택 필요"라고 적음) | 5 / 2026-09-16 | `32b005b` (2026-09-16)를 `repo/`에 얕게 복제해 정적 읽기만 함 |
| [프로젝트 페이지](https://dongyuluo.github.io/controltac/) | 공식 프로젝트 페이지 | - | - | HTML만 받음. 프로젝트 페이지 헤더에 "ICCV CDEL Workshop Oral"과 "CoRL 2026" |
| 릴리스 v0.1.0 (같은 저장소) | 체크포인트 `force_control.pth` 590,210,867 B, `force_pose_control.pth` 891,837,471 B, 주석 압축 `controltac_annotations.zip` 3,148,600 B | 저장소와 같음(없음) | 2026-09-15 게시 | 받지 않음. README는 승인된 GitHub 계정용 릴리스 자산으로 안내함. 크기는 릴리스 API 메타데이터 |
| [amirsh1376/FeelAnyForce (HF)](https://huggingface.co/datasets/amirsh1376/FeelAnyForce) | 학습 데이터 출처 | CC BY 4.0 (데이터셋 카드) | 마지막 수정 2025-03-24 | 약 82 GB 분할 압축. 받지 않음 |
| [NVlabs/Sana](https://github.com/NVlabs/Sana) | 오토인코더(DC-AE)와 변환기 모듈의 출처 | Apache-2.0 | 약 9.9k / 2026-11-24 | 읽지 않음. 공식 코드는 DC-AE 가중치를 `mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers`에서 받습니다 |
| [facebookresearch/DiT](https://github.com/facebookresearch/DiT) | 위치·시간 임베딩과 변환기 구조의 계보 | CC BY-NC 4.0 (저장소 LICENSE 텍스트, API는 NOASSERTION) | 약 8.7k / 2024-05-31 | 비상업 조건이 공식 코드의 "출처 확인" 문구와 관련됩니다(`04` D13) |
| [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) | 2단계 어댑터의 개념 출처 | Apache-2.0 | 약 34k / 2024-02-25 | 읽지 않음 |
| [Robo-Touch/Taxim](https://github.com/Robo-Touch/Taxim) | Simulator 기준선 | MIT | 127 / 2022-11-02 | 읽지 않음. 기준선 구현은 공개 저장소에 없음(`04` D11) |
| [facebookresearch/tacto](https://github.com/facebookresearch/tacto) | 시뮬레이터 비교 대상 | MIT | 471 / 2022-02-17 | 읽지 않음 |

공식 저장소 이름 확인: GitHub 검색에서 같은 이름의 다른 저장소들(`control-tracker/controltac-midia`, `ControlTac/ControlTac.github.io` 등)이 나오지만 논문과 관계가 없고, 공식 저장소는 프로젝트 페이지의 "Code" 링크(`github.com/DongyuLUO/ControlTac`)입니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- "촉각 영상의 외형 변화는 힘 라벨이 있어야 힘 추정에 쓸모 있다"는 전제는 FeelAnyForce [51]의 힘 추정 틀과 DINOv2 사전학습에 기댑니다(§3.3, 부록 D.2). ControlTac의 하류 힘 추정 성능은 이 틀이 이미 "거의 오라클"(§4.2)이라고 적은 전체 데이터 성능으로 상한이 정해집니다.
- 교차 센서 결과(그림 5)는 9DTact [18]의 한 센서 쌍이고, 센서 다양성 자체는 AnyTouch/T3 계열이 더 넓게 다룹니다.
- 모방학습 결과(그림 8)는 ACT와 xArm7 한 과제입니다. 선행 촉각 모방학습 연구와의 직접 비교는 없습니다.

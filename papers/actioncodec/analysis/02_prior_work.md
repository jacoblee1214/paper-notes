# 02. 선행 연구와 공식 코드

## 1. 계보: 행동을 토큰으로 바꾸는 방법

| 계열 | 방법 | 핵심 아이디어 | ActionCodec의 위치 |
|---|---|---|---|
| 균일 구간화(Binning) | RT-2, OpenVLA [13], MolmoAct [32], VLA-0 [14] | 차원마다 값을 N개 구간으로 나눠 토큰 한 개로 표현. VLA-0는 숫자를 문자열로 | 표 1(c), 표 2의 기준선. 길이가 T × D로 늘어 지연이 큼 |
| 문자열(String) | Actions as Language [10] | 행동 배열을 Python 리스트 문자열로 BPE | 같은 기준선 (토큰 779.2개) |
| 주파수 + BPE | FAST [15] | 행동 청크에 DCT를 하고 계수를 양자화한 뒤 BPE로 압축 | 가장 강한 기준선. 토큰 24.3 ± 9.1, OR 19% (표 2) |
| 학습형 VQ | VQ-VLA [16], MiniVLA [28], QueST [33], VQ-BeT [34] | CNN 또는 트랜스포머 인코더 + 벡터 양자화 | "기존 VQ는 재구성 충실도만 봤다"는 주장의 대상 |
| 신경 토크나이저 + 블록 단위 자기회귀 | FASTer [23] | RVQ 층마다 블록으로 묶어 병렬 예측 | BAR의 출처(부록 B.4). 같은 연구실의 선행 연구 |

- **VQ-VAE [18]와 RVQ [17].** 식 (1)과 RVQ 사후 학습의 기반입니다. RQ-VAE는 이미지 생성용입니다.
- **Perceiver [37].** 학습 가능한 질의가 입력에 교차 어텐션하는 구조입니다. 논문은 이 구조가 토큰 간 관계(독립, SA, 인과)를 바꿔 끼우기 쉬워서 골랐다고 씁니다(부록 B.2).
- **InfoNCE [38].** OR을 조절하는 실험용 손실입니다(부록 B.2.1). 최종 모델의 TCL(식 5)과 CLIP형 손실(식 6)은 인용이 없습니다. 식 (6)은 쌍마다 시그모이드를 쓰는 형태(SigLIP 계열)로 읽히지만 논문은 "CLIP-based"라고만 부릅니다.
- **VLA 패러다임.** PD는 OpenVLA-OFT [4]의 병렬 디코딩, KI는 π0.5 [7]와 knowledge insulating [8], BAR는 FASTer [23]에 대응합니다.
- **교차 임베딩 토크나이저.** 소프트 프롬프트는 로봇 ID별 학습 질의입니다(부록 C). 비슷한 "임베딩 조건 토크나이저"는 언어 모델의 어댑터 문헌에 있으나 논문은 따로 인용하지 않습니다.

## 2. 가까운 대안과 차이

| 대안 | 같은 점 | 다른 점 |
|---|---|---|
| FAST [15] | 한 초(20스텝) 창, 범용 토크나이저를 지향 | 학습이 필요 없고 토큰 수가 가변. 청크가 한 칸 밀리면 DCT 계수가 모두 바뀌어 OR이 낮음(19%). ActionCodec은 고정 길이 16 |
| VQ-VLA [16] | VQ 기반, 사전학습 가능 | 구조와 토큰 4개. 논문은 공개 체크포인트에 데이터 통계만 맞춰 비교(부록 B.3) |
| MiniVLA [28] | 잔차 VQ(VQ-BeT 계열) | 토큰 7개, 지평선 8 |
| FASTer [23] | 같은 연구실의 블록 단위 자기회귀 토크나이저 | BAR는 이 방법의 디코딩 방식을 ActionCodec의 RVQ에 적용 |
| 연속 행동 전문가(π0, π0.5) | 같은 VLM 계열 | 토큰이 아니라 flow matching. KI 변형이 이 구조를 따라 한 것 |

## 4. 코드와 모델 자원 (GitHub와 Hugging Face)

조회일 2026-11-26. 별과 갱신일은 GitHub API 값입니다.

| 자원 | 관계 | 라이선스 | 별 | 최근 푸시 | 비고 |
|---|---|---|---|---|---|
| [ZibinDong/actioncodec](https://github.com/ZibinDong/actioncodec) | 공식 코드(논문 저자 계정) | README는 MIT, LICENSE 파일 없음(GitHub API license: null) | 82 | 2026-03-01 | 단일 커밋 `f034eb9a9da6d6f37b9522b4548779e8a9886b21`(2026-03-01). 얕은 복제로 정적 읽기만 함 |
| [ZibinDong/ActionCodec-Base](https://huggingface.co/ZibinDong/ActionCodec-Base) | 토크나이저 가중치(VQ, 토큰 16) | 카드에 라이선스 표기 없음 | – | 2026-01-09 | 180.4 MB, 받지 않음 |
| [ZibinDong/ActionCodec-Base-RVQft](https://huggingface.co/ZibinDong/ActionCodec-Base-RVQft) | 토크나이저 가중치(RVQ, 토큰 48) | 표기 없음 | – | 2026-03-01 | 197.3 MB, 받지 않음 |
| SmolVLM2-2.2B-ActionCodec-{AR, PD, BAR}-LIBERO | 정책 체크포인트 3개 | 표기 없음 | – | 2026-02-28 ~ 03-01 | 4.7, 5.5, 5.6 GB. KI, 256M, 500M, Qwen/InternVL 변형은 없음 |
| [ZibinDong/actioncodec2](https://github.com/ZibinDong/actioncodec2) | 후속 연구 ActionCodec2: A Streaming Action Tokenizer의 코드 | Apache-2.0 | 28 | 2026-11-13 | 이 논문의 코드가 아님. 분석 범위 밖 |
| [ZibinDong/actioncodec2-page](https://github.com/ZibinDong/actioncodec2-page) | 후속 연구 프로젝트 페이지 | 표기 없음 | 2 | 2026-10-29 | 이 논문의 프로젝트 페이지가 아님 |
| [Stanford-ILIAD/openvla-mini](https://github.com/Stanford-ILIAD/openvla-mini) | MiniVLA 기준선 코드 | MIT | 384 | 2025-03-19 | 논문 각주 4 |
| [xiaoxiao0406/VQ-VLA](https://github.com/xiaoxiao0406/VQ-VLA) | VQ-VLA 기준선 코드 | MIT | 135 | 2025-11-15 | 논문 각주 5 |
| [physical-intelligence/fast](https://huggingface.co/physical-intelligence/fast) | 범용 FAST 토크나이저 | Apache-2.0 | – | 2025-01-16 | 논문 각주 6 |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | SO100 조립, 데이터셋 도구 | Apache-2.0 | 29,102 | 2026-11-26 | 논문 [36] |
| HF 데이터셋 `ZibinDong/actioncodec-so100-part1`, `ZibinDong/xarm_pickplace` | 논문 각주 1, 2의 실로봇 데이터 | 확인하지 않음 | – | – | 받지 않음 |

- **프로젝트 페이지.** arXiv 초록 페이지와 저장소 README에서 이 논문의 프로젝트 페이지를 찾지 못했습니다(2026-11-26 확인 기준).
- **arXiv 주석과 채택.** arXiv 주석란은 비어 있습니다. 채택 정보는 corl.org 공식 채택 목록(포스터, 세션 Poster post-p1, 11/9)을 읽어 둔 `conferences/corl2026/papers.json`으로 확인했습니다.
- 코드 대조 결과는 `04_code_verification.md`에 있습니다.

## 5. 구성 요소별 출처

| 구성 요소 | 출처 |
|---|---|
| VQ 목적(식 1) | VQ-VAE [18] |
| 잔차 양자화 | RQ-VAE [17] |
| 교차 어텐션 인코더/디코더 | Perceiver [37] |
| 겹침률 손실(InfoNCE) | CPC [38] |
| TCL, 시그모이드형 CLIP 손실 | 인용 없음 |
| 병렬 디코딩 | OpenVLA-OFT [4] |
| 지식 격리 | π0.5 [7], knowledge insulating [8] |
| 블록 단위 자기회귀 | FASTer [23] |
| 소프트 프롬프트(로봇별 질의) | 이 논문 |
| 겹침률 지표(식 7) | 이 논문 |

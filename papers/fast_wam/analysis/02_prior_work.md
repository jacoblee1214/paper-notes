# 02. 선행연구, 가까운 대안, 공식 저장소

## 1. 계보

| 흐름 | 대표 연구 (논문 참고문헌 번호) | 이 논문과의 관계 |
|---|---|---|
| VLA (정적 이미지-텍스트 사전학습) | OpenVLA [9], π0 [10], π0.5 [11], GR00T N1 [12], RT-2 [13], RDT [14], SmolVLA [16], Gemini Robotics [17], Galaxea g0 [18] | Fast-WAM은 테스트 인터페이스가 VLA와 같고, 학습에만 영상 목적함수를 더합니다 (§2) |
| 영상 생성 기반 정책 | UniPi 계열 [8] (Du 외 2023), GR-2 [24], Vidar [7], DreamGen [25], Genie Envisioner [33], Cosmos Policy [32] | 미래 영상을 만든 뒤 행동을 뽑는 흐름. Fast-WAM은 이 생성을 뺍니다 |
| 공동 영상-행동 모델 (Joint) | UWM [6], Motus [5], Ye 외 [4] (World Action Models are zero-shot policies), mimic-video [1], Video generators are robot policies [2] | Fast-WAM-Joint 변형의 원형입니다 |
| 영상 먼저, 그다음 행동 (IDM / 인과 WAM) | LingBot-VA [3] (Causal World Modeling), Vidar [7] | Fast-WAM-IDM 변형의 원형입니다. LingBot-VA는 RoboTwin 비교 기준입니다 |
| 영상 표현을 쓰되 테스트 때 영상 복원을 건너뜀 | VPP [34] (예측 영상 표현), UVA [35] (추론 때 영상 디코딩 생략) | 논문이 가장 가까운 대안으로 꼽는 연구입니다. 차이는 "학습 때 공동학습 효과와 테스트 때 상상 효과를 통제된 변형으로 분리했는가"입니다 (§1 기여 3) |
| 대형 영상-행동 공동 모델 | DreamZero [4] (영상 확산 14B 기반 공동 예측) | 중심 주장이 "영상-행동 공동 예측으로 일반화가 늘어난다"입니다. Fast-WAM은 이 공동 예측 자체가 테스트 시 필요한지 묻습니다 |

## 2. 구성 요소별 출처

| 구성 요소 | 출처 (논문 기준) | 확인 |
|---|---|---|
| 영상 백본과 VAE, T5 인코더 | Wan2.2-5B [36] (Wan 팀, 2025) | 논문 §4.1 |
| 흐름 정합 학습 | 식 (5)–(6)에 인용 없음 | 논문이 출처를 적지 않습니다 |
| logit-normal 시간 샘플링 | "Following [36]" (§4.1) | 코드는 다른 샘플링을 씀 (`04` D2) |
| IDM 변형의 잡음 증강 p=0.5 | [3] (LingBot-VA) | §4.1 |
| LIBERO 벤치마크 | [37] (Liu 외 2023) | §4.2 |
| RoboTwin 2.0 벤치마크 | [38] (Chen 외 2025), 학습 구성은 [5], [3] | §4.2 |
| 평가 프로토콜 (지시문 unseen) | Motus 방식 (코드 README) | 논문 미기재 |

## 4. 공식 코드, 프로젝트 페이지, 체크포인트, 데이터

| 항목 | 위치 | 라이선스 (확인값) | 상태 |
|---|---|---|---|
| 논문 | arXiv:2603.16666 (v1, v2) | CC BY 4.0 (abs 페이지 링크) | `paper/paper.pdf` (v2, 2.7 MB), `paper_v1.pdf` (2.8 MB). 25 MB 미만이라 gitignore하지 않음 |
| 프로젝트 페이지 | https://yuantianyuan01.github.io/FastWAM/ | 확인 안 함 | 상태 확인 (HTTP 200), 저장하지 않음 |
| 공식 코드 | github.com/yuantianyuan01/FastWAM | MIT (`LICENSE`). GitHub API는 "Other"로 표시 | 얕은 복제 `repo/` 커밋 `7faa711` (2026-08-20). 정적 읽기만 함. 별 1665, 포크 214, 마지막 push 2026-08-20 |
| 코드 동봉 RoboTwin | `third_party/RoboTwin/` (원 저장소 복사) | MIT (`third_party/RoboTwin/LICENSE`, 2025 Tianxing Chen) | 정적 읽기만 함 |
| 체크포인트 | huggingface.co/yuanty/fastwam (모델 메타데이터 확인) | 라이선스 필드 없음 (`license: None`) | 다운로드 안 함. 약 수 GB (미확인) |
| LIBERO 데이터 (전처리) | huggingface.co/datasets/yuanty/LIBERO-fastwam | cc-by-4.0 | 다운로드 안 함 |
| RoboTwin 데이터 (전처리) | huggingface.co/datasets/yuanty/robotwin2.0-fastwam | 확인 안 함 | 다운로드 안 함 |
| LIBERO 시뮬레이터 | github.com/Lifelong-Robot-Learning/LIBERO | 확인 안 함 | 설치 안 함 |
| RoboTwin 시뮬레이터 | github.com/RoboTwin-Platform/RoboTwin | 확인 안 함 | 설치 안 함 |

## 5. 학회 (venue) 확인

- **판정: NeurIPS 2026 (poster).** 근거 세 가지입니다.
  1. NeurIPS 2026 공식 virtual 사이트의 포스터 페이지 `neurips.cc/virtual/2026/poster/149922`: 제목, 저자 (Tianyuan Yuan, Zibin Dong, Yicheng Liu, Hang Zhao)가 일치하고, "Poster Wed, Dec 9, 2026 10:00 AM – 1:00 PM AEDT, Hall 1-4 #S2509", 메뉴에 "Main Conference"가 있습니다. 페이지에 "accepted"라는 문구는 없습니다.
  2. arXiv abs의 Comments와 Journal ref 필드는 비어 있습니다 (v2 기준).
  3. 프로젝트 페이지 BibTeX는 `journal={arXiv preprint}`로만 적습니다.
- 한계: 공식 accepted-paper 목록 전체를 확인하지는 못했습니다. 포스터 페이지의 존재가 채택을 가장 강하게 보여 주는 근거입니다. 발표 형식(oral, spotlight)은 확인하지 못했습니다.
- 권고: 포스터 날짜는 2026-12-09이므로, 이후 학회 버전(camera-ready)이 arXiv에 반영되는지 확인해야 합니다.

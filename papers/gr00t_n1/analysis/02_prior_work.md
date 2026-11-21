# GR00T N1 선행연구와 자료 (02)

## 1. 계보

GR00T N1은 세 줄기를 합친 연구다. 하나는 로봇 VLA 계보, 다른 하나는 사람 영상·잠재 행동 계보, 셋째는 시뮬레이션과 영상 생성으로 데이터를 늘리는 계보다.

- **VLA 계보.** RT-1·RT-2 (Brohan 외, 2022/2023) → OpenVLA (Kim 외, 2024) → π0 (Black 외, 2024, 흐름 매칭 행동 전문가를 MoE로 VLM에 붙임) → GR00T N1. GR00T N1은 MoE 대신 **교차 주의 한 가지**로 VLM과 행동 모듈을 잇는다고 대비해서 적었다 (§5 관련 연구).
- **사람 영상·잠재 행동.** VPT (Baker 외, 2022)의 역동역학 라벨링 → LAPA (Ye 외, 2025)의 VQ-VAE 잠재 행동 → GR00T N1의 "LAPA 임베디먼트". 사람 영상 7개 데이터셋(Ego4D 등)을 같은 잠재 공간에 넣는다.
- **합성 데이터.** MimicGen (Mandlekar 외, 2023) → DexMimicGen (Jiang 외, 2024) (시뮬 시연 증식). 영상 생성 쪽은 Cosmos (Agarwal 외, 2025)와 Wan2.1 (Wan Team, 2025)을 쓴다. 신경 궤적은 DreamGen과 같은 방향이지만, GR00T N1은 규모(827 h)를 키웠다고 적었다 (§5).
- **기준선.** BC-Transformer (RoboMimic, Mandlekar 외, 2021)와 Diffusion Policy (Chi 외, 2024).

## 2. 구성 요소별 출처

| 구성 요소 | 논문 출처 (본문 인용) | 비고 |
|---|---|---|
| 흐름 매칭, 시간 분포 Beta(1.5,1), s=0.999 | Black 외 2024 (π0) | 논문이 그대로 따른다고 적음 (§2.1) |
| 행동 청크, 임베디먼트별 인코더 | Zhao 외 2023 (ACT) | 청크 H=16 |
| 교차 주의 DiT, adaLN 시간 조건 | Peebles & Xie 2023 (DiT), Alayrac 외 2022 (Flamingo), Jiang 외 2023 (VIMA) | 자기 주의와 교차 주의를 번갈아 씀 |
| VLM Eagle-2 (SmolLM2, SigLIP-2) | Li 외 2025 (Eagle-2), Allal 외 2025, Tschannen 외 2025 | 12번째 층 표현 |
| 픽셀 셔플 64 토큰 | Shi 외 2016 | 224×224 입력 |
| 잠재 행동 VQ-VAE | Ye 외 2025 (LAPA) | "LAPA" 임베디먼트로 학습 |
| 역동역학 모델 (IDM) | Baker 외 2022 (VPT) | SigLIP-2 임베딩 + DiT, 흐름 매칭 |
| 신경 궤적 영상 생성 | Wan Team 2025, Agarwal 외 2025 (Cosmos) | Wan2.1-I2V-14B + LoRA |
| 시뮬 데이터 증식 | DexMimicGen (Jiang 외 2024), MimicGen (Mandlekar 외 2023) | 벤치마크 과제당 1,000회, 사전학습은 receptacle 쌍당 10,000회 |
| 시뮬 과제 | RoboCasa (Nasiriany 외 2024) | 24과제 |
| 데이터 형식 | LeRobot (Cadene 외 2024) | modality.json 확장 |
| 보조 검출 손실 | OWL-v2 (Minderer 외 2023) | 부록 F에만 있음 |

## 4. 공식 코드·자료 표

조회일 2026-11-21. GitHub API와 HF API로 확인했다.

| 구분 | 항목 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| 공식 코드 | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | Apache-2.0 | 8,402 | 2026-11-17 (기본 브랜치 N1.7) | N1 대응 태그 `n1-release` (2025-06-11, 커밋 `755876a`)를 얕게 복제해 `repo/`에서 정적으로만 읽었다. 기본 브랜치는 N1.7이라 N1 대조에 쓰지 않았다. |
| 프로젝트 페이지 | [developer.nvidia.com/isaac/gr00t](https://developer.nvidia.com/isaac/gr00t) | 페이지 안에 라이선스 문구를 찾지 못함 | - | - | Isaac GR00T 제품 페이지. GitHub와 HF(N1.7-3B) 링크를 건다. arXiv 초록의 "this https URL"이 여기를 가리킨다. |
| 모델 가중치 | [nvidia/GR00T-N1-2B](https://huggingface.co/nvidia/GR00T-N1-2B) | **NVIDIA One-Way Noncommercial License** (모델 카드) | - | 2025-09-02 (수정), 다운로드 517, 좋아요 359 | HF 태그에 라이선스 항목이 없고 카드 본문에 NC 라이선스 링크가 있다. 연구 목적 사용 가능, 상업 사용 불가로 읽힌다. 법률 판단은 아니다. |
| 시뮬 데이터셋 | [nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) | CC BY 4.0 (HF 태그) | - | 2026-03-05 (수정) | 시뮬 평가·학습 데이터. 크기는 확인하지 않았다 (다운로드하지 않음). |
| 신경 궤적 코드 | [NVIDIA/GR00T-Dreams](https://github.com/NVIDIA/GR00T-Dreams) | Apache-2.0 | 633 | 2025-10-24 | DreamGen 공식 코드. N1 신경 궤적 파이프라인과 같은 계열. |
| 데이터 형식 기반 (N1 부록 E.1) | [huggingface/lerobot](https://github.com/huggingface/lerobot) | Apache-2.0 | 29,006 | 2026-11-21 | N1 데이터 형식의 기반 |
| 시뮬 데이터 생성 | [NVlabs/dexmimicgen](https://github.com/NVlabs/dexmimicgen) | GitHub 자동 판별 실패 (NOASSERTION) | 301 | 2025-12-06 | 라이선스 파일을 직접 확인해야 한다 (미확인). |
| 시뮬 환경 | [robocasa/robocasa](https://github.com/robocasa/robocasa) | GitHub 자동 판별 실패 (NOASSERTION) | 1,880 | 2026-11-18 | 라이선스 파일 직접 확인 필요 (미확인). |
| 설치 도우미 | [nvidia-isaac/isaac-gr00t-ready](https://github.com/nvidia-isaac/isaac-gr00t-ready) | Apache-2.0 | 6 | 2026-11-10 | 프로젝트 페이지에서 링크됨. 별 6개라 신뢰도는 낮게 본다. |

### 4.1 복제한 코드 정보

- `repo/`: `n1-release` 태그, 커밋 `755876a9afdb`, 커밋 시각 2025-06-11 (UTC). 파일 128개. `.gitignore`에 넣었다.
- 실행하지 않았고, 설치하지 않았다. 정적 읽기만 했다 (`04_code_verification.md`).

### 4.2 라이선스 요약

| 대상 | 라이선스 | 이 분석의 공개 판단 |
|---|---|---|
| 논문 (arXiv) | CC BY 4.0 (abs 페이지 링크) | PDF의 "© 2025 NVIDIA. All rights reserved." 때문에 논문 그림은 공개하지 않음 |
| 코드 (Isaac-GR00T) | Apache-2.0 | 정적 대조 결과와 짧은 코드 인용만 씀 |
| 가중치 | NVIDIA One-Way Noncommercial | 가중치는 받지 않음. 공개본에도 가중치 재배포 없음 |
| 시뮬 데이터 | CC BY 4.0 | 받지 않음 |

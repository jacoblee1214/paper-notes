# 02. 선행연구, 관련 저장소

번호는 논문 참고문헌 번호(`paper/paper.txt`의 References, v2)입니다. arXiv 번호와 날짜는 2026-11-26에 arXiv API로 확인했습니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| NAC의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 합성곱 인코더·디코더 + RVQ 오디오 코덱 | SoundStream [1], EnCodec [2], DAC = Improved RVQGAN [3] (arXiv:2306.06546) | NAC의 큰 틀. "행동을 오디오처럼"이라는 비유의 출처 |
| **다중 스케일 RVQ (스테이지마다 시간 풀링)** | SNAC [16] (arXiv:2410.14411) | 거친 토큰과 세밀한 토큰의 시간 해상도를 달리하는 핵심. 공식 코드 이름도 `SnacVocos`(`04` D1) |
| SEANet 인코더 | SEANet [36], SoundStream [1] | 인코더 본체. 공식 코드의 인코더는 LSTM 2층을 포함(`04` D2) |
| ISTFT 헤드가 있는 Vocos 디코더 | Vocos [32] (arXiv:2306.00814), HiFi-GAN [37] | 크기와 위상을 예측해 ISTFT로 파형 복원 |
| 판별기 (DAC, MPD, MRD) | DAC [3], HiFi-GAN [37], UnivNet [49] | 표 1b에서 DAC, MPD, MRD를 비교 |
| RVQ, 커밋 손실, EMA 코드북 | RVQ [11], VQ-VAE [44], k-means 초기화 [43] | 식 3–4, 알고리즘 1. 공식 코드 기본 설정은 FSQ(`04` D1) |
| 오프셋 코드북으로 단일 어휘 만들기 | OpenVLA류 어휘 오프셋 관행 [6], OAT [7] | 스케일마다 ID 구간을 나눠 한 시퀀스로 예측(식 6) |
| 행동 청크 정책 | ACT [34], Diffusion Policy [53] | 행동 지평 32, 앞 16스텝 실행(표 5) |
| 인과 트랜스포머 AR 정책 틀 | OAT 저장소의 정책·워크스페이스 구조(공식 README의 Acknowledgement), Decision Transformer [33] | NAC 정책 코드는 OAT 코드 위에 만들어졌다고 README가 밝힘 |

## 2. 가까운 대안: 행동 토크나이저 계보

| 방법 | 핵심 | NAC와의 관계 |
|---|---|---|
| Bin (OpenVLA, RT-2 계열 [6]) | 차원별 균등 구간 | 토큰 224개(2240비트). 기준선 |
| **FAST** [9] (arXiv:2501.09747) | DCT + BPE, 손으로 설계한 주파수 prior | NAC의 주 비교 대상. 사전 학습 토크나이저 대신 공식 코드는 청크 데이터로 BPE를 다시 맞춤(`04` D9) |
| VQ-VLA [24] (arXiv:2507.01016) | 인과 합성곱 VAE + 12그룹 VQ | 12토큰, 120비트. NAC와 같은 비트 수 |
| ActionCodec [26] (arXiv:2602.15397), FASTer [27] | RVQ 기반 학습형, FASTer는 트랜스포머 오토인코더와 DCT L1 재구성 | 관련 연구로 언급만 하고 비교하지 않음 |
| **OAT** [7] (arXiv:2602.04215) | FSQ + 레지스터 토큰 + 중첩 드롭아웃으로 "순서 있는" 토큰 | NAC와 가장 가까운 학습형 대안. 12토큰, 120비트. 공식 코드에서 OAT도 FSQ [8, 8, 4, 4]를 씀(`04` D1) |
| MolmoAct [10] | FAST 토큰을 쓰는 행동 추론 모델 | VLA 쪽 사용처 |

NAC의 차별점으로 논문이 내세우는 것은 "완전 합성곱 SEANet 인코더 + 다중 스케일 RVQ + 적대적 학습"입니다(§2.3). 이 중 적대적 학습의 기여는 v2의 자체 결과로 약해졌고(`01` §0, `03` V6), 다중 스케일 자체의 기여는 따로 떼어 보지 않았습니다(`03` C14).

## 4. GitHub와 자료 표

조회일 2026-11-26 (`gh api`, 별 수는 조회 시점 값).

| 저장소·자료 | 관계 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| [ahadjawaid/nac](https://github.com/ahadjawaid/nac) | **공식 구현** (토크나이저·정책 학습, LIBERO·RoboMimic 평가) | Apache-2.0 (LICENSE 파일) | 22 | 2026-06-28 | 대조 기준 커밋 `2e2d9db3d6abde0415f77500f41ff292262a1343`, 400 KB. 최초 공개 2026-06-28(README News). v2(09-25)보다 앞선 코드 |
| [ahadjawaid.com/nac](https://ahadjawaid.com/nac) | 프로젝트 페이지 | 해당 없음 | | | "Conference on Robot Learning (2026)" 표기. 초록은 v1 문장 그대로 |
| [Chaoqi-LIU/oat](https://github.com/Chaoqi-LIU/oat) | OAT 공식 코드. NAC 저장소가 기반으로 삼음 | 라이선스 미지정(API: NOASSERTION) | 121 | 2026-07-27 | NAC 코드 상당 부분이 이 구조를 따름 |
| [Chaoqi-LIU/LIBERO](https://github.com/Chaoqi-LIU/LIBERO) | 공식 저장소가 서브모듈로 쓰는 LIBERO 포크 | MIT | 0 | 2026-02-05 | `third_party/LIBERO` |
| [Chaoqi-LIU/sim_env](https://github.com/Chaoqi-LIU/sim_env) | README Acknowledgement의 환경 코드 | 미지정 | 6 | 2026-05-22 | |
| [hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac) | SNAC 다중 스케일 코덱 (논문 [16]) | MIT | 780 | 2024-11-19 | |
| [descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec) | DAC 코덱·판별기 (논문 [3]) | MIT | 1898 | 2026-07-16 | |
| [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) | Vocos 디코더 (논문 [32]) | MIT | 1168 | 2026-08-29 | |
| [xiaoxiao0406/VQ-VLA](https://github.com/xiaoxiao0406/VQ-VLA) | 기준선 VQ-VLA | MIT | 135 | 2025-11-15 | 공식 저장소가 기준선 코드로 포함 |
| Hugging Face 데이터셋 `chaoqi-liu/libero10_N500.zarr` | LIBERO-10 사전 변환 zarr (README가 안내) | 확인 안 함 | | 2026-01-23 수정 | 사용 저장 약 3.5 GB (HF API 메타데이터, 받지 않음) |
| Hugging Face 데이터셋 `ahad-j/mt4_N800.zarr` | RoboMimic mt4 사전 변환 zarr (README가 안내) | 확인 안 함 | | 2026-05-14 수정 | 사용 저장 약 3.5 GB (HF API 메타데이터, 받지 않음) |

- 공식 저장소에는 학습된 체크포인트의 공개 링크, 실로봇 코드·데이터, 표 3의 지연 측정 스크립트가 없습니다(`04` D12, D13). 2026-11-26 기준 릴리스는 없습니다.
- 가중치와 데이터는 받지 않았습니다. 크기는 API 메타데이터 값입니다.

## 5. 정리

NAC의 새로움은 부품이 아니라 "조합과 이식"입니다. 오디오 코덱의 부품(SEANet, SNAC식 다중 스케일 RVQ, Vocos 디코더, DAC 판별기)을 거의 그대로 가져와 행동에 맞추는 데 필요한 변경을 손실(멜 → MSE)과 표현(평탄화, 작은 hop)에서 찾았습니다. 그 변경이 얼마나 필요한지는 논문 안에서 서로 다른 ablation 행으로 확인되는데(`03` V6), 구성 요소 단위의 기여(다중 스케일, 오프셋 코드북, 거친-먼저 순서, 코덱 대신 단순 합성곱 오토인코더)는 분리되지 않았습니다.

# 02. 선행연구와 관련 저장소

이 문서는 Genie의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드가 있는지를 정리합니다. 번호는 논문 참고문헌 번호가 아니라 저자 연도로 적었습니다(참고문헌 목록은 `paper/paper.txt`의 References).

## 1. 선행연구 계보 (구성 요소별 출처)

| Genie의 요소 | 출처 (저자, 연도) | 관계 |
|---|---|---|
| 이산 잠재 코드북 (VQ-VAE) | van den Oord 외 2017 | 토크나이저와 LAM의 공통 손실 틀 (§2.1) |
| 마스크 토큰 예측 (MaskGIT) | Chang 외 2022 | 동역학 모델의 샘플링과 학습 방식 (§2.4) |
| 시공간 트랜스포머 (ST-트랜스포머) | Xu 외 2020 (arXiv:2001.02908) | 원래는 교통량 예측 논문입니다. 공간 주의와 시간 주의를 번갈아 쓰는 구조를 가져왔습니다 (§2.1, 참고문헌 확인) |
| 영상 토큰화, 시간을 쓰는 인코더 | Phenaki의 C-ViViT (Villegas 외 2023), MaskViT (Gupta 외 2023), TECO (Yan 외 2023), VideoGPT (Yan 외 2021) | Genie는 C-ViViT를 표 3에서 비교 대상으로 씁니다. 메모리 1.6 GB 대 ST-ViViT 0.9 GB |
| 영상 생성 모델의 틀 | Imagen Video, Video diffusion, Phenaki, MaskViT | 같은 흐름의 영상 생성. 행동 입력이 없다는 점이 다릅니다 (§4 비디오 모델) |
| 행동 조건 세계 모델 | Ha & Schmidhuber 2018, Hafner 외 (Dreamer 계열), Micheli 외 2023 (IRIS), Robine 외 2023 | 행동 라벨이 필요합니다. Genie는 같은 역할을 잠재 행동으로 대체합니다 (§4 세계 모델) |
| 잠재 행동 + 영상 (재생 가능한 영상 생성) | Playable Video Generation (Menapace 외 2021), Playable Environments (Menapace 외 2022) | 가장 가까운 선행 연구입니다. 도메인 특화 정적 예시에 한정되고 새 환경을 프롬프트로 만들지 못한다고 저자가 씁니다 (§4 PVG) |
| 행동 없는 영상에서 잠재 행동 학습 | Edwards 외 2019 (ILPO), Schmidt & Jiang 2024 (LAPO), Ye 외 2022 | 같은 목적입니다. 저자는 이 방법들이 대규모로 적용되지 않았다고 씁니다 (§4 에이전트) |
| 인터넷 영상의 역동역학 라벨링 | VPT (Baker 외 2022) | 행동 라벨이 있는 데이터로 역동역학 모델을 학습합니다. Genie는 라벨 없이 잠재 행동을 씁니다 (§4 에이전트) |
| 영상 품질 지표 | FVD (Unterthiner 외 2019) | 주된 품질 지표 |
| 데이터 품질 필터 | Baker 외 2022 (분류기 필터), DINOv2 (Oquab 외 2023) | Baker 외 2022의 분류기 필터 방식을 씁니다 (부록 B.1). DINOv2는 품질이 양보다 중요하다는 근거로 인용됩니다 (B.1) |
| 시뮬레이션 평가 환경 | Procgen CoinRun (Cobbe 외 2020) | 에이전트 실험과 축소판 재현 환경 |
| 로봇 데이터 | RT-1 (Brohan 외 2023), QT-Opt (Kalashnikov 외 2018) | 로봇 학습 모델의 영상 데이터 |
| 학습 인프라 | ZeRO (Rajbhandari 외 2020), 텐서 병렬 (Shoeybi 외 2019), TPU (Jouppi 외 2020), QK 정규화 (Henry 외 2020, Dehghani 외 2023) | 학습 안정화와 규모 확장 |

**가까운 대안과의 차이.** Genie의 핵심 차이는 행동 라벨을 쓰지 않는다는 점과, 텍스트와 행동을 모두 입력으로 받는 모델(GAIA-1, UniSim)보다 영상만으로 학습한다는 점입니다(§4 세계 모델). 가장 가까운 PVG는 정적 예시에 한정되어 새 환경을 만들지 못합니다. Genie가 이 한계를 넘었다고 저자가 주장하며, 그 근거는 정성 그림입니다(`04` C7).

## 2. 코드와 저장소 (GitHub 표)

**공식 코드는 없습니다.** GitHub 조직 `google-deepmind`에 `genie` 저장소가 없고(404), 저장소 검색에서도 공식 구현은 나오지 않았습니다. 프로젝트 페이지(Google Sites)에는 논문 링크만 있고 코드 링크는 없습니다. arXiv 댓글 필드에도 프로젝트 페이지 URL만 있습니다. 가중치와 데이터는 논문에서 공개하지 않는다고 명시합니다(`01` §5).

| 저장소 | 관계 | 라이선스 (GitHub API) | 최근 push | 별 | 상태 |
|---|---|---|---|---|---|
| [myscience/open-genie](https://github.com/myscience/open-genie) | 커뮤니티 PyTorch 구현 (비공식) | MIT | 2024-08-21 | 298 | 저장소 설명만 확인. 클론 안 함 |
| [FLAIROx/jafar](https://github.com/FLAIROx/jafar) | 커뮤니티 JAX 재구현 (비공식, 설명에 "JAX reimplementation") | Apache-2.0 | 2025-01-23 | 107 | 저장소 설명만 확인. 클론 안 함 |
| [alexcbb/Genie-Generative-Interactive-Environments](https://github.com/alexcbb/Genie-Generative-Interactive-Environments) | 커뮤니티 재현 시도 (비공식) | MIT | 2024-04-20 | 9 | 저장소 설명만 확인. 클론 안 함 |
| [kyegomez/Genie](https://github.com/kyegomez/Genie) | 커뮤니티 구현 (비공식) | MIT | 2024-11-11 | 5 | 저장소 설명만 확인. 클론 안 함 |

- 커뮤니티 구현은 공식 결과와 같다고 볼 수 없습니다. 이 분석에서 실행하거나 설치하지 않았습니다.
- 공식 코드가 없으므로 `repo/` 얕은 복제는 하지 않았습니다. 대신 `04_claim_verification.md`에서 주장별 대조를 했습니다.
- 프로젝트 페이지: <https://sites.google.com/corp/view/genie-2024/> (arXiv 댓글 필드의 URL을 그대로 적음).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **VPT (Baker 외 2022)** 는 행동 라벨이 있는 데이터로 역동역학 모델을 학습합니다. Genie §3.3은 VPT가 라벨을 요구한다는 점을 대비점으로 씁니다. 대비가 성립하려면 잠재 행동이 실제 행동과 같은 정보를 담아야 하는데, 논문은 CoinRun의 200개 라벨로 이를 보입니다(`04` C8).
- **MaskGIT (Chang 외 2022)** 의 샘플링 방식은 Genie 추론(25 스텝, 온도 2)의 바탕입니다. 온도 2의 근거는 논문에 없습니다. 축소판(표 17)은 온도 1.0입니다.
- **C-ViViT (Villegas 외 2023)** 는 표 3에서 비교 기준입니다. 저자는 C-ViViT가 과적합 경향이 있다고 씁니다(§3.4). 이 설명은 확인되지 않았습니다.

# SmolVLA 선행연구와 구성 요소 (02)

## 1. 계보

- **로봇 정책의 출발:** Diffusion Policy(Chi et al.), ACT(Zhao et al., 과제별 학습 CVAE 트랜스포머). 청크 예측과 시간적 일관성의 기준선입니다.
- **VLA의 초기 계열:** RT-1, Octo(Team et al. 2024), RT-2(Brohan et al. 2023). OpenVLA(Kim et al. 2024, 7B, 이산 행동 토큰)는 공개 가중치 VLA의 기준점입니다.
- **연속 행동 VLA:** π0(Black et al. 2024, PaliGemma 3B + 흐름 매칭 행동 전문가). SmolVLA가 가장 직접 계승하는 구조입니다. 흐름 매칭, 행동 전문가, 청크 출력이 같습니다.
- **효율 지향 VLA:** TinyVLA(Wen et al. 2024, 1B 미만 모델을 처음부터 학습), DexVLA(Wen et al. 2025). SmolVLA는 이들과 같은 목표를 가지되 로봇 데이터로 사전학습한 작은 VLM을 씁니다.
- **커뮤니티 데이터:** LeRobot(Cadene et al. 2024), 그리고 SO-100/SO-101 저가 로봇 생태계. SmolVLA의 데이터 전략은 Open X-Embodiment류의 대규모 혼합과 다릅니다.

## 2. 가장 가까운 대안

| 대안 | SmolVLA와 같은 점 | 다른 점 | 논문 비교 |
|---|---|---|---|
| π0 (3.3B) | 흐름 매칭, 청크, VLM 기반 | 파라미터 7배, 로봇 사전학습, 전체 VLM 사용 | 표 2, 표 3 |
| ACT | 청크 출력, 시연 학습 | 작은 CVAE, 사전학습 없음 | 표 3, 표 4 |
| TinyVLA | 소형 VLA | 로봇 데이터 없이 처음부터 학습 | 표 2 (Meta-World) |
| OpenVLA-OFT | 청크 출력, 빠른 추론 | 7B, 이산 행동 대신 연속 행동 | 비교하지 않음 |

## 3. 구성 요소별 출처

| 구성 요소 | 출처 (논문 참고문헌) | 비고 |
|---|---|---|
| VLM 백본 SmolVLM-2 | Marafioti et al. 2025 | 논문에 기재된 대로 문서·OCR 중심 사전학습 |
| 시각 인코더 SigLIP | Zhai et al. 2023 | 공개 설정에서 VLM 안에 포함 |
| 언어 디코더 SmolLM2 | Allal et al. 2025 | |
| 흐름 매칭 | Lipman et al. 2022; Liu 2022; Esser et al. 2024 | 시간 분포 Beta는 π0 설정을 따름 (§3.1) |
| 층 건너뛰기 | Shukor & Cord 2024; Tang et al. 2023; El-Nouby et al. 2024; Bolya et al. 2025 | 중간 층 특징이 낫다는 근거 |
| 교차·자기 어텐션 교차 배치 | 논문 고유 설계 (§3.1) | 선행 근거는 Bjorck et al. 2025(교차어텐션 전용), π0(자기어텐션 전용) |
| 비동기 추론 | 논문 고유 (Algorithm 1) | 같은 문제를 다루는 실시간 청크 실행(RTC) 계열 연구가 있으나 논문은 이를 인용하지 않음 |

## 4. GitHub, 코드, 가중치, 프로젝트 자료

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (LeRobot)** | https://github.com/huggingface/lerobot | Apache-2.0 | 별 29,020, 마지막 push 2026-11-22 03:32 UTC | `repo/`에 얕은 clone (HEAD e5be0c1, 2026-11-21). 저장소 크기 약 32 MB. v0.3.2 태그도 얕게 받아 비교 (`git show v0.3.2:...`) |
| 공식 모델 가중치 | https://huggingface.co/lerobot/smolvla_base | Apache-2.0 | 다운로드 127,025, 좋아요 500, 마지막 수정 2026-10-06 | 받지 않음. `config.json`만 읽음 |
| 학습 데이터 예 | https://huggingface.co/datasets/lerobot/svla_so100_pickplace | Apache-2.0 | 마지막 수정 2026-07-16 | 받지 않음. 실로봇 평가 데이터셋(svla_so100_pickplace/stacking/sorting, svla_so101_pickplace)과 같은 계열 |
| 프로젝트 페이지 | https://huggingface.co/blog/smolvla | - | 2025-06-03 게시 | 블로그. arXiv 주석의 "Code and assets" 링크가 가리키는 곳으로 보임 (arXiv 본문에는 URL 미기재) |
| 사전학습 데이터 목록 | 논문 부록 A.1 | - | - | 목록은 본문에 있고 repo에는 없음 (`04` D15) |
| 웹 검색 한 번 | - | - | - | 학회 채택 기록 없음 |

자료 크기: 가중치 파일 크기와 데이터셋 크기는 받지 않아서 확인하지 않았습니다.

## 5. 학회 표기

- arXiv abs: 학회 표기 없음, journal-ref 없음.
- 프로젝트 페이지와 HF 블로그: 학회 표기 없음.
- 웹 검색 한 번: 채택 기록을 찾지 못했습니다.
- **판정:** `학회 표기 없음 (arXiv)`. 워크숍 발표 여부는 확인하지 못했습니다.

## 7. 공개 범위

- 공개 가능한 인용은 논문 텍스트, 표 값, arXiv 메타데이터입니다.
- 논문 그림은 arXiv non-exclusive 라이선스라서 게시본에 넣지 않습니다.

# 02. 선행연구와 공식 자료

## 1. 계보와 가까운 대안

| 축 | 논문 | RDT가 가져온 것 / 다른 점 | 출처 |
|---|---|---|---|
| 확산 정책 | Diffusion Policy (Chi et al., 2023) | 행동 청크를 확산으로 모델링. RDT는 같은 아이디어를 1.2B 트랜스포머로 키움 | 논문 §4.1, 참고문헌 |
| 확산 트랜스포머 | DiT (Peebles & Xie, 2023) | 백본. 조건 주입을 adaLN 대신 교차 어텐션으로 바꾸고 QKNorm, RMSNorm, MLP 디코더를 추가 | 논문 §4.1, 그림 4 |
| 파운데이션 확산 | Octo (Ghosh et al., 2023) | 같은 계열의 확산 파운데이션 모델. 최대 93M. 논문 표 3에서는 확산이어도 성능이 낮음(Octo 종합 1.6) | 논문 §5.1, 표 7 |
| 이산 토큰 VLA | OpenVLA (Kim et al., 2024) | 7B, 행동을 이산화. 논문 표 3에서 양팔 과제 대부분 실패 | 논문 §5.1, 표 3 |
| 양팔 VAE | ACT (Zhao et al., 2023) | ALOHA 하드웨어와 같은 계열. 논문의 가장 강한 베이스라인 | 논문 §5.1, 부록 H 표 11 |
| 다중 로봇 학습 | Open X-Embodiment (Collaboration et al., 2023) | 사전학습 데이터 풀. RDT는 46개 데이터셋(표 5) 사용 | 논문 §4.2 |
| 노이즈 스케줄러 가속 | DPM-Solver++ (Lu et al., 2022) | 100 → 5 스텝 샘플링 | 논문 §5.1, 부록 H |
| 비전·언어 인코더 | SigLIP (Zhai et al., 2023), T5-XXL (Raffel et al., 2020) | 고정 가중치로 사용 | 논문 §4.1, 표 8 |
| 행동 공간 통합 | 본 논문의 128차원 통합 공간 | 다른 로봇을 물리 의미 보존으로 한 벡터에 넣음. 기존 방법은 같은 구조의 로봇만 쓰거나 입력을 버림 | 논문 §3, §4.2, 부록 C |

- **논문 자신의 위치.** 양팔 파운데이션 모델은 드물고(서론, 관련 연구 §2), 기존 양팔 방법은 작은 모델과 단순 과제에 머문다고 씁니다. 그 주장은 2024년 시점의 문헌 검색에 기댑니다. 검색 방법이나 목록을 본문에서 제시하지는 않습니다.

## 2. 공식 자료와 저장소

| 저장소 | 관계 | 라이선스 | 별 (2026-11-22) | 마지막 push | 커밋 | 상태 |
|---|---|---|---|---|---|---|
| [thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer) | 공식 PyTorch 구현 (학습, 파인튜닝, 추론, ManiSkill 평가, ALOHA 배포 스크립트) | MIT (`LICENSE`, API spdx MIT) | 1,821 | 2026-01-21 | `cd79363` (HEAD) | `repo/`에 얕은 복제(`--depth 1`). 정적으로만 읽음. 실행과 설치 안 함 |
| [robotics-diffusion-transformer/rdt-1b](https://huggingface.co/robotics-diffusion-transformer/rdt-1b) (Hugging Face) | 공식 사전학습 체크포인트 (1M 스텝). 파인튜닝 시작점 | MIT (카드 `cardData` 기준) | 다운로드 290, 좋아요 106 (카드 기준) | 마지막 수정 2024-10-17 | — | **받지 않음.** `pytorch_model.bin` 2,456,755,578 바이트 (약 2.46 GB), `head.mp4` 약 16 MB |
| [rdt-robotics.github.io/rdt-robotics](https://rdt-robotics.github.io/rdt-robotics/) | 프로젝트 페이지 (개요, 데모 영상, 코드·데이터·모델 링크) | — | — | — | — | HTML 메타데이터만 확인. 영상은 받지 않음 |
| [openvla/openvla](https://github.com/openvla/openvla) | 베이스라인 OpenVLA 공식 코드 | MIT | 7,329 | 2025-03-23 | — | 참고만 |
| [octo-models/octo](https://github.com/octo-models/octo) | 베이스라인 Octo 공식 코드 | MIT | 1,851 | 2024-07-31 | — | 참고만 |
| [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | 베이스라인 ACT 공식 코드 | MIT | 2,352 | 2024-07-23 | — | 참고만 |

- **원래 코드 구조와 공개 범위.** 저장소에는 `configs/`(모델과 데이터 설정, 128차원 `state_vec.py`, 데이터셋 가중치), `models/`(RDT, 인코더), `train/`, `data/`(HDF5 변환), `scripts/`(ALOHA 배포와 언어 임베딩 사전계산), `eval_sim/`(ManiSkill, OpenVLA, Octo, DP 평가)이 있습니다. 실로봇 평가 자동화와 자체 ALOHA 데이터, 베이스라인 학습 설정은 찾지 못했습니다(`04` §5).
- **학회 확인.** ICLR 2025 포스터입니다(iclr.cc 공식 페이지 `virtual/2025/poster/27746`, proceedings.iclr.cc 항목). 프로젝트 페이지에는 학회 표기가 없습니다.
- 공식 코드의 세부 대조는 `04_code_verification.md`에 있습니다.

## 4. 참고문헌 출처 표기

- 인용 번호는 본문과 참고문헌 목록을 따릅니다(논문 PDF 참고문헌 절). 본 문서의 출처 표기는 논문 절과 표 번호로 합니다.
- 외부 저장소와 HF 메타데이터는 2026-11-22에 GitHub API와 Hugging Face API로 확인했습니다.

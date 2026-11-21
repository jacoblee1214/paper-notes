# 02. 선행연구, 가까운 대안, 공식 코드

## 1. 계보

- **영상으로 지식 배우기:** VideoWorld(Ren et al., CVPR 2025; 본문 [51])는 Go 기보와 시뮬 로봇 영상에서 규칙과 계획을 배우는 자기회귀 영상 모델입니다. 같은 저자들의 직접 선행연구이며, 논문은 VideoWorld의 LDM을 실제 영상에 옮기면 외형에 얽혀 실패한다고 보고합니다(§1, §3.2).
- **잠재 행동 모델 (LAM):** LAPA([72], arXiv:2410.11758)는 프레임 쌍의 변화를 이산 코드로 압축하고, 이를 행동 사전학습에 씁니다. Moto([15], ICCV 2025)는 사전학습 시각 인코더로 동역학을 뽑습니다. AdaWorld([24], arXiv:2503.18938)는 잠재 행동을 조건으로 하는 적응형 세계 모델입니다. iVideoGPT([67], NeurIPS 2024)는 상호작용 영상 GPT입니다. CoLA([62], arXiv:2510.26433)는 VDM으로 잠재 행동을 최적화하지만 두 프레임 전이만 다룹니다.
- **영상 생성 모델 (기준선):** Cosmos Predict2 AR 4B와 DiT 2B([1], NVIDIA 2025), Wan 2.2 14B([59], arXiv:2503.20314), HunyuanVideo 13B([36], arXiv:2412.03603)입니다. 이들은 Video-CraftBench에서 텍스트 단계 설명을 붙여 미세조정합니다(§5.2).
- **분리(disentanglement):** 영상 분리는 스타일 전이나 시점 편집에서 움직임과 외형을 나누는 연구가 많습니다. 논문은 이들이 기하 감독(geometric supervision)이나 수작업 잔차에 기대고, 과제 지식 학습과는 목적이 다르다고 봅니다(§2.1).
- **VDM을 쓰는 다른 방식:** ε-VAE([80])와 DiVoT([26])는 VDM을 픽셀 재구성에 쓰므로 잠재 코드가 외형까지 담아야 합니다. IGOR([13])와 AdaWorld는 VDM을 렌더링에만 쓰고 잠재 최적화와 분리합니다(§6).
- **세계 모델 전반:** 영상 생성 쪽([19, 30, 32, 49, 54, 55, 75]), 강화학습·로봇 쪽([4, 12, 27–29, 37, 43, 50, 65]), JEPA 계열([3, 7, 8, 25, 34, 41, 82, 83])로 나뉩니다. 논문은 이 셋과 달리 긴 과제의 전이 가능한 지식을 목표로 한다고 씁니다(§2.2).

## 2. 가까운 대안과 구성 요소별 출처

| 구성 요소 | 이 논문 | 가장 가까운 출처 | 차이 |
|---|---|---|---|
| 잠재 동역학 인코더 (질의 N, FSQ) | MAGVIT v2형 인과 코덱, FSQ 1000 코드 | VideoWorld, MAGVIT-v2([76], ICLR 2024) | 같은 구조. 새 점은 디코더 교체 |
| 잠재 코드 재구성 | VDM(Cosmos DiT 2B)이 외형을 맡음 | ControlNet([78]) 방식의 조건 분기 | 인과 교차 주의와 기울기 차단 |
| 외형 분리 | 사전학습 VDM을 외형 사전 지식으로 사용 | CoLA([62]), AdaWorld([24]) | 다단계 시퀀스와 긴 영상 대상 |
| 자기회귀 정책 | Cosmos AR 4B에 잠재 토큰 다음 예측 | LAPA의 잠재 사전학습 프로토콜(§5.4) | 잠재 토큰을 긴 영상 단위로 예측 |
| 긴 과제 벤치마크 | Video-CraftBench (종이접기 2, 블록 3, 약 7시간) | CALVIN([45]), Open-X([48]) | 수작업 튜토리얼 영상. 로봇 데이터 아님 |
| 시각 지표 | SSIM, LPIPS | LPIPS([79], CVPR 2018), SSIM([64]) | 표준 |
| 단계 판정 | DINOv2-Base 분류기 ([47]) | DINOv2 | 자체 라벨 약 25k 프레임 |

**가장 가까운 대안은 CoLA입니다.** CoLA도 VDM으로 잠재 행동을 학습하지만 두 프레임 전이에 한정됩니다. 논문은 CoLA의 학습 방식을 자기 AR 설정에 넣어도 최종 단계 성공률이 40.2%(OpenX 설정)에 머문다고 보고합니다(표 1). 이는 같은 설정 비교이므로 `03` §1.1의 검정 대상입니다.

## 4. 공식 코드와 저장소

| 저장소 | 관계 | 라이선스 | 지표 (조회일 2026-11-21) | 상태 |
|---|---|---|---|---|
| [ByteDance-Seed/VideoWorld](https://github.com/ByteDance-Seed/VideoWorld) (`videoworld2/` 하위 폴더) | 공식 구현 (dLDM 학습과 추론 스크립트) | 저장소 루트 Apache-2.0 (GitHub API spdx: Apache-2.0) | 별 809, 포크 42, 마지막 push 2026-02-25 | `repo/`로 얕은 복제 (커밋 3dc3e56), 정적 읽기만 |
| [프로젝트 페이지](https://maverickren.github.io/VideoWorld2.github.io/) | 논문 페이지, 영상, BibTeX | 사이트 템플릿이 Nerfies 기반 CC BY-SA 4.0 (논문 라이선스 아님) | — | 읽기만 |
| [maverickrzw/VideoWorld2_dLDM_2B](https://huggingface.co/maverickrzw/VideoWorld2_dLDM_2B) | dLDM 체크포인트 (VAE, DiT, 코드 사전) | HF 메타데이터 apache-2.0 | 파일 8개, 수정 2026-02-02 | 받지 않음 |
| [maverickrzw/Video-CraftBench_v0.1](https://huggingface.co/datasets/maverickrzw/Video-CraftBench_v0.1) | 벤치마크 데이터 (슬라이스 클립, MP4, 키프레임) | HF 메타데이터 apache-2.0 | 파일 3개, 수정 2026-02-10 | 받지 않음 |
| [nvidia/Cosmos-Predict2-2B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World) | 기반 VDM 가중치 (필수) | HF 메타데이터 `other` (NVIDIA 라이선스, 자동 게이트) | 다운로드 19,146 | 받지 않음 |
| [jxu124/OpenX-Embodiment](https://huggingface.co/datasets/jxu124/OpenX-Embodiment) | 사전학습 데이터 (선택) | HF 메타데이터 cc-by-4.0 | 파일 1,529개 | 받지 않음 |

- **공식 코드의 범위:** dLDM 학습(`scripts/train_dldm*.sh`), 잠재 코드 추출, VDM 디코딩 추론(`scripts/test.sh`), 예제 스크립트가 있습니다. **AR 4B 학습 코드, CALVIN 평가 코드, 단계 분류기(DINOv2) 코드, 기준선 구현은 없습니다.** README는 "AR 모델 학습용 잠재 코드 파일 `latent_code_infos.pt`를 제공한다"고 쓰지만, 이 파일을 쓰는 AR 학습 코드는 저장소에서 찾지 못했습니다(`04` §2).
- **README의 문제:** Paper 링크가 `arxiv.org/pdf/2501.09781`을 가리킵니다. 본 논문은 2602.10102입니다. 설치 명령의 `git clone https://github.com/bytedance/VideoWorld2.git`도 실제 저장소 이름(ByteDance-Seed/VideoWorld)과 다르고, 추론 예시의 예제 경로 `asset/videocraft_example`은 실제 폴더 `assets/videocraft_example`과 다릅니다(`04` §2).
- **Cosmos 기반 모델의 설치 조건:** 설치 스크립트가 CUDA 전용 패키지(transformer-engine 1.12.0, apex, flash-attn 2.6.3)와 CUDA 12.4 PyTorch 휠을 요구합니다. 이 맥에서 설치하지 않았습니다(`04` §4).

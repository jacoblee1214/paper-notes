# 02. 선행연구와 관련 분석

## 1. 계보 한눈에

- **VLA 기반:** 시연 모방으로 학습하는 VLA(RT-2, OpenVLA, π0 계열). 논문은 이들의 오차 누적을 문제로 듭니다(§1). 기본 정책은 VLA-Adapter(Wang 외 2025)의 경량판입니다(§4.1, 부록 A.1).
- **RL 미세조정 쪽:** GRPO(논문 인용 Chen 외 2025b)를 씁니다. 흐름 정합 정책의 확률적 공식화는 ReinFlow(Zhang 외 2025d)를 따릅니다(§3.3). 분산 학습은 VERL(Sheng 외 2025)입니다.
- **VLA+RL 비교 대상:** 온라인은 VLA-RL(Lu 외 2025), 오프라인은 ARFM, RWR, ReinboT(표 10). 세 방법 모두 기본 SR 88.1에서 시작한 결과를 가져왔다고 씁니다(부록 A.3).
- **보상 대체:** 논문은 학습된 보상 모델(VLAC, Zhai 외 2025)을 향후 과제로 듭니다(§5).
- **평가 지표 쪽:** LPIPS(Zhang 외 2018), PSNR, SSIM, MSE로 세계모델 품질을 봅니다(표 1).

**구성 요소별 출처 표** (논문 참고문헌으로 확인한 것만)

| 구성 요소 | 논문에서 쓴 것 | 출처 |
|---|---|---|
| 기본 VLA | VLA-Adapter 경량판 | Wang 외 2025 (arXiv:2509.09372) |
| 흐름 정합 헤드 | 흐름 정합 행동 헤드 | Black 외 2024 (π0) 등 |
| 확률적 흐름 RL | 시그마 넷과 SDE 정식화 | ReinFlow, Zhang 외 2025d |
| 세계모델 토큰 | 자기회귀 영상 토큰 모델 | iVideoGPT, Wu 외 2024 |
| 픽셀 손실 | LPIPS | Zhang 외 2018 |
| 옵티마이저 | GRPO | Chen 외 2025b (논문 표기) |
| 분산 RL 프레임워크 | VERL | Sheng 외 2025 |

## 2. 가까운 대안 (논문 자신이 비교한 것)

| 방법 | 종류 | 이 논문과 차이 | 논문 결과 |
|---|---|---|---|
| VLA-RL | 온라인, 시뮬레이터 RL | 상호작용 많음 (10k 단계, 기본 76.5) | 81.0 (+4.5) |
| ARFM, RWR, ReinboT | 오프라인 RL | 40k 단계, 기본 88.1 | 92.1, 90.8, 91.2 |
| WorldGym, Cosmos, Ctrl-World | 세계모델 평가 | 이 논문은 보상으로 씀 | 본 논문 범위 밖 |

## 4. GitHub, 프로젝트 페이지, 관련 자료 (확인 2026-11-20)

| 자료 | 링크 | 라이선스 | 별 / 최근 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (VLA-RFT)** | https://github.com/OpenHelix-Team/VLA-RFT | MIT (LICENSE 파일 확인) | 별 162, 포크 1. 생성 2025-09-30, 마지막 push 2025-10-06 | `repo/`에 얕은 clone(`376693e`, 2025-10-06 커밋). 정적 읽기만. 로컬에서는 1개 커밋만 보임 |
| 프로젝트 페이지 | https://vla-rft.github.io/ | - | - | 학술 템플릿 잔재. 본문 없음 |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | 미확인 | 미확인 | 공식 코드의 git submodule이지만 `repo/third_party/`는 비어 있음(README만) |
| VLA-Adapter | https://github.com/VLA-Adapter/VLA-Adapter.github.io | 미확인 | 미확인 | 기본 정책 출처(프로젝트 페이지 링크) |
| 세계모델·정책 가중치 | 공식 README의 TODO에서 "Release pre-trained and rft VLA(policy) weights", "Release pre-trained World Model weights"가 미체크 | - | - | 공개되지 않음 (`04` D14) |
| LIBERO RLDS 데이터 | huggingface.co/datasets/openvla/modified_libero_rlds (공식 README) | 미확인 | - | 받지 않음. 약 10 GB(공식 `data/README.md`) |

## 5. 관련 연구의 공통 위험

- 같은 벤치마크라도 기본 정책과 평가 횟수가 다르면 표 간 비교가 어렵습니다(표 10, `01` §5.4).

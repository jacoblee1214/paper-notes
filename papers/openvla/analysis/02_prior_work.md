# 02. 선행연구, 구성 요소의 출처, 공식 코드

## 1. 계보

- **VLM을 로봇 행동에 쓰는 흐름.** RT-2(Brohan et al., 2023, 논문 [7])는 PaLI 계열 VLM을 행동 토큰으로 미세조정했습니다. OpenVLA는 이 레시피를 열린 백본(Prismatic, Llama 2)과 공개 데이터로 옮기는 것이 목적입니다(§1, §2).
- **Open X-Embodiment와 RT-2-X.** 여러 기관의 로봇 데이터를 한 형식(RLDS)으로 모은 OXE(논문 [1])와 그 위에서 학습한 RT-2-X(55B, 닫힌 모델, 논문 [1, 7]). OpenVLA는 RT-2-X를 직접 기준선으로 씁니다. 두 모델의 학습 데이터가 다르고(970k 대 350k 궤적, §5.1), 전처리도 다릅니다(부록 C).
- **일반 정책 대 VLA.** Octo(논문 [5], 93M, 공개 모델)와 RT-1-X(35M)는 처음부터 학습한 정책이라 인터넷 사전학습이 없습니다. OpenVLA는 이들과 비교해 "인터넷 사전학습 + 로봇 데이터" 효과를 봅니다.
- **시연 모방의 기준선.** Diffusion Policy(논문 [3]). 학습 설정이 여러 가지라 논문은 두 판본을 씁니다. 공식 설정과 OpenVLA 입출력에 맞춘 "matched" 판본(§5.2, 각주 3).
- **백본.** Prismatic-7B(논문 [44], 2인칭 두 인코더 DINOv2와 SigLIP, Llama 2 7B 기반, LLaVA 1.5 데이터). DINOv2 [25], SigLIP [79], Llama 2 [10]. OpenVLA는 이들의 가중치를 그대로 씁니다(각주 1).
- **효율 미세조정과 양자화.** LoRA(Hu et al., 논문 [26]), QLoRA(논문 [27]), LLM.int8(논문 [88]). 이 조합을 VLA에 처음 적용했다고 주장합니다(§2, 기여 3).

## 2. 구성 요소별 출처 (논문 기준)

| 구성 요소 | 출처 (논문 참고문헌 번호) | 논문에서 주장하는 역할 | 이 분석의 확인 |
|---|---|---|---|
| 이산 행동 토큰, 256 빈 | RT-2 [7] (방식), 분위수 정규화 (본문) | 연속 행동을 LLM 어휘로 옮김 | 코드 확인 (`04` D1, D2) |
| Prismatic VLM | [44] | 2인칭 인코더와 Llama 2 백본 | 백본 선택 실험은 논문에만 있음 |
| 데이터 혼합 | Octo [5], OXE [1], DROID [11] | 가중치 (표 3) | 가중치 목록은 코드에서 확인, 비율은 재계산 불가 (`04` D5) |
| 실로봇 평가 | BridgeData V2 [6], RT-1/RT-2 Google 로봇 [2, 7], Franka (DROID [11]) | 과제와 시도 수 | 실로봇 없음, 표만 재계산 (`03`) |
| 시뮬 평가 | LIBERO [116] | 부록 E | 실행 안 함 |
| LoRA | Hu et al. [26] | r=32, 모든 선형층 | 코드 확인 (`04` D9) |
| 양자화 추론 | LLM.int8 [88], QLoRA [27] (bitsandbytes 계열) | int8, int4 | 코드 확인 (`04` D12) |

## 4. 공식 코드와 저장소

| 저장소 | 관계 | 라이선스 (확인 방법) | 상태와 기준 | 비고 |
|---|---|---|---|---|
| [openvla/openvla](https://github.com/openvla/openvla) | 공식 학습, 미세조정, 추론, 평가 코드 | MIT (GitHub API `license: mit`, LICENSE 파일 확인) | 얇은 복제 `repo/` (depth 1), 커밋 `c8f03f4`(2025-03-23, "Update README: Remove code block typo"). 별 7,327, 포크 877, 마지막 push 2025-03-23 (API 조회 2026-11-21) | 정적 읽기만. 실행과 설치 안 함 (`04`) |
| [openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b) | 공식 가중치 (HF 허브) | 모델 카드 태그 `license:mit`. 코드 README는 Llama 2 라이선스 적용 가능성을 따로 적음 (`04` D14) | 받지 않음. 다운로드 약 64,500, 좋아요 267, 최근 수정 2026-02-17 (HF API) | 가중치 크기 약 15 GB (bf16 추정) |
| [openvla.github.io](https://openvla.github.io/) | 프로젝트 페이지 | 페이지 자체 라이선스 표기 없음 | 정적 HTML 조회 (2026-11-21). 학회 표기 없음 | 링크: arXiv, HF, GitHub, Colab 노트북 |
| [TRI-ML/prismatic-vlms](https://github.com/TRI-ML/prismatic-vlms) | 기반 학습 코드 (Prismatic) | MIT (GitHub API) | 복제 안 함. 별 1,026, 마지막 push 2024-07-04 | openvla 저장소의 `prismatic/` 패키지와 같은 계열. 대조는 `repo/`로만 함 |

- **라이선스 요약.** 코드는 MIT입니다. 가중치는 HF 카드가 MIT라고 하지만, 코드 README는 "Llama 2에서 파생되어 Llama Community License의 제약을 받을 수 있다"고 명시합니다. 두 문서의 불일치는 `04` D14에서 다룹니다.
- **Colab 노트북.** `examples/openvla.ipynb`가 공식 저장소에 있습니다(README와 프로젝트 페이지 링크). 이 분석에서는 실행하지 않았습니다.

## 5. 이 분석이 조회하지 않은 것

- 후속 연구(OpenVLA를 개선한 모델, 재현 보고서)는 검색하지 않았습니다. 주장은 이 논문 판본과 공식 저장소 코드에 한정합니다.
- LIBERO 저장소(Lifelong-Robot-Learning/LIBERO)와 Octo 저장소는 README에서 언급만 확인했습니다. 복제하지 않았습니다.
- 공식 체크포인트 파일은 받지 않았습니다. 가중치 해시와 실제 크기는 확인하지 않았습니다.

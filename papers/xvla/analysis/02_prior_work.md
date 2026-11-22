# 02. 선행연구와 관련 자료: X-VLA

## 1. 계보

- **프롬프트 학습 (NLP → 로봇).** Soft prompt (Lester et al., 2021), prefix tuning (Li & Liang, 2021), 다중 작업 계층 프롬프트 (Liu et al., 2023c, CVPR). 논문은 이것을 "이질적 원천에 원천별 임베딩"으로 옮겼다고 씁니다 (§3, §B). 로봇 쪽에서 이를 원천 단위로 쓴 직접 선행작은 본문에 없습니다.
- **이질적 사전학습.** HPT (Wang et al., 2024c)는 관측을 원천별 리샘플러로 공통 공간에 올리는 방식입니다. 논문은 이를 (b)로 두고 PT 불안정성을 보입니다 (§3, 부록 I). 그림 2의 (a) 도메인별 action 투영은 π0 (Black et al., 2024a), GR00T N1 (Bjorck et al., 2025), Universal Actions (Zheng et al., 2025) 계열입니다.
- **VLA 흐름 정책.** π0 (Black et al., 2024a), π0.5 (Physical Intelligence et al., 2025), FLOWER (Reuss et al., 2025), SmolVLA (Shukor et al., 2025)가 flow-matching 정책의 주요 비교 대상입니다.
- **표준 구성 요소.** Florence-2 (Xiao et al., 2024, VLM 인코더), flow matching (Lipman et al., 2023), Rotate6D (Zhou et al., 2019, 회전 표현), DiT (Peebles & Xie, 2023)와 MM-DiT (Esser et al., 2024, 비교용 디코더), MLP-Mixer (Tolstikhin et al., 2021, π0-style 비교용), MoE (Shazeer et al., 2017, 실패 시도), LoRA (Hu et al., 2022, PEFT), DAgger (Ross et al., 2011)와 ACT (Zhao et al., 2023) (Soft-Fold 수집).
- **데이터.** Droid (Khazatsky et al., 2024), RoboMind (Wu et al., 2025), AgiBot World (Bu et al., 2025). 사전학습 원천이고 Open X-Embodiment (O'Neill et al., 2024)는 한계에서 언급됩니다.

## 2. 가까운 대안 (논문 표 2 기준)

| 방법 | 크기 | 본문 비교 대상인 이유 | 본 논문 대비 |
|---|---|---|---|
| π0 | 3B | 흐름 VLA 대표. LIBERO 94.1, Simpler-WidowX 27.8, RoboTwin Easy 46.4 | 0.9B X-VLA가 LIBERO 평균과 RoboTwin에서 앞섬. 논문은 PEFT에서 π0 (9M 대 3B)와 비교 |
| π0.5 | 3B 기준 | 표 2에 없음 (참고문헌만) | 비교 없음 |
| FLOWER | 1B | CALVIN ABC→D 4.53 (표 2의 최고) | X-VLA CALVIN 4.43 < 4.53 |
| OpenVLA-OFT | 7B | LIBERO 평균 97.1 | X-VLA 98.1로 우위 (평균) |
| UniVLA (Wang et al., 2025a, Unified VLA) | 9B | NAVSIM PDMS 81.7 | X-VLA 87.3 |
| GR00T-N1 | 3B | VLABench 39.7 | X-VLA 51.1 |

- 기준선 수치는 원논문 보고값을 가져온 것으로 보입니다. 같은 평가 설정으로 저자가 다시 돌렸는지는 본문에 없습니다.

## 4. GitHub, 프로젝트, 모델·데이터 자료 (2026-11-22 확인)

| 자료 | 링크 | 라이선스 | 별 / 포크 | 마지막 push (API) | 비고 |
|---|---|---|---|---|---|
| **공식 코드** | https://github.com/2toinf/X-VLA | Apache-2.0 | 762 / 79 | 2026-10-20 (HEAD `6c4ecfb`) | GitHub 설명: "[ICLR 2026] The official Implementation ..." `repo/`에 얕은 clone (HEAD만, 487개 파일, 126 MB). 정적 읽기만 |
| 프로젝트 페이지 | https://thu-air-dream.github.io/X-VLA/ | - | - | - | 데모 영상과 기준 링크. 본문은 HTML 페이지로 확인 |
| HF 모델 컬렉션 | https://huggingface.co/collections/2toINF/x-vla | - | - | 2026-03-02 (컬렉션 갱신) | 체크포인트 묶음 |
| `2toINF/X-VLA-Pt` (사전학습) | https://huggingface.co/2toINF/X-VLA-Pt | Apache-2.0 | 좋아요 14, 다운로드 8,577 | 2025-11-16 | `model.safetensors` 3.52 GB. 받지 않음 |
| `2toINF/X-VLA-Libero` 외 | HF 컬렉션 참조 | Apache-2.0 (Pt 기준) | - | 2025-11-11 (Libero) | 파일 크기 약 3.5 GB 규모 (Libero 확인). 받지 않음 |
| `Facebear/XVLA-Soft-Fold` (데이터) | https://huggingface.co/datasets/Facebear/XVLA-Soft-Fold | MIT | - | 2025-11-03 | 카드 "1,500 에피소드" (논문 1,200과 불일치), 파일 1,724개, 475.6 GB. 받지 않음 |
| LeRobot 통합 | https://huggingface.co/docs/lerobot/xvla | - | - | - | README가 "LeRobot에 기본 통합"이라고 씀 (확인 안 함) |
| 사전학습 데이터·학습 스크립트 | 공개 없음 | - | - | - | `train.py`는 메타 JSON 경로를 받을 뿐 원천 메타파일은 없음 |

- 라이선스 메모: 논문 자체는 CC BY-NC-ND 4.0 (`README.md` 권리 판정). 코드는 Apache-2.0이고 데이터셋은 MIT, 가중치는 Apache-2.0입니다. 권리 주체가 섞여 있으니 재배포 전에 확인해야 합니다.
- 자료 크기와 "받지 않음": 사전학습 가중치 3.5 GB, Soft-Fold 데이터 476 GB. 이번 분석에서는 다운로드하지 않았습니다.

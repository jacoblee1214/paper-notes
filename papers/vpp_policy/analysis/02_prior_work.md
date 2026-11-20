# 02. 선행연구와 관련 자료

## 1. 계보

VPP는 두 흐름이 만나는 자리에 있습니다. 하나는 로봇용 시각 표현 학습이고, 다른 하나는 미래 예측을 정책에 쓰는 방법입니다. 논문은 두 흐름의 빈칸으로 "영상 확산 모델의 내부 표현"을 내세웁니다 (§1 Introduction, §2 Related Works).

| 흐름 | 대표 연구 (논문 인용) | 입력이 되는 시각 표현 | VPP와의 차이 |
|---|---|---|---|
| 자기지도 시각 표현 | R3M (Nair 2022), VIP (Ma 2022), VC-1 (Majumdar 2023), Voltron (Karamcheti 2023), MVP (Radosavovic 2023) | 한 장 재구성, 두 장 대조, 언어 생성 | 현재 장면만 담고, 미래를 명시적으로 담지 않음 (논문 §2) |
| 생성 모델의 표현 | 확산 모델 표현 (Xiang 2023, Gupta 2024), Hyperfeatures (Luo 2024) | 이미지 확산 모델의 중간 특징 | 이미지 모델에 머묾. 영상 모델의 시간 방향 표현은 탐색되지 않았다고 주장 |
| 미래 예측 + 역동역학 | UniPi (Du 2024), SuSIE (Black 2023) | 생성한 미래 프레임 두 장 또는 목표 이미지 | 여러 번 디노이즈해 느리고 제어 주파수가 낮음 (논문 §2) |
| 영상·행동 자기회귀 | GR-1 (Wu 2023a) | 프레임 한 장씩 생성 후 행동 | 프레임 품질이 확산 방식보다 낮고 사전학습 영상 모델을 쓰지 않음 |
| 영상 모델 특징 + 어댑터 | Vidman (Wen 2024, NeurIPS) | OXE 영상으로 학습한 영상 확산 특징 | 다운스트림 영상 미세조정 없음 (논문 §5.1) |
| 같은 저자의 전작 | PAD, "Prediction with action" (Guo 2024, NeurIPS) | 영상 예측과 행동을 공동 디노이즈 | VPP는 행동을 공동 디노이즈하지 않고 동결된 영상 특징을 읽음 |

## 2. 가장 가까운 대안 (2025–2026)

| 연구 | 미래 정보를 쓰는 방식 | VPP와 비교할 때 볼 것 |
|---|---|---|
| Cosmos Policy (arXiv:2601.16163, ICLR 2026) | 영상 모델을 정책으로 직접 쓰는 방식 | 1스텝 특징 대 다단계 생성의 비용 |
| Motus (arXiv:2512.13030, CVPR 2026) | 잠재 행동 세계 모델 | 미래 정보를 행동 잠재로 줄이는지 |
| LingBot-VA (arXiv:2601.21998, RSS 2026) | 영상과 행동 자기회귀 | 생성 대 표현 |
| Fast-WAM (arXiv:2603.16666, NeurIPS 2026) | 빠른 세계 행동 모델 | 지연과 제어 주파수 |
| WorldVLA (arXiv:2506.21539) | 세계 모델과 행동 자기회귀 | 표현 공유 방식 |
| UVA (arXiv:2503.00200, RSS 2025) | 영상과 행동의 통합 확산 | 공동 디노이즈 대 동결 특징 |
| DreamZero (arXiv:2602.15922, CoRL 2026) | 영상 확산을 영상·행동 공동 예측으로 사용 | 14B 규모의 같은 계열 |
| DreamX (arXiv:2611.05377) | 사전학습 영상 확산의 잠재 행동 토큰 | 같은 백본을 디코더로 씀 |

## 4. 공식 코드와 관련 저장소

| 저장소 | 관계 | 라이선스 | 마지막 push | 스타 | 대조 기준 | 상태 |
|---|---|---|---|---|---|---|
| [roboterax/video-prediction-policy](https://github.com/roboterax/video-prediction-policy) | 공식 구현 (VPP 학습, CALVIN 평가, XBot 실로봇 학습 코드) | MIT (LICENSE 파일, Tsinghua University 2025) | 2025-05-17 | 418 | `75cf37ad0f537627e1a3aa9a04bab70324ec27ff` (2025-05-17) | 얕은 복제 `repo/`, 정적 읽기만 |
| [mees/calvin](https://github.com/mees/calvin) | CALVIN 벤치마크 환경과 데이터 | MIT | 2025-09-08 | 1005 | 공식 저장소 | 서브모듈 미포함 (`repo/calvin/`은 빈 폴더) |
| [intuitive-robots/mdt_policy](https://github.com/intuitive-robots/mdt_policy) | VPP 행동 헤드의 뿌리 (MDT) | MIT | 2024-10-16 | 171 | 공식 저장소 | 읽지 않음 |
| [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models) | SVD 기반 모델 코드 (README가 인용) | MIT (저장소 기준) | 2025-12-16 | 27347 | 공식 저장소 | 읽지 않음. SVD 가중치의 라이선스는 별도 확인 필요 |
| [huggingface.co/yjguo/svd-robot-calvin-ft](https://huggingface.co/yjguo/svd-robot-calvin-ft) | CALVIN용 영상 모델 가중치 (~8 GB) | HF 메타데이터에 라이선스 없음 | 2025-05-06 | — | — | 받지 않음 |
| [huggingface.co/yjguo/svd-robot](https://huggingface.co/yjguo/svd-robot) | 일반 영상 모델 가중치 (~8 GB) | HF 메타데이터에 라이선스 없음 | 2025-05-04 | — | — | 받지 않음 |
| [huggingface.co/yjguo/dp-calvin](https://huggingface.co/yjguo/dp-calvin) | CALVIN용 행동 모델 (~1 GB) | HF 메타데이터에 라이선스 없음 | 2025-05-06 | — | — | 받지 않음 |
| [huggingface.co/datasets/yjguo/vpp_svd_latent](https://huggingface.co/datasets/yjguo/vpp_svd_latent) | 1단계 잠재 데이터 (sthv2, bridge, rt1, calvin) | HF 메타데이터에 라이선스 없음 | 2026-01-11 | — | — | 받지 않음 |
| [video-prediction-policy.github.io](https://video-prediction-policy.github.io) | 프로젝트 페이지 | — | — | — | 2026-11-20 확인 | "ICML 2025 Spotlight", "41.5%", "31.6%", "4k trajectories" 표기 |

- 체크포인트 크기는 README 표에서 가져왔습니다(clip ~600 MB, svd-robot ~8 GB, svd-robot-calvin ~8 GB, dp-calvin ~1 GB).
- HF 저장소의 라이선스는 메타데이터에서 확인하지 못했습니다. 받기 전에 모델 카드를 다시 확인해야 합니다.
- 공식 README는 CALVIN ABC-D 데이터셋을 "about 500 G"라고 적고 있습니다.

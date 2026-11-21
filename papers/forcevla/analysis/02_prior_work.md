# 02. 선행연구, 공식 코드, 이 저장소의 관련 분석

## 1. 공식 자료와 저장소 (조회일 2026-11-21)

| 자료 | 관계 | 라이선스 | 최근 push | 별 | 기준 커밋 | 상태 |
|---|---|---|---|---|---|---|
| [ft-robotic/ForceVLA](https://github.com/ft-robotic/ForceVLA) | **공식 코드로 판단** (근거: 프로젝트 페이지의 "code and data" 링크가 이 저장소를 가리킴, README가 논문 데이터셋 주소를 적음, 데이터셋 계정이 논문 교신 저자 이름과 같음). 포크가 아닌 독립 저장소 | Apache-2.0 (루트 LICENSE 파일 확인) | 2026-01-27 | 121 | `79de88f` (2026-01-27) | **정적 읽기만** (`repo/`, gitignore). 설치·실행 안 함 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 상위 코드 기반 (π0, JAX). ForceVLA 저장소는 이 구조를 포크해 `pi0_force.py`와 `limoe_simple.py`를 추가 | Apache-2.0 | 2026-08-24 | 14,720 | — | 참조만 |
| [HF 데이터셋 qiaojunyu/ForceVLA-real-data](https://huggingface.co/datasets/qiaojunyu/ForceVLA-real-data) | 공식 데이터 (논문 §4.3 "will be made publicly available") | Apache-2.0 (카드) | 2026-01-05 | 좋아요 7, 다운로드 3,013 | sha `6070cbe` | 과제 메타 `meta/info.json` 여섯 개만 받음(`04` D10). 영상·parquet는 받지 않음 |
| [프로젝트 페이지](https://sites.google.com/view/forcevla2025/) | 논문 프로젝트 페이지 (초록·§4.3이 가리키는 주소로 보임) | — | — | — | — | WebFetch 요약으로 확인. 페이지에 "code and data" 링크가 GitHub 저장소를 가리킴. 직접 HTML은 보지 않음 |
| [NeurIPS 2025 poster 120169](https://neurips.cc/virtual/2025/poster/120169) | 학회 공식 목록 | — | — | — | — | 포스터 확인 |
| ForceVLA2 (arXiv:2603.15169) | 후속 연구 (하이브리드 힘-위치 제어, 교차 스케일 MoE). Qiaojun Yu가 교신 저자로 적힌 것으로 검색 결과에 나옴. arXiv 댓글 "Accepted by CVPR 2026" | arXiv 비배타 1.0 | — | — | — | 참조만 (본 분석 대상 아님) |

### 비공식 복제와 모델

GitHub 검색에서 같은 README를 가진 다른 저장소가 여러 개 나왔습니다. 공식이라고 판단할 근거가 없어 쓰지 않았습니다.

| 저장소 | 별 | 최근 push | 메모 |
|---|---|---|---|
| tshiamor/ForceVLA | 3 | 2026-06-06 | README 복제 |
| leversparrow/ForceVLA | 3 | 2025-10-08 | openpi 사본 |
| ZihuatanejoYu/ForceVLA | 2 | 2026-03-10 | README 복제 |
| yuankai0114/ForceVLA | 0 | 2026-10-02 | 설명에 "based on Pi 0.5" |
| xx0508-pig/ForceVLA | 0 | 2026-10-14 | 설명 없음 |

HF에는 제3자가 올린 `forcevla` 체크포인트가 여럿 있습니다(예: `ShaswatRobotics/forcevla_clean`, `magic0/forcevla-flexiv-tactar-june26`). 저자가 공개한 체크포인트는 찾지 못했습니다. 제3자 가중치는 받지 않았습니다.

## 2. 계보

- **VLA 기반:** RT-1, RT-2 (Brohan 외, arXiv:2212.06817, 2307.15818), OpenVLA (arXiv:2406.09246), π0 (arXiv:2410.24164, 흐름 매칭 액션 헤드), π0.5 (arXiv:2504.16054), π0-FAST (arXiv:2501.09747). ForceVLA는 π0 위에 쌓습니다(§4.1).
- **힘을 쓰는 정책:** Adaptive Compliance Policy (arXiv:2410.09309), ForceMimic (arXiv:2410.07554), 힘 인식 반응형 정책 FARP (arXiv:2411.15753), TacDiffusion (arXiv:2409.11047), 온라인 어드미턴스 잔차 학습 (Zhang 외, CoRL 2023), FORGE (Noseworthy 외, RA-L 2025). 논문은 이들을 "힘을 넣지만 정적 융합"으로 묶습니다(§2).
- **촉각 VLA:** TLA (arXiv:2503.08548), Tac-Man (IEEE T-RO 2024). 힘이 아니라 촉각 신호를 씁니다.
- **MoE:** 희소 MoE (Shazeer 외, arXiv:1701.06538), Switch (JMLR 2022), GShard, GLaM, V-MoE (NeurIPS 2021), LIMoE (NeurIPS 2022, 시각-언어 MoE). ForceVLA의 FVLMoE는 LIMoE 블록을 가져옵니다(코드 `limoe_simple.py`, `04` D2).
- **관련 조사:** Xie & Correll, "Towards forceful robotic foundation models" (arXiv:2504.11827). 논문 [35]이 힘 기반 파운데이션 모델의 배경으로 인용합니다.

## 3. 판정 (이 논문의 위치)

- 이 논문의 기여는 "힘을 VLM 뒤에서 MoE로 융합하는 구조"와 "그 구조의 공개 데이터셋"입니다. 두 기여 모두 공개 코드와 데이터로 확인됩니다(`04`).
- 같은 시기 힘 VLA 연구(FR-VLA, LIFT 등)와 비교하면, 이 논문의 실험은 기준선(π0 w/ F)과 어블레이션이 비교적 충실하지만, 통계 검정과 시드 보고가 없습니다(`03`).

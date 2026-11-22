# SPIDER 선행 연구와 관계

## 1. 계보

SPIDER는 사람 시연을 로봇 동작으로 옮기는 세 갈래 사이에 있습니다. 운동학 리타게팅은 빠르지만 물리와 접촉을 무시합니다. RL 리타게팅은 물리를 지키지만 궤적마다 정책을 학습해야 합니다. 논문은 그래디언트 없는 **샘플링 MPC**로 두 장점을 잇습니다(§1, §2, 2–3쪽).

| 갈래 | 대표 연구 (논문 참고 문헌) | 논문이 든 한계 |
|---|---|---|
| 운동학 리타게팅 | Qin et al. 2022 (One Hand to Multiple Hands), Yin et al. 2025a (Geometric Retargeting), Araujo et al. 2025 (GMR, "Retargeting matters"), Lakshmipathy et al. 2025 (접촉이 많은 운동학 리타게팅) | 물리적으로 불가능한 동작, 접촉 관통 |
| 최적화 기반 물리 리타게팅 | Reda et al. 2023 (희소 입력 물리 리타게팅), Yang et al. 2025 (궤적 최적화 데이터 생성), Nakaoka et al. 2005 (무용 리타게팅) | 비볼록성, 강한 사전 지식이 필요 |
| 샘플링 기반 최적화 | Mannor et al. 2003 (CEM), Pan et al. 2024 (Model-Based Diffusion), Xue et al. 2025 (토크 수준 보행, 확산식 어닐링), Howell et al. 2022 (Predictive Sampling), Kamat et al. 2022 (조기 종료), Williams et al. 2018 (강건 샘플링 MPC) | 접촉 모드 모호성, 불안정 (§2.3) |
| RL 리타게팅 | Li et al. 2025c (ManipTrans), Mandi et al. 2025 (DexMachina), Lum et al. 2025 (sim-to-real RL, 사람 시연 1개), Weng et al. 2025 (HDMI, 휴머노이드 정책 틀) | 궤적마다 학습, 계산 비용 큼 |
| 사람 데이터를 로봇 정책에 쓰기 | Park et al. 2025 (DemoDiffusion), Li et al. 2024 (OKAMI), Papagiannis et al. 2025 (R+X), Liu et al. 2025 (EgoZero), Wang et al. 2023 (MimicPlay), Shaw et al. 2022 (VideoDex), Xu et al. 2025 (DexUMI), Yin et al. 2025b (DexterityGen) | SPIDER를 이 파이프라인의 **드롭인 대체**로 위치시킴 (§2.1) |

## 2. 가까운 대안

| 방법 | 같은 점 | 다른 점 | 이 저장소 |
|---|---|---|---|
| ManipTrans (Li 2025c) | Oakink, Inspire 손, 같은 평가 설정 | RL 잔차 정책, 궤적마다 학습 | 비교 기준 (논문 표 1, 17) |
| DexMachina (Mandi 2025) | ARCTIC, Genesis, 목표 보상 | 물체 보조 커리큘럼, RL | 비교 기준 (표 1, 19) |
| GMR (Araujo 2025) | 휴머노이드 운동학 리타게팅 | 물리 없음 | 비교 기준 (표 2) |

## 4. 공식 코드와 자료

| 자료 | 관계 | 라이선스 | 상태 |
|---|---|---|---|
| `github.com/facebookresearch/spider` | 공식 구현 (리타게팅, 시뮬레이터 4종, 데이터 처리) | `LICENSE` 본문 CC-BY-NC 4.0 (비상업) | 얕은 복제, 정적 읽기만 (`repo/`, gitignore). 커밋 `1d607af` (2026-11-17), 별 607, 마지막 push 2026-11-21 |
| `jc-bao.github.io/spider-project` | 프로젝트 페이지 (영상, BibTeX `@misc`, 6,523 시연 보고) | 페이지 바닥 CC BY-SA 4.0 (웹 템플릿) | 저장만 (`paper/project_page.html`) |
| HF `retarget/retarget_full` | 전체 데이터셋 (README 공지 7,876 궤적) | 확인하지 않음 | 받지 않음 |
| HF `retarget/retarget_example` | 예제 장면 (ARCTIC 277 시퀀스) | 확인하지 않음 | 받지 않음 (CPU 워크플로에 필요) |
| `facebookresearch.github.io/spider` | 문서 사이트 | 공식 저장소와 같음 | 읽기만 |

공식 저장소 README의 arXiv 배지는 `2406.12345`로 잘못 적혀 있습니다. 실제 번호는 2511.09484입니다.

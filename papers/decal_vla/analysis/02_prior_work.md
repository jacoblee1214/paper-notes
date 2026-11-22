# 02. 선행연구와 관련 저장소

## 1. 선행연구 계보 (구성 요소별 출처)

| 구성 요소 | DeCAL이 기대는 것 | 출처 (논문 참고문헌 번호) | DeCAL에서 달라진 점 |
|---|---|---|---|
| 베이스 모델과 초기화 | InternVLA-A1: 이해·생성·행동을 한 Mixture-of-Transformers에 묶음, 병렬(비자기회귀) 미래 프레임 디코딩, Cosmos 토큰 압축 | [41] Cai et al., InternVLA-A1, arXiv:2601.02456 | 가중치(A1-3B)로 초기화. 촉각 입력, 촉각 생성, 게이트, 팔·손 분리가 추가 |
| MoT 계열 | Motus(잠재 행동 월드 모델), LaST0(잠재 시공간 사고) | [42] arXiv:2512.13030, [43] arXiv:2601.05248 | 촉각을 같은 구조에 넣음 |
| 세계 행동 모델 | 미래 시각 잠재 예측을 행동에 쓰는 흐름 | [58] Cosmos Policy arXiv:2601.16163, [59] HarmoWAM arXiv:2605.10942, [60] Fast-WAM arXiv:2603.16666 | 시각에 촉각 잠재를 더함. Fast-WAM은 "테스트 시점의 미래 상상이 필요한가"를 묻는 논문이며 DeCAL은 이를 인용하면서도 추론 때의 상상 효과를 따로 검증하지 않음(`04` D10) |
| 이해 백본 | Qwen3-VL (2B 변형을 코드에서 사용), Qwen3(전문가 백본) | [3] arXiv:2511.21631, [56] | — |
| 시각 토크나이저 | NVIDIA Cosmos 연속 이미지 토크나이저 CI8x8 | [61] | 32×32 → 4×4 압축은 [41]을 따름 |
| 흐름 정합 | Flow matching, π0류 시간 샘플링 Beta(1.5, 1.0) | [63] Lipman et al., arXiv:2210.02747 | 팔·손을 두 토큰열로 분리(Factorized FM) |
| 촉각 센서와 표현 | 시각 기반 촉각(손끝 카메라), 변형 깊이 지도 | [55] TacMap, arXiv:2602.21625 | Sharpa 센서의 세 표현(원본, 힘, 변형 지도)을 모두 사용 |
| 촉각 정책(기준선) | ViTacFormer(힘 + 시각 교차 표현), DECO(분리 멀티모달 확산 변환기 + 촉각 어댑터) | [28] arXiv:2506.15953, [32] arXiv:2602.05513 | 둘 다 6축 순힘만 입력으로 바꿔 실행(부록 F) |
| 촉각 미래 예측 | 미래 촉각 신호로 행동을 유도 | [53] DreamTac arXiv:2606.08737(같은 연구실 계열), [28], [29] | 촉각 잠재를 시각 잠재와 한 전문가에서 함께 예측 |
| 촉각 사전학습 | 촉각 자기지도 표현 | [30] 정준 표현·힘 기반 3D 촉각 사전학습, [50]–[52] | DeCAL은 사전학습을 하지 않음(§7 한계) |
| 데이터 수집 | 장갑 + 트래커 원격조작 | [17] Metis(같은 저자 그룹), [64] DexCap | 힘 피드백 없음(§7 한계) |
| 게이트 | 스칼라 시그모이드 게이트로 보조 경로를 조절하는 일반 기법(예: 게이트 교차어텐션) | 논문은 선행을 명시하지 않음 | "접촉 인지" 게이트라는 이름이지만 입력은 손가락 평균 토큰(`04` D2, D3) |

- 같은 저자 그룹(Peking University, BAAI, Shanghang Zhang 그룹)의 선행: CordViP [2], Metis [17], DreamTac [53], Data Pyramid [22]. DeCAL의 데이터 수집과 하드웨어(UR5 + SharpaWave)는 이전 작업과 이어집니다.
- 처음 보는 촉각 센서 사전학습 없이 InternVLA-A1(시각 사전학습)만 쓴다는 점이 논문의 한계 절과 맞습니다.

## 2. 코드와 저장소 (GitHub 표)

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [AureleoPKU/DeCAL](https://github.com/AureleoPKU/DeCAL) | 공식 구현 (LeRobot 포크 기반 학습·추론) | Apache-2.0 (LICENSE, GitHub API). `pyproject.toml`은 CC BY-NC-SA 4.0이라 불일치 | 29 (포크 1, 열린 이슈 1) | 2026-11-21 (커밋 `109f390`, 2026-11-22 확인) | 정적 읽기만(`repo/`, 실행·설치 안 함) |
| [aureleopku.github.io/DeCAL](https://aureleopku.github.io/DeCAL/) | 프로젝트 페이지 ("CoRL 2026", 영상, 요약) | — | — | — | 읽음 |
| [ModelScope Aureleo/DeCALdata_OXT](https://www.modelscope.cn/datasets/Aureleo/DeCALdata_OXT) | 공개 데이터(OXT 형식, 13과제, 1,279 궤적, 약 2.2 TB) | Apache-2.0 표기 | — | 2026-11-21 갱신 | 메타데이터만 확인, 받지 않음 |
| [InternRobotics/InternVLA-A-series (InternVLA-A1)](https://github.com/InternRobotics/InternVLA-A-series) | 초기화 가중치와 기반 코드 | (이 분석에서 확인하지 않음) | — | — | 받지 않음 |
| huggingface/lerobot, openpi, NVIDIA Cosmos tokenizer | 코드 파생 원본(README 감사 인사) | Apache-2.0 계열(복제본 헤더 확인) | — | — | — |
| ViTacFormer, DECO, GR00T N1.6 | 기준선 | — | — | — | 기준선 학습·평가 코드는 DeCAL 저장소에 없음(`04` D18) |

- 확인 방법: `gh api repos/AureleoPKU/DeCAL`(별, 라이선스, push 시각), 커밋 목록, 이슈, ModelScope 메타데이터 API. 클론 해시 `109f3904c46edc96e532e8fd2078c917191b1840`.
- 이슈 1개는 Hugging Face 팀의 논문 페이지 제안이고 저자 답이 없습니다. 릴리스와 체크포인트는 없습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- "시각만으로는 가림 때문에 접촉이 어렵다": [2], [44]의 결과에 기댑니다. 이 논문의 실험에서 같은 구조의 촉각 없는 변형(InternVLA-A1)은 평균 SR 33.3%이고, 촉각을 단순히 더한 InternVLA-A1t는 39.2%로 큰 개선이 아닙니다. 그래서 촉각을 넣는 방식이 중요하다는 주장이 나옵니다. 다만 이 두 기준선의 차이(+5.8 %p, 합산 Fisher p = 0.42)는 시도 수에 비해 작아서 이 주장의 근거로는 약합니다. A1t가 A1보다 낮은 과제도 있습니다(Pipetting 20 대 35, Screw 25 대 30).
- "게이트가 단순 융합의 간섭을 막는다": 이 논문에서 직접 비교는 게이트 유무 절제(표 2)뿐이고 차이는 −15/−10 %p, 비유의입니다(`03` §3).

# The Flip Test: 선행연구와 관련 자료 (arXiv:2611.19822)

## 1. 계보: 이 논문이 어디에 서 있나

| 구성 요소 | 선행 출처 (논문 본문 인용) | FlipTest에서 쓴 방식 |
|---|---|---|
| 세계모델 개념 | Craik 1967, Ha & Schmidhuber 2018 (arXiv:1803.10122), Hafner 외 2025 (Nature) | "예측은 현재 상태에 인과적으로 묶여야 한다"는 전제 (§1) |
| 영상 세계모델 | Cosmos (arXiv:2501.03575), Cosmos 3 (arXiv:2606.02800), OpenAI Sora system card (2024) | 평가 대상과 "물리 AI" 범주 |
| 직관 물리 벤치 | IntPhys (1803.07616), Physion (2106.08261), PHYRE, CLEVRER | 합성 장면에서 물리 예측을 보는 전통. 생성 영상 평가로 옮기지 않음 (§2) |
| 자기지도 직관 물리 | Garrido 외 2025 (V-JEPA 계열, 2506.09985) | 위반 탐지 설정을 비디오 자기지도 모델에 적용한 선행 (§2) |
| 생성 영상 물리 평가 | VideoPhy (ICLR 2025), VideoPhy-2 (ICLR 2026), PhysicsIQ (WACV 2026), WorldModelBench | 영상마다 점수를 매김. 쌍 비교는 없음. FlipTest의 "단일 영상 평가 맹점" 비판 대상 (§2) |
| 최소쌍 | BLiMP (2020), Winoground (CVPR 2022), MMVP (CVPR 2024) | 쌍 정확도의 원형. 한 요인만 바꾸는 설계 |
| 대조 집합 | Gardner 외 2020 (contrast sets, EMNLP Findings) | 결정 경계 주변의 지역 행동 점검 |
| 위반 기대 패러다임 | Baillargeon 외 1985, 1992; Margoni 외 2024; Téglás 외 2011 | 근접 쌍으로 사건 차이를 보는 발달 심리 설계 (§2) |
| 역그래픽, 분석-합성 | Helmholtz 1867, Yuille & Kersten 2006, Picture (Kulkarni 2015), Neural Scene De-rendering (Wu 2017b) | 장면 상태를 명시적으로 복원하는 전통 |
| 세 단계 분해 | Visual De-animation (Wu 2017a, NeurIPS 2017): 지각·물리·그래픽 | 예측기를 세 단계에 대응시켜 실패 지점 진단 (§5) |
| 코딩 에이전트 세계 표현 | Code World Model (arXiv:2608.25927), Agentic World Action Models (arXiv:2610.30136), Code as Worlds (arXiv:2608.27549) | 측정 에이전트, 시뮬레이션 에이전트의 선행. 블로그와 논문 §6이 인용 |
| 판독기 | Gemini 3.6 Flash, Gemini 3.8 Flash (Google DeepMind 2026a, 2026b) | 블라인드 영상 판독, 두 판정기 AND 규칙 (§3.3) |
| 평가 대상 모델 | Wan2.2 (arXiv:2503.20314), HunyuanVideo-1.5 (arXiv:2511.18870), CogVideoX (ICLR 2025), VBVR (arXiv:2602.20159), VPP2 (arXiv:2610.10270), LingBot-Video (arXiv:2607.07675), MiniMax-H3 (블로그, 2026-07), Gemini Omni Flash 1.1 (모델 카드, 2026-08) | 11개 모델 (논문 §4) |

**계보에서 보이는 위치.** 이 논문은 (a) 생성 영상 물리 평가(VideoPhy, PhysicsIQ, WorldModelBench)와 (b) 직관 물리 벤치(IntPhys, Physion, PHYRE)의 중간에 있습니다. 두 쪽 모두의 "단일 영상 점수" 방식을 쌍 대조로 바꿨습니다. 진단 부분은 Visual De-animation(VDA)의 세 단계 분해를 가져왔습니다. 코딩 에이전트를 "지각 모듈"로 쓰는 착상은 2026년 Code World Model 계열과 맞닿아 있습니다.

## 2. 구성 요소와 근거의 질

- 쌍 정확도(두 프레임 모두 맞아야 함)는 최소쌍 평가의 표준 관행을 따릅니다. 우연 25%와 상수 예측기 0%를 명시한 점은 좋습니다. 다만 3지선다 과제가 섞여 있어 "우연 25%"는 정확하지 않습니다(03 §2).
- 선택성 SS = CFR − SFR은 상수 예측기와 무작위 뒤집기 모델에서 0이 되도록 설계되었습니다. 간단하고 해석하기 쉽습니다. 제어 집단(네 상태)은 "같은 크기의 시각 변화"를 같은 쪽에 주므로 "시각 변화에 반응했는가"를 분리합니다.
- 스윕(L3)의 P(event|m)은 심리 측정의 정신물리 곡선과 같습니다. 이 논문에서 비디오 모델의 평평함을 보여 주는 결정적 증거입니다.
- 두 가지 약점이 선행 전통에서 이어집니다: 판독기(VLM)에 의존하는 라벨(VideoPhy 계열과 같음)과 사람 평가의 표본 크기(600개, 판정기 일치 79%).

## 4. GitHub, 프로젝트 페이지, 코드 자료 (확인 2026-11-21, GitHub API 메타데이터만 조회)

| 자료 | 링크 | 라이선스 | 별 / 마지막 push | 비고 |
|---|---|---|---|---|
| **이 논문의 공식 코드** | 없음. 블로그 [Code&Data]는 "expected Dec 2026" | - | - | GitHub 검색(`flip test world model`, `FlipTest` 등)에서 관련 저장소 없음. 공식 코드 clone 없음 |
| 프로젝트 페이지 | https://flip-test.github.io/ → https://romyluo.com/blog/fliptest/ | - | - | 블로그. 논문, 영상, BibTeX 링크 |
| 평가 대상: Wan2.2 | https://github.com/Wan-Video/Wan2.2 | Apache-2.0 | 18,802 / 2026-09-21 | 기본 I2V 모델의 공식 코드 |
| 평가 대상: HunyuanVideo-1.5 | https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5 | NOASSERTION (GitHub가 표준 라이선스를 인식 못 함, 본문 확인 필요) | 4,609 / 2026-04-10 | 라이선스 원문 확인 필요 |
| 평가 대상: LingBot-Video | https://github.com/Robbyant/lingbot-video | Apache-2.0 | 989 / 2026-09-17 | 논문 인용(arXiv:2607.07675)과의 대응은 README 대조 안 함 |
| 평가 대상: VPP2 (후보) | https://github.com/Haodong-Yan/VPP2-project | 없음 | 0 / 2026-09-25 | 저자 목록의 H. Yan과 이름이 맞는 후보. 공식 여부 확인 안 함 |
| 평가 대상: CogVideoX, VBVR, Cosmos3, MiniMax-H3, Gemini Omni | - | 미확인 | - | 이번 분석에서 조회하지 않음. MiniMax-H3와 Gemini Omni는 API 전용으로 보임 (확인 불가) |
| 판독기 Gemini 3.6/3.8 Flash | API | 상용 | - | 재현에 API 키와 비용 필요 |
| 관련 공식 코드 (참고) | https://github.com/ByteDance-Seed/VideoWorld | Apache-2.0 | 809 / 2026-02-25 | 비디오로 세계 학습 (CVPR 2025). 이 논문과 직접 관계는 없음 |

- 원문 PDF의 공식 원자료(영상, 판정 결과)는 공개되지 않았습니다. 블로그의 [Code&Data]가 공개되면 공식 자료 기반의 코드·데이터 대조로 주장 검증을 보강할 수 있습니다.
- 모델 가중치, 데이터셋, API 사용은 이번 분석에서 하지 않았습니다.

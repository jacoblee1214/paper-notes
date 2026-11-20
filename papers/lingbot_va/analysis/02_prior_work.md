# LingBot-VA 선행연구와 관련 자료 (02)

## 1. 계보

| 흐름 | 논문 속 대표 (references 번호는 v2 기준) | LingBot-VA와의 차이 (논문 설명 기준) |
|---|---|---|
| 개별 VLA 정책 | π0 [7], π0.5 [29], GR00T-N1 [6], X-VLA [93], OpenVLA [34], CronusVLA [37], FLOWER [62] | 행동을 직접 예측. 동역학을 명시적으로 모델링하지 않음 (§1, §5) |
| 영상 확산을 행동에 쓰는 흐름 | UVA [40], UWM [97] (chunk 단위 양방향 확산) | 같은 시퀀스 안의 양방향 어텐션 때문에 인과성 위반, KV 캐시 기억 없음 (§3.2) |
| 영상 생성기를 부목표·정책에 쓰는 흐름 | Gen2Act [4], Act2Goal [95], VideoVLA [64] | 열린 루프 생성, 지연이 큼 (§5) |
| 상호작용 시뮬레이터 | UniSim [86], Genie [13] 계열 | 게임과 시뮬레이션 영역, 정밀 조작으로 옮기기 어려움 (§3.2) |
| 세계 모델 기반 정책 (행동을 함께 생성) | Motus [5] (잠재 행동 세계 모델), Vidar [22] | 실험 비교 대상 (Motus는 표 1 최강 비교군) |
| 역동역학 기반 정책 | Seer [73], IDM 계열 [1, 20, 55] | 미래 목표를 조건으로 행동 추론 (식 9) |
| 영상 생성 백본 | Wan2.2-5B [79], 인과 VAE [79], T5 텍스트 인코더 [59], MoT [43] | 백본 초기화 (§4.2) |

핵심 대비는 "양방향 확산 대 인과 자기회귀", "열린 루프 대 KV 캐시 기반 폐루프", "영상 생성 대 행동 복원"입니다. 논문은 이 세 축에서 UVA/UWM과 대비합니다 (§3.2, §5).

## 2. 구성 요소별 출처

| 요소 | 출처 | 비고 |
|---|---|---|
| 흐름 매칭 | [46, 50, 75] | 식 (1)–(3) |
| 영상 VAE와 백본 | Wan2.2 [79] | 4×16×16 압축 |
| 텍스트 인코더 | T5 [59] | 고정 |
| MoT 구조 | [43] (Mixture-of-Transformers), [5] Motus | 저장소의 README가 MoT 참조 |
| 분위수 정규화, LIBERO 필터링 | OpenVLA [34] | §4.2 |
| 비동기 실행 | [8] real-time chunking [Black 외] 계열 | 본문에 직접 인용 |
| 평가 벤치마크 | RoboTwin 2.0 [15], LIBERO [47] | 표 1, 2 |

## 4. 공식 코드, 프로젝트 페이지, 체크포인트, 데이터 (확인 2026-11-20)

| 항목 | 위치 | 라이선스 (확인값) | 상태 |
|---|---|---|---|
| 논문 | arXiv:2601.21998 (v1 2026-01-29, v2 2026-03-22) | CC BY 4.0 (abs 페이지 "view license" 링크 확인) | `paper/` |
| 공식 코드 | github.com/robbyant/lingbot-va, 커밋 `7c6ffa9` (2026-07-10), 별 1988, 포크 195, 마지막 push 2026-07-09, 갱신 2026-11-19 | Apache-2.0 (`LICENSE.txt`, 저장소 메타데이터 일치) | `repo/` 얕은 복제, 정적 읽기만 |
| 프로젝트 페이지 | technology.robbyant.com/lingbot-va | 표기 없음 (페이지 하단 "© 2026 Robbyant ... All rights reserved") | 상태만 확인 |
| 후속 프로젝트 페이지 | technology.robbyant.com/lingbot-va-v2 | - | 다른 모델 (`repo/LingBot_VA2_paper.pdf`) |
| 체크포인트 | huggingface.co/robbyant/lingbot-va-base, `...-posttrain-robotwin`, `...-posttrain-libero-long` | 확인 안 함 (HF 카드 미확인) | 받지 않음 |
| 데이터 | huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot, `libero-long-lerobot` | 확인 안 함 | 받지 않음 |
| 백본 | Wan2.2-5B | 확인 안 함 | 받지 않음 |

저장소의 `README.md`는 공개 체크포인트가 "shared backbone" 버전이라고 적고, 분리된 버전은 "stay tuned"라고 합니다 (2026-01-29 항목). 이 점은 `04` D3에서 다룹니다.

## 5. 학회 판정

- **판정: RSS 2026.** RSS 2026 논문집(Robotics: Science and Systems XXII) 15번 논문으로 실려 있습니다(roboticsproceedings.org/rss22, 2026-11-20 확인). 프로젝트 페이지와 GitHub 저장소 표기도 같습니다.
- 근거 1: 프로젝트 페이지 상단 "RSS 2026 Tech Report" 문구. 근거 2: GitHub 저장소 설명 "[RSS 2026] Causal video-action world model...". 근거 3: 중국 매체 보도(QbitAI, 2026-05, 채택 소식). 공식 RSS 2026 accepted-paper 목록은 찾지 못했습니다.
- arXiv 댓글과 저널 참조에는 학회 표기가 없습니다. 본문에도 채택 문구가 없습니다.
- 따라서 front matter는 `venue: "RSS 2026"`, `venue_note: "본 학회 (RSS 2026 논문집 p015, roboticsproceedings.org/rss22 확인)"`로 씁니다.

## 6. 미확인

- 중국 매체 보도와 프로젝트 페이지의 RoboTwin 수치(92.0/91.1)가 논문 표 1의 92.9/91.6과 다릅니다. 어느 버전의 수치인지 확인하지 못했습니다.
- Hugging Face 체크포인트의 라이선스 필드와 Wan2.2 백본 라이선스는 받지 않아서 확인하지 못했습니다.

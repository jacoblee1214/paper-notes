# 선행연구와 관련 자료 (π0.5)

## 1. 학회·버전 판정

| 항목 | 확인 결과 | 판정 |
|---|---|---|
| arXiv 버전 | v1 한 개 (2025-04-22). abs 페이지 Submission history에 v2 없음 | 단일 버전 |
| 학회 목록 | PMLR `v305` “Proceedings of The 9th Conference on Robot Learning”, pp. 17–40, 2025 (`proceedings.mlr.press/v305/black25a.html`, 2026-11-21 직접 확인) | **CoRL 2025 공식 proceedings에서 확인** |
| 우리 저장소 목록 | `conferences/corl2026/papers.json`에는 없음 (해당 목록은 CoRL 2026 대상이라 이 논문은 대상 아님) | 참고 |
| 프로젝트 페이지 | 자동 접근 실패(429)로 학회 표기를 직접 확인하지 못함 | 판정에 쓰지 않음 |

판정: `venue: "CoRL 2025"`, venue_note “PMLR v305 (CoRL 2025 공식 proceedings)”.

## 2. 공식 코드와 관련 저장소

| 이름 | 위치 | 라이선스 | 최근 push | 별 | 상태 |
|---|---|---|---|---|---|
| openpi (Physical Intelligence) | github.com/Physical-Intelligence/openpi | Apache-2.0 | 2026-08-24 | 14,720 | 얕은 clone 완료 (commit `215abfb`, 2026-08-25). `repo/`에 두고 정적으로만 읽음 |
| 프로젝트 페이지 | pi.website/blog/pi05 | — | — | — | 접근 실패 (429, Vercel 확인 페이지) |
| 논문 HTML/TeX 원본 | arXiv (TeX Source 링크) | — | — | — | 미받음 (PDF로 충분) |

openpi README는 π0.5를 “knowledge insulation으로 학습한 π0의 개선판”이라고 소개합니다. 같은 README는 **이 저장소에서는 π0.5 학습과 추론 모두 flow matching 헤드만 지원한다**고 명시합니다. 두 번째 문장이 이 논문과 코드의 가장 큰 차이입니다 (`04`의 D1).

공개 체크포인트(README 표): `pi05_base`(미세조정 기반), `pi05_libero`, `pi05_droid`. 이들은 논문 Sec. IV-D의 “post-training 결과물”에 해당하는 것으로 보이지만, 논문이 가리키는 학습 완료본과 같은지는 확인할 수 없습니다.

## 3. 이 논문이 기대는 선행연구 계보

논문 Sec. II의 인용을 다섯 줄기로 묶었습니다. 번호는 원문 참고문헌 번호입니다.

1. **VLA 기반 모델**: RT-1 [9], PaliGemma [5] (백본), π0 [8] (flow matching 액션 전문가). 논문은 VLM을 VLA로 바꾸는 흐름을 [23, 92] 등으로 묶어 인용합니다. π0.5는 π0의 백본과 액션 전문가 구조를 그대로 가져옵니다.
2. **이산 액션 토큰**: FAST [64] (Pertsch 외). π0.5 pre-training의 액션 표현. 논문은 “학습이 훨씬 빠르지만 추론은 자기회귀 디코딩이 느리다”는 FAST의 한계를 그대로 받아, post-training에 flow 전문가를 붙이는 설계를 택했습니다.
3. **이질 데이터 공동 학습**: OXE [15], Bridge [25], RH20T [30], AgiBot World [1] 등 (논문 Sec. II 인용). π0.5는 여기에 웹 VQA·검출·캡셔닝과 고수준 subtask 라벨을 같은 transformer에 넣는다는 점을 차별점으로 내세웁니다 (Sec. II, “Non-robot data co-training”).
4. **고수준 추론과 계층 구조**: SayCan [2], RT-H [4], NaVILA [13], embodied chain-of-thought 계열 [88, 46, 61] 등. 대부분 VLM과 저수준 정책을 **두 모델**로 나눕니다. π0.5는 **한 모델**이 두 수준을 모두 생성한다는 점을 내세웁니다.
5. **일반화 평가**: UMI [14], Robot Utility Models [28], SPOC [26] 등 (논문 Sec. II, 새 환경 일반화). 논문은 “짧은 단일 과제 일반화는 있었지만 10–15분짜리 다단계 과제는 보고되지 않았다”고 주장합니다 (Sec. II 마지막 단락). 이 주장은 참고문헌 전체를 확인하지 않아 검증하지 못했습니다.

## 4. 가까운 대안 (논문이 직접 비교한 것)

| 비교군 | 설명 | 논문에서의 위치 |
|---|---|---|
| π0 | flow matching 단독, 연속 액션 | Fig. 12, 15 |
| π0-FAST+Flow | π0.5와 같은 hybrid 손실, 로봇 데이터만, HL·WD 없음 | Fig. 12, 15 (π0.5 설계의 “고수준·웹 데이터 제거판”) |
| GPT-4 HL | 고수준을 GPT-4로 대체 | Fig. 13, 17 |
| implicit HL | 고수준 추론 없이 HL 데이터로만 학습 | Fig. 13 (두 번째로 높음) |
| human HL | 사람을 고수준 오라클로 | Fig. 13 (상한선) |



# 01. 튜토리얼 페이지와 챌린지 페이지 분석

- **대상 1 (튜토리얼):** *From Human Videos to Dexterous Robot Capability — CoRL 2026 Tutorial*, https://nvidia-isaac.github.io/video_to_data/v2d_corl_tutorial/
- **대상 2 (동반 챌린지):** *Video to Data (V2D) Challenge*, https://nvidia-isaac.github.io/video_to_data/v2d_challenge/
- **종류:** 튜토리얼 웹 페이지와 챌린지 웹 페이지, 그리고 저장소 `nvidia-isaac/video_to_data`. **심사를 거친 논문이 아니고 arXiv 판도 없습니다.** 근거는 페이지 문장, 저장소 코드, 키트 채점 코드입니다.
- **접근일:** 2026-11-12 (HTML 직접 수집, `paper/tutorial.html`, `paper/challenge.html`, 추출 텍스트 `paper/*.txt`). 챌린지 FAQ PDF와 제출 키트는 2026-11-12 내려받음.
- **저장소 기준:** `5226e3474173129545d703aaf5a9761fcd9b3a3d` (2026-11-09 커밋, "docs: update tutorial venue, instructors, and panelist affiliation (#219)").
- **날짜:** 튜토리얼 날짜는 페이지에 적혀 있지 않습니다. 챌린지 일정표에 "November 12 — Winner announced at our CoRL 2026 tutorial"이 있으므로 **2026-11-12 오후 2시–6시(Hilton Austin, Salon D)로 봅니다.** 추정입니다.

## 0. 권리 판정

| 대상 | 판정 | 근거 |
|---|---|---|
| 페이지 글(텍스트) | 저장소 라이선스에 따라 CC BY 4.0 표기가 필요하다고 읽음. 인용은 짧게 하고 출처를 적음 | 저장소 `LICENSE` 머리글: "documentation/skills under the CC-BY-4.0 AND source code under Apache-2.0". 페이지 HTML은 `docs/v2d_corl_tutorial/index.html`에 있음 (저장소 사본과 바이트 단위로 같음) |
| 페이지 미디어 (히어로 이미지, 강사·패널 사진, 기관 로고, 그림, 영상) | **공개물에서 제외** | 저장소 라이선스 머리글이 미디어 개별 라이선스를 밝히지 않음. 사람 사진이 포함됨. 사용자 지시에 따라 제외 |
| 챌린지 FAQ PDF, 제출 키트 zip | 재배포하지 않음. 수치와 수식만 인용 | 라이선스 표기 없음. 키트 루트에 `LICENSE` 없음 (MHR 부분만 Apache-2.0 `metric_code/track_1/LICENSE.txt`) |
| 저장소 소스 | Apache-2.0 (NVIDIA 작성 코드). 단, `reconstruction/NOTICE`에 있는 제3자 모델은 비상업·파생금지 등 별도 조건 | `LICENSE`, `reconstruction/NOTICE`, `video_ingestion_agent/NOTICE` |
| 허깅페이스 데이터·모델 | 데이터셋 2종 CC BY 4.0, VLM은 OpenMDW-1.1, SigLIP-2는 Apache-2.0. 이번 작업에서 다운로드 안 함 | HF 메타데이터 (`data/hf_*.json`, 2026-11-12) |

요약: 이 폴더의 공개 산출물은 **직접 계산한 수치, 직접 그린 그래프, 직접 쓴 toy 코드, 짧은 인용**만 담습니다. 페이지 미디어와 키트 파일은 넣지 않았습니다.

## 1. 페이지가 약속하는 것

- **목표:** "검증 게이트" 순서로 파이프라인을 가르칩니다. (1) 검색 가능한 영상 구간, (2) 믿을 만한 재구성, (3) 시뮬레이션용 과제 자산, (4) 리타겟된 동작, (5) 정책 학습 산출물. 원문: "We teach the pipeline as a sequence of validation gates".
- **학습 목표 5개:** 영상을 질의 가능하게 만들기(구간 분할, 개체 그래프, 프레임 임베딩, 자연어 검색), 재구성 검사(깊이·마스크·메시·손/몸·6D 자세·시간 정렬), 로봇용 데이터로 바꾸기(모션 파일, 물체 자산, URDF/USD, 지지면, 접촉 필드, 품질 검사), 정책 학습 연결(Isaac Lab 소규모 학습·평가, 추적·접촉 보상), 실패 진단(화질, 마스크 오류, 메시 축척, 자세 drift, 잘못된 리타겟, 불가능한 지지 기하, 약한 접촉).
- **실습 형식:** 짧은 개념 설명 뒤에 바로 실습. 각 실습은 세 경로를 가짐: **노트북만으로 하는 점검 경로**, **선택적 라이브 GPU 실행**, **미리 만든 산출물**. 판정은 **ready / questionable / unsuitable** 셋 중 하나.
- **실습 산출물(체크포인트):** 실습 1은 JSONL 클립 목록, 실습 2는 메시·자세 오버레이, 실습 3은 리타겟 모션 파일, 실습 4는 씬 뷰어·정책 롤아웃.
- **참가 후 제공하겠다는 자료:** 설정 안내와 노트북 점검 경로, 실습마다 실행 가능한 스크립트 또는 노트북, 샘플 영상·미리 만든 출력·주석 달린 결과 폴더, Isaac Lab 설정과 문제 해결 메모, 정책 롤아웃 예, 새 데이터에 적용하는 검증 체크리스트. 원문은 "Materials will be public and useful even for participants who do not use NVIDIA infrastructure after the event"입니다.

**저장소 대조 결과:** 실습용 노트북과 "laptop" 경로는 현재 저장소에 없습니다(04 D1). 실습 자료는 행사 후 공개 예정으로 적혀 있습니다.

## 2. 일정 (페이지 표, 2026-11-12 추정)

| 시간 | 내용 | 이 폴더의 대응 |
|---|---|---|
| 2:00–2:05 | 준비와 파이프라인 지도 | 03 §1 |
| 2:05–2:20 | 실습 1: 영상을 질의 가능하게 만들기 | 게이트 1, `video_ingestion_agent/` |
| 2:20–2:35 | 실습 2: 사람 시연 재구성 | 게이트 2, `reconstruction/` |
| 2:35–2:50 | 실습 3: 재구성 결과를 로봇 과제로 바꾸기 | 게이트 3·4, `robotic_grounding/scripts/retarget/` |
| 2:50–3:05 | 실습 4: Isaac Lab에서 손 과제 학습 또는 평가 | 게이트 5, `robotic_grounding/` |
| 3:05–3:20 | 실습 5: AI 에이전트 활용 | `.claude/skills/`, `.codex/skills/` (04 C14) |
| 3:20–3:25 | 추가 과제: 실패 진단과 자기 데이터 가져오기 | 03 §6 |
| 3:25–3:30 | 정리와 피드백 | — |
| 3:30–4:00 | 포스터 세션, 쉬는 시간 | — |
| 4:00–5:00 | 패널: 사람 영상에서 배우는 미래 | — |
| 5:00–6:00 | 챌린지 우승 발표와 상위 팀 발표 | §4 |

## 3. 강사·패널·주최

| 이름 (페이지 기재) | 소속 | 페이지에 적힌 역할 |
|---|---|---|
| Yan Chang | NVIDIA | 튜토리얼 조정, 로봇 학습 검증 |
| Xinghao Zhu | NVIDIA | 전체 파이프라인, 시뮬레이션용 로봇 과제 변환 |
| Mrinal Verghese | NVIDIA | 영상 수집, 개체 그래프, 임베딩, 자연어 검색 |
| Bowen Wen | NVIDIA | 재구성: 깊이, 마스크, 메시, 6D 자세 추적 |
| Zeo Liu | NVIDIA | 손 정책 학습, 에이전트 워크플로 |
| Shalin Jain | NVIDIA | 전신 손 조작, 리타게팅, 접촉 보상, 롤아웃 |
| Naema Bhatti, Abrar Anwar | NVIDIA | 역할 미기재 (주최) |

패널 8명: Jeannette Bohg (Stanford), Kuan Fang (Cornell), Yuting Ye (Meta), Kaifeng "Alex" Zhang (Sharpa), Roei Herzig (UC Berkeley · MIT-IBM), Louis Lian (Lightwheel), Javier Romero (Meta), Yan Chang (NVIDIA). 주최 목록에는 Joydeep Biswas (UT Austin)과 Michael Lin 등이 더 있습니다.

## 4. 챌린지 (V2D Challenge)

### 4.1 트랙

| 트랙 | 입력 → 출력 | 데이터 (페이지·FAQ) | 지표 (키트 코드에 정의) | 채점 |
|---|---|---|---|---|
| 1. Reconstruction | 3인칭 단안 RGB → 4D 사람-물체 상호작용 (사람 MHR 궤적, 물체 메시·자세) | 헤비 오클루전, 양손 협응, 긴 시퀀스를 포함한 3인칭 영상. held-out GT | 정확도: CD-H, CD-O (쌍방향 Chamfer, cm). 물리 타당성: ACC-H, ACC-O (가속도 오차), PEN (관통 깊이, cm). 두 축을 같은 비중으로 | Kaggle, MV 기준선 대비 |
| 2. Robotic grounding | 3인칭 4D HOI (트랙 1 출력 형식) → G1 + Dex3 정책 (전신 제어 허용) | 과제 30개 궤적 (FAQ). 3단계: **Tier 1** 깨끗한 다시점 캡션, **Tier 2** 합성 교란, **Tier 3** 기성 재구성 (CARI4D 제공) | AUC (ADD, 1–9 cm), SP-SR, MP-SR, MPPE (cm) | 티어마다 따로 제출, 세 티어 동일 비중 |
| 3. Egocentric | 1인칭 영상 → 정책 (Sharpa Wave 기본, 시뮬레이션만) | 영상 20개 (FAQ). held-out GT | AUC, SP-SR, MP-SR, RPE (cm), MPPE, CD-O (메시) | Kaggle, 최종 순위는 지표별 순위 합이 가장 낮은 팀 (FAQ) |

트랙 2 설명에서 키트 README는 "Tier 2·3은 추적을 정책 입력 기준으로 보고하고, 리더보드 지표는 깨끗한 기준 궤적에 대해 계산한다"고 적습니다.

### 4.2 규칙과 상금

- **제출:** Kaggle 대회 4–6개(트랙·티어·지표별)에 제출. 페이지는 "주당 5회"라고 하고, FAQ는 "트랙과 티어마다 따로 계산"이라고 합니다(§5 D4). 마지막 3일은 제한 없음.
- **자격:** 유효한 최종 제출, 코드 공개 합의, 짧은 기술 보고서. 페이지는 "research license"라고, FAQ는 "Apache 2.0"이라고 적음(D5).
- **컴퓨팅 크레딧:** 대학·비영리 소속 팀에게 최대 100 크레딧, 11월 4일까지 유효 제출한 팀만 (FAQ).
- **상금:** 페이지는 총 $54K. 트랙별 우승 DGX Spark ($6K 가치), 준우승 $3K GPU 크레딧, 산업 혁신상 DGX Spark + $3K ($9K 가치). 산업 혁신상 기준: 경제성, 데이터 플라이휠, 혁신성, 복잡도 감소. 페이지 문장은 "Three winners per track"과 "Two winners per track"이 섞여 있습니다(D3). 상금 합계는 트랙 2를 티어 세 개로 셀 때만 $54K이 됩니다(03 §2 N1).
- **심사위원:** Hao Su, Yunzhu Li, Louis Lian, Changliu Liu, Yuting Ye, Kaifeng Zhang, Javier Romero (페이지).

### 4.3 일정

| 날짜 | 사건 |
|---|---|
| 2026-09-21 | 등록 시작, 챌린지 데이터 공개와 공식 출범 |
| 2026-11-04 (수) 17:00 EST | 리더보드 동결 (최종 제출 마감) |
| 2026-11-12 | 튜토리얼에서 우승 발표, 상위 팀 발표 |
| 이후 | 우승 솔루션 평가와 기술 보고서 준비 (후속 보고서에 팀 공동 저자 초대) |

### 4.4 리더보드 스냅샷 (2026-11-09T20:04:43Z 생성, 2026-11-12 수집)

집계만 적습니다. 팀 이름은 넣지 않았습니다. 원본 JSON은 챌린지 페이지가 읽는 별도 저장소에 있고, 저장소 `docs/v2d_challenge/leaderboard.json`은 2026-09-29에 만든 빈 파일("No submissions scored yet")입니다(D7).

| 트랙 | 항목 수 | 미완 | 주요 지표 최선 / 중앙값 |
|---|---|---|---|
| 트랙 1 | 60 | 1 | CD-H 11.37 / 14.86 cm, ACC-H 0.132 / 0.140 cm, PEN 0.000 / 0.0001 cm |
| 트랙 2 티어 1 | 29 | 1 | AUC 0.949 / 0.593, SP-SR 1.00 / 0.70, MP-SR 1.00 / 0.57, MPPE 0.84 / 14.2 cm |
| 트랙 2 티어 2 | 25 | 0 | AUC 0.823 / 0.429, SP-SR 0.967 / 0.567, MP-SR 0.967 / 0.267, MPPE 1.69 / 18.1 cm |
| 트랙 2 티어 3 | 20 | 0 | AUC 0.616 / 0.236, SP-SR 0.667 / 0.20, MP-SR 0.467 / 0.033, MPPE 11.0 / 47.8 cm |
| 트랙 3 | 50 | 3 | AUC 0.985 / 0.498, SP-SR 1.00 / 0.55, MP-SR 1.00 / 0.15, RPE 0.63 / 9.22 cm, MPPE 0.60 / 7.42 cm, CD-O 0.207 / 0.396 |

티어 2의 MPPE 최선(1.69 cm)과 중앙값(18.1 cm)은 티어 1(0.84 cm, 14.2 cm)보다 나쁩니다. 교란 입력이 정확도를 떨어뜨린다는 기대와 맞습니다. 트랙 3의 MPPE 최선(0.60 cm)은 다른 과제이므로 직접 비교하지 않습니다.

## 5. 페이지 안의 쟁점 (04 문서의 D 항목과 연결)

1. **실습 자료 부재:** 노트북, 'laptop' 경로, 실습별 산출물 파일이 저장소에 없습니다 (D1).
2. **지표 정의의 출처:** 트랙 2 지표는 "as defined in the CHORD manuscript"라고 적혀 있습니다. 임계값은 키트 코드가 정본입니다. CHORD 본문은 지표 출처만 밝히고 임계 숫자는 찾지 못했습니다 (D2).
3. **문구 불일치:** 우승자 수, 제출 제한의 단위, 코드 라이선스 요구 (D3–D5).
4. **Tier 3 기준선:** 페이지는 CARI4D를 "off the shelf"라고 하지만, 저장소 체크포인트 이름은 `nvidia/cari4d_commercial`이고 리더보드 주석은 "commercial friendly recreation ... performance may differ"라고 씁니다 (D8).
5. **Tier 2 교란의 정의:** 페이지는 "Track 1의 오차 분포에서 뽑았다"고 하지만 교란 생성 코드와 분포가 저장소·키트에 없습니다 (D9).

## 6. 이 폴더가 다루는 범위

- 이 튜토리얼은 **페이지 + 저장소 + 챌린지**라서, 리포트의 판단 단위는 "논문 주장"이 아니라 "페이지 주장 ↔ 코드"입니다. 대응은 `04_code_verification.md`에 있습니다.
- 챌린지 평가 수식은 키트 `metric_code/`를 읽어 확인했고, 실행하지 않았습니다.
- 재현(학습·추론 실행)은 하지 않았습니다. 구현 가능성은 `04_code_verification.md` §4에 있습니다.

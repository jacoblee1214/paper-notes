# 02. 선행 연구와 가까운 대안

## 1. 계보: 이 논문이 서 있는 자리

GazeVLA는 세 줄기의 결합입니다.

1. **사람 1인칭 데이터를 로봇 학습에 쓰는 흐름.** 중간 표현(키포인트, 어포던스, 궤적)을 뽑는 방식, 사람과 로봇을 한 정책의 두 "몸"으로 공동 학습하는 방식(EgoMimic, EgoVLA, H-RDT, EgoScale, π0.5의 사람 데이터 전이), 잠재 행동 공간으로 맞추는 방식(LAPA, UniVLA), 세계모델 방식으로 갈라집니다(논문 1–2절).
2. **flow matching VLA와 CoT.** π0 계열의 PaliGemma + 행동 전문가에 중간 신호 예측(하위 과제, 키포인트 등)을 끼워 넣는 흐름(2절 "VLA", "Mid-Level Representation").
3. **시선(gaze)을 정책 입력·감독으로 쓰는 흐름.** AV-ALOHA(능동 시각), LFA(Look, Focus, Act: 시선 기반 포비에이티드 비전 트랜스포머)는 로봇 원격조종자의 시선을 쓰는 시뮬레이션 기준선이며, 이 논문의 시뮬레이션 벤치마크이기도 합니다(4.2절 [84, 85]).

이 논문의 새로운 점은 (a) 시선을 **출력 토큰**으로 두어 행동 생성 앞의 CoT 단계로 만든 것, (b) 그 시선을 로봇 쪽 라벨 없이 사람 데이터에서만 배우게 한 것(사후학습 로봇 데이터에는 의도 라벨 없음, 3.3절)입니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 이 논문에서의 사용 |
|---|---|---|
| VLM 백본 | PaliGemma(SigLIP + Gemma-2B) | 224×224 영상 한 장, 본문 "PaliGemma", 부록 "PaliGemma-2" |
| 행동 전문가, flow matching | π0 / π0.5 | 공개 코드는 openpi 기반(`04`) |
| 시선 토큰화 | 행동 토큰화 관행(OpenVLA, RT-2의 빈 방식) | 224 bin, 시선 두 좌표를 어휘 끝 224개 토큰으로 사상(`04` D5) |
| 사람 행동 표현 | MANO, 6D 회전, EgoMimic의 카메라 기준 좌표 정렬 | 48차원: 손끝 5점 + 손목 위치·6D 회전 × 양손 |
| 손 복원 | HaWoR(과제 맞춤 사람 데이터) | 부록 A.1 |
| 클립 분할 | Qwen2.5-VL-7B | 부록 A.2 |
| 데이터 | EgoDex, Ego4D, HOT3D 외 10개 | 합계 약 1.5억 프레임, 시선이 없는 데이터는 마스크 |
| 시뮬레이션 | AV-ALOHA 벤치마크와 LFA 기준선 | 6과제, ID와 OOD 두 가지 |
| 기준선 | EgoMimic(ACT 기반), H-RDT, π0.5 | 4.3절 |
| 코드 | openpi, GraspVLA | 공개 저장소 감사 문구 |

## 4. GitHub와 코드 자료 (확인 2026-11-26)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (GazeVLA)** | https://github.com/lichy2004/GazeVLA | LICENSE 파일 Apache-2.0(GitHub API). README 배지는 MIT, README 본문은 CC BY-NC 4.0 | 별 47, 포크 1, 커밋 16, 마지막 푸시 2026-09-08, 커밋 `0a834e8748b8d799166712a098fff91263109c43` | `repo/` 얕은 clone, 정적 읽기만. 학습 코드(AV-ALOHA 설정 하나). 체크포인트·평가 스크립트 없음(README TODO) |
| 공식 데이터 처리 (GazeVLA-Data) | https://github.com/lichy2004/GazeVLA-Data | GitHub API NOASSERTION(LICENSE 파일은 CC BY-NC 4.0 문구), README 배지는 MIT | 별 6, 포크 2, 마지막 푸시 2026-09-08, 커밋 `e47f5dc2cadf8b0d5b7566a410fb0cd9c66e15a1` | `data/GazeVLA-Data/`에 얕은 clone, 정적 읽기만. 13개 중 EgoDex, HOT3D 처리 완료, 나머지는 TODO |
| 프로젝트 페이지 | https://gazevla.github.io/ | 페이지 하단 CC BY-SA 4.0 표기 | - | 영상(YouTube), 코드, 데이터 링크 |
| 프로젝트 페이지 저장소 | https://github.com/GazeVLA/GazeVLA.github.io | 없음 | 별 0, 푸시 2026-09-16 | 정적 사이트 |
| AV-ALOHA / LFA | https://github.com/Soltanilara/giava | MIT | 별 38, 푸시 2026-11-08 | 시뮬레이션 데이터와 기준선(README: 평가 전 이 저장소를 받아 수정해야 함) |
| openpi | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 별 14,766, 푸시 2026-08-24 | 공식 코드가 이 코드의 파생 |
| GraspVLA | https://github.com/PKU-EPIC/GraspVLA | 없음(미명시) | 별 435, 푸시 2026-08-19 | 공식 코드 감사 문구에 등장 |
| EgoMimic | https://github.com/SimarKareer/EgoMimic | MIT | 별 231, 푸시 2024-11-10 | 실로봇 기준선의 원 구현 |
| H-RDT | https://github.com/HongzheBi/H_RDT | MPL-2.0 | 별 162, 푸시 2025-12-21 | 기준선 HRDT |
| 가중치, 평가 스크립트, 사전학습 설정 | 공개되지 않음 | - | - | README TODO: 체크포인트와 평가 스크립트 미체크 |

별과 푸시 날짜는 GitHub API 값이며 시간이 지나면 달라집니다.

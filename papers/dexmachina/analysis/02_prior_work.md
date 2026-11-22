# 02. 선행연구와 계보

번호 [n]은 논문 참고문헌 번호입니다. 공식 저장소와 프로젝트 페이지는 2026-11-22에 GitHub API와 웹으로 확인했습니다. 나머지 선행연구는 따로 열어 검증하지 않았고, 논문이 적은 내용만 옮겼습니다.

## 1. 계보도

```
[시연 한 개로 물체 추적]
  ObjDex [35] (Chen et al., 객체 중심 손 조작, 2024)
      ├─ 고수준 손목 계획기 + 저수준 RL, 혼합 행동 (DexMachina 기준선 2)
      └─ 저자들은 원 논문 수치를 재구현으로 넘었다고 보고 (부록 B.3)
  ManipTrans [36] (Li et al., 2025, 잔차 학습)
      └─ 사전학습 모방 + 잔차, 물리 파라미터 커리큘럼 (DexMachina 기준선 4)
  DexMachina [9] (이 논문, 2025-05)
      └─ 가상 물체 제어기 감쇠 커리큘럼 + 모방·접촉 보상 + 혼합 행동 + ADD-AUC 벤치마크

[운동학 리타겟팅과 접촉 기반 손 설계]
  AnyTeleop [3] (kinematic retargeting, 본문 §4.2 인용)
  Open-TeleVision [26] (kinematic retargeting, App. A.1–A.2 인용)
  VideoDex [31] (인터넷 영상), XSkill [32] (교차 구현 기술 발견): 사람 영상을 쓰지만 짧은 과제 위주 (§2)

[물리 보조 커리큘럼 (RL)]
  커리큘럼 RL: [40] 근골격 기술, [41] ArtiGrasp, [36] 물리 파라미터 감쇠
  접촉 전 힘 허용: [42] (privileged action)
  최적화 워밍스타트 계열: [37] Mordatch 2012, [38, 39] Pang 외

[데이터 수집 (사람 손)]
  DexCap [24], ACE [25], Open-TeleVision [26], DOGlove [28]: 원격 조작과 장갑

          ▼ 이 논문
  DexMachina: 시연 한 개 + 가상 힘 커리큘럼 + 손 여러 개 비교
```

**차별점 요약:** 이 논문은 "사람 시연 한 개를 가져와, 각 로봇 손마다 정책을 따로 학습한다"는 설정에서 긴 양손 관절 물체 과제를 다룹니다. 기존 방식 중 ObjDex는 짧은 과제 위주이고, ManipTrans는 물리 파라미터만 줄이므로 긴 과제에서 불안정하다고 저자들은 주장합니다(§5.2). 두 방식 모두 이 논문의 기준선으로 다시 구현했습니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 이 논문의 변경 |
|---|---|---|
| 혼합 행동 (손목 잔차 + 손가락 절대) | ObjDex [35] | 손목 기준 자세는 운동학 리타겟팅 결과를 쓰고, 한계를 손목 움직임 범위에 맞춤 (App. A.3, Fig. 8) |
| 과제 보상의 곱셈 형태 (위치·회전·관절) | ObjDex [35] | 같은 형태 (§4.1) |
| 물체 쪽 가상 제어기 | 이 논문 (가상 PD 관절 7개) | 제어기 이득을 학습 진행에 따라 줄임 (§4.3, Alg. 1) |
| 물리 파라미터 커리큘럼 (중력, 마찰, 허용 오차) | ManipTrans [36] | 기준선으로 재구현, 지수 스케줄 사용 (App. B.3). 저자들은 값 범위를 "정확히 공개하지 않는다"고 적음 |
| 모방 키포인트 보상 | 이 논문 | 손 링크 키포인트와 기준 위치 (§4.2) |
| 접촉 보상 (접촉 근사 + 매칭) | 이 논문 (ObjDex는 과제 보상만, §5) | 거리 임계 γ = 0.01로 접촉 근사, N_c = 50 (App. A.4) |
| 운동학 리타겟팅 | AnyTeleop [3] 또는 Open-TeleVision [26] (논문 안에서 불일치) | dex_retargeting 패키지를 씀(04 D9) |
| 물리 엔진과 PPO | Genesis [45], rl-games [47] | 기준선도 같은 엔진으로 옮김. ObjDex와 ManipTrans의 원래 설정은 Isaac Gym (App. B.3) |
| 지표 ADD-AUC | 객체 자세 추적 [55–57] (FoundationPose, PoseCNN, 모델 기반 자세 추정) | 물체 부품별 ADD를 쓰고 AUC를 계산 (App. B.4) |

## 4. 공식 코드, 프로젝트 페이지, 관련 저장소

확인일 2026-11-22. 스타와 마지막 push는 GitHub API 값입니다.

| 이름 | 종류 | 라이선스 | 스타 | 마지막 push | 확인 방법 |
|---|---|---|---|---|---|
| [MandiZhao/dexmachina](https://github.com/MandiZhao/dexmachina) | **공식 코드** | MIT | 283 | 2026-03-23 | 얕은 clone, HEAD `adae5bf` (2026-03-22 커밋 "eval instructions"). `repo/`에 정적 읽기용으로만 둠. 실행, 설치 안 함 |
| [project-dexmachina.github.io](https://project-dexmachina.github.io) (저장소 `project-dexmachina/project-dexmachina.github.io`) | **프로젝트 페이지** | 표기 없음 | 0 | 2026-10-14 | 페이지 설정 파일 `content.js`에서 "ICML 2026" 표기 확인. 저장소 라이선스 파일 없음 |
| [MandiZhao/dexmachina-docs](https://github.com/MandiZhao/dexmachina-docs) | 공식 문서 | 표기 없음 | 0 | 2025-07-09 | README의 "Code Documentation" 링크 대상. 읽지 않음 |
| [dexmachina-submission/dexmachina-submission.github.io](https://github.com/dexmachina-submission/dexmachina-submission.github.io) | 제출용 사이트 | 표기 없음 | 0 | 2026-03-03 | 본문 각주 "submission website"가 가리킴. 이 저장소의 index.html은 학회 표기 없음 |
| [MandiZhao/Genesis](https://github.com/MandiZhao/Genesis) | 의존성 포크 | Apache-2.0 | 0 | 2026-01-22 | 공식 코드 README가 설치하라고 함. 설치 안 함 |
| [MandiZhao/rl_games](https://github.com/MandiZhao/rl_games) | 의존성 포크 | MIT | 0 | 2025-07-23 | 공식 코드 README가 설치하라고 함. 설치 안 함 |
| [Dexterous-Hand-Manipulation-Benchmarks/dexhand-bench](https://github.com/Dexterous-Hand-Manipulation-Benchmarks/dexhand-bench) | 제3자 벤치마크 | MIT | 0 | 2026-11-21 | DexMachina 과제에서 FlashSAC, FastTD3, REPPO, PPO를 비교한다고 설명. 읽지 않음 |
| [HlnnnWnng/dexmachina_human_policy](https://github.com/HlnnnWnng/dexmachina_human_policy) | 제3자 포크 | MIT | 0 | 2025-10-22 | 설명 없음. 읽지 않음 |

- **원 논문의 데이터:** 집계 파일이 가리키는 per-run 데이터(체크포인트와 평가 `.npy`)는 README의 Google Drive 링크에 있습니다. 용량과 내용은 확인하지 않았습니다(받지 않음). 이 데이터를 받으면 seed별 값을 얻을 수 있습니다(03 §8).
- **ARCTIC 데이터:** 저장소에 없습니다. 별도 다운로드가 필요하고 라이선스는 확인하지 않았습니다.
- **복제본 크기:** `repo/`의 작업 트리는 약 366 MB, `.git`은 144 MB입니다. 대부분 손 자산 메시(STL, GLB, OBJ)입니다.

## 5. 주의

- 학회 표기는 여러 곳에서 확인했습니다(ICML 2026 공식 포스터 목록, 프로젝트 페이지 설정, CHORD 분석의 인용). 이 논문의 arXiv는 v1 하나이므로 채택본 PDF와 본문은 다를 수 있습니다.
- 이 논문의 기준선 재구현은 저자들이 ObjDex 저자와 연락해 설정을 받았다고 적습니다(App. B.3). 원 논문 수치(41.2%, 57.6%)는 확인하지 않았습니다.

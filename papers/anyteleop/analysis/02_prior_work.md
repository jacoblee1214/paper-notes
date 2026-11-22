# 02. 선행연구와 관련 자료

## 1. 계보

AnyTeleop은 세 흐름이 만나는 곳에 있습니다. 첫째는 **영상 기반 손 원격조작**입니다. 손 추적 신경망이나 최적화 기반 리타게팅으로 사람 손을 로봇 손에 옮깁니다. DexPilot [16]은 한 로봇 팔-손(Kuka + Allegro)을 위해 손가락 끝 벡터를 맞추는 최적화를 쓰고, RMP로 팔을 움직였습니다. Telekinesis [54]는 XArm6 + Allegro 실로봇 과제 10개를 만들었고, AnyTeleop은 이 과제를 그대로 재현해 비교합니다. Qin et al. [43]("From one hand to multiple hands")은 RGB-D 한 대로 시뮬에서 원격조작 시연을 모아 DAPG로 학습하는 시스템으로, 시뮬 비교의 기준선입니다. DexMV [42]는 사람 영상에서 시연을 얻는 선행 작업입니다.

둘째는 **손 자세 추정과 모션 생성 부품**입니다. 손가락 키포인트에 MediaPipe [64], 손 형상과 손목 추정에 FrankMocap [50]과 SMPL-X [40]를 씁니다. 팔의 충돌 회피 궤적에는 RMP [7](DexPilot의 방식)와 GPU 가속 CuRobo [56](AnyTeleop의 방식)가 있습니다.

셋째는 **다중 현실과 원격 시각화**입니다. SAPIEN [62], Isaac Gym [32], PyBullet [8]을 한 시스템에서 쓰는 데 초점을 둡니다. HAPTIX [23]는 MuJoCo 환경에만 묶인 모션캡처 원격조작으로 언급됩니다. 브라우저 뷰어는 meshcat [10]과 Three.js [9]에 기대고, 시뮬 기준 원격조작과 크라우드소싱 원격조작 [33], 다중 팔 협업 [59]도 같은 맥락입니다.

논문 표 I의 9개 비교 시스템(DexPilot [16], Holo-Dex [4], DIME [5], TeachNet [24], Telekinesis [54], Qin et al. [43], MVP-Real [45], Transteleop [26], Mosbach et al. [37])은 이 세 축으로 나뉩니다. 이 표 자체의 셀은 원자료로 대조하지 않았습니다(03 C5).

## 2. 가까운 대안과 공식 코드

| 자료 | 역할 | 라이선스 | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting) | 리타게팅 라이브러리, "AnyTeleop 프로젝트에서 시작" | MIT (LICENSE 있음) | 1321 | 2025-08-21 | 공식 (프로젝트 페이지의 "Retargeting Code" 링크). 얕은 복제 `repo/dex-retargeting` (`3f56141`) |
| [NVlabs/sim-web-visualizer](https://github.com/NVlabs/sim-web-visualizer) | 웹 시뮬 뷰어 (meshcat), "AnyTeleop 프로젝트의 일부" | MIT (LICENSE 있음) | 429 | 2024-05-26 | 공식 (프로젝트 페이지의 "Web Visualizer Code" 링크). 얕은 복제 `repo/sim-web-visualizer` (`20b02f3`) |
| 프로젝트 페이지 `yzqin.github.io/anyteleop/` | 논문 주소 (v3) | — | — | — | GitHub Pages 404. 보관본 2025-11-18 존재 |
| `anyteleop.com` | 논문 부록 A 주소 | — | — | — | 무관한 콘텐츠 사이트. 공식 아님 |
| RalphFH/AnyTeleop | 비공식 구현 (단안 RGB) | MIT | 17 | 2026-09-30 | 복제하지 않음 |
| Telekinesis 저장소 [54] | 실로봇 기준선 | — | — | — | 기준선 수치만 옮김. 저장소 확인 안 함 |
| DexPilot 저장소 [16] | 원조 비교 대상 | — | — | — | 확인 안 함 |

- 서버 본체(검출, 융합, CuRobo 동작 생성, 클라이언트)는 공개 저장소가 없습니다. 자세한 대조는 04 §2–§3입니다.
- 두 공식 저장소의 README에는 AnyTeleop BibTeX(`qin2023anyteleop`)가 있습니다.

## 4. 참고문헌의 확인 상태

- 참고문헌은 68개입니다. 이 문서에서 인용한 번호는 `paper/paper.txt`의 참고문헌 목록에서 확인했습니다. 인용된 개별 논문 원문은 받지 않았습니다.
- 그림 10의 손 이름(Schunk SVH, Shadow, DLR, Allegro)은 Fig. 10 캡션에서 옮겼습니다.

# DiffWAM 선행연구와 관련 자료

## 1. 학회와 버전

**학회 표기 없음 (arXiv).** 확인한 곳과 결과는 다음과 같습니다.

| 확인한 곳 | 결과 |
|---|---|
| arXiv abs (2611.19904) | 학회, 저널 참조, 코멘트 항목 없음. v1만 있음 |
| 원문 PDF | 학회 양식이나 채택 문구 없음. 첫 쪽 상단에 "November 2026" |
| 프로젝트 페이지 (zgca-hmi-lab.github.io/DiffWAM) | 학회 표기 없음. BibTeX는 `@article`, 2026 |
| 공식 GitHub README (2026-11-17 커밋) | "Code coming soon." 학회 표기 없음 |
| CoRL 2026 공식 채택 목록 (`conferences/corl2026/papers.json`, 755항목) | 제목과 arXiv 번호 모두 일치 항목 없음 |
| 웹 검색 (제목 기준, 2026-11-22) | 일치하는 자료 없음 |

제출 후 7일이고 다른 학회의 채택 목록을 모두 확인하지는 못했습니다. 따라서 학회 칸은 "학회 표기 없음 (arXiv)"으로 둡니다. 나중에 채택이 확인되면 갱신합니다.

## 2. 공식 코드와 관련 저장소

| 저장소 | 관계 | 라이선스 | 최근 push | 별 | 상태 |
|---|---|---|---|---|---|
| [ZGCA-HMI-Lab/DiffWAM](https://github.com/ZGCA-HMI-Lab/DiffWAM) | 논문 공식 저장소 (저자 소속 조직) | 없음 (API license null, LICENSE 파일 없음) | 2026-11-17 (커밋 7fc02f7, "Update README.md"); 생성 2026-10-22 | 0 | README 한 쪽과 그림 한 장(Teaser). README에 "Code coming soon." 코드 없음. 얕은 복제(`repo/`)로 정적 확인만 함 |
| [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) | 기준선 FastWAM의 공식 코드 | MIT | 커밋 7faa711 (2026-08-20) | 1665 (2026-11-20 확인) | 이 논문의 RoboTwin·LIBERO 기준선 (arXiv:2603.16666) |
| [robbyant/lingbot-va](https://github.com/robbyant/lingbot-va) | 표 1, 2, 12 비교 방법 | Apache-2.0 | 2026-07-09 | 1988 (2026-11-20 확인) | 비교 값만 사용 |
| Motus, WLA, ImageWAM, X-WAM, GigaWorld-Policy, AHA-WAM, StarVLA, ABot-M0, QwenVLA | 비교 표의 값 | 확인 안 함 | 확인 안 함 | 확인 안 함 | 논문 표 값만 인용. 저장소는 확인하지 않았음 |

**결론.** DiffWAM 공식 코드는 아직 없습니다. 따라서 논문 대 코드 대조(D1–Dn)는 할 수 없습니다. `04_claim_verification.md`에 주장 대조(C1–Cn)를 썼습니다. 저장소 복제는 README와 이미지 한 장뿐이므로 `repo/`는 참고용입니다. 실행, 설치, import는 하지 않았습니다. 복제 내용은 신뢰하지 않는 자료로 취급했습니다.

## 3. 선행연구 계보

논문 §2의 분류를 따릅니다.

**(가) VLA.** 관측과 언어에서 행동을 바로 냅니다(식 1). RT-2, OpenVLA, π0, Octo, Diffusion Policy, RDT-1B, Xiaomi-Robotics-0이 여기 속합니다. 이 논문의 표 1, 2에서 비교 대상은 OpenVLA, OpenVLA-O, MemoryVLA, π0, π0.5, GR00T-N1, X-VLA, ABot-M0, QwenVLA, StarVLA입니다.

**(나) 미래 예측 WAM.** 미래 프레임과 행동을 함께 모델링합니다(식 2). Motus, LingBot-VA, Fast-WAM, GigaWorld-Policy, IRASim, VideoVLA, Video Prediction Policy, WorldVLA가 여기 있습니다. 가장 가까운 기준선은 **FastWAM**입니다. 논문 §3.2의 구조화 마스크 (a)항은 FastWAM의 첫 프레임 조건 어텐션을 그대로 따른다고 밝힙니다. Fast-WAM 논문(arXiv:2603.16666)의 표 값으로 다시 계산하면 FastWAM에서 학습 중 비디오 공동학습을 빼면 RoboTwin 평균 약 8 pp, LIBERO 약 4 pp가 떨어집니다. 반면 추론 때 미래 장면을 상상하는 단계를 빼는 것(Joint 대 Fast-WAM)의 차이는 작았습니다(LIBERO 0.9 pp, RoboTwin은 과제 단위에서 Joint가 더 많이 이김). 즉 미래 예측이 학습 신호로 쓰일 때 효과가 크다는 쪽이 근거가 더 강합니다. DiffWAM은 이 신호에 차이 목적을 더한 설계입니다(`04` C13).

**(다) 구조화 신호 예측 WAM (WAM-s).** 깊이, 분할, 광학 흐름, 사전학습 특징을 보조 목표로 씁니다(식 3). X-WAM, AIM, AGRA, DeFI, MoLA가 여기 있습니다. DiffWAM은 외부 추정기 없이 잠재 차이를 씁니다. 저자의 주장이 맞으려면 잠재 차이가 로봇 행동 신호를 실제로 담아야 하는데, 그 증거는 표 7과 표 11 정도입니다(`03` §3).

**(라) 근접한 동시대 연구.** 같은 "차이 예측" 아이디어를 가진 WAM 논문이 제출 전후로 나왔다는 것을 웹 검색으로 확인했습니다(DC-WAM, DeltaWAM, CoRe-WAM, DeltaWorld/DeltaTok 등, 2026년 공개). 이들의 실험 프로토콜은 확인하지 않았습니다. 이 분석의 비교 대상은 DiffWAM 논문 안의 표로 한정합니다.

## 4. 이 논문의 기준선 값 출처

| 기준선 | 이 논문 값 (재현 또는 인용) | 분석 폴더가 적은 원 보고 값 | 차이 |
|---|---|---|---|
| FastWAM, LIBERO 평균 | 97.05 (†, 재현) | 97.6 (arXiv:2603.16666 보고 값) | −0.55 |
| FastWAM, RoboTwin 2.0 평균 | 91.10 (†, 재현) | 91.8 (arXiv:2603.16666 보고 값) | −0.70 |

기준선 값이 원 보고값과 다르면 DiffWAM의 이득 해석이 달라집니다. 예를 들어 RoboTwin에서 원 보고값 91.8과 비교하면 DiffWAM-L의 이득은 +1.55가 아니라 +0.85입니다(`04` C2). 기준선 재현의 학습 설정과 초기 체크포인트는 논문에 충분히 적혀 있지 않습니다.


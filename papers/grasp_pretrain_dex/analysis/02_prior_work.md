# 02. 선행연구, 공식 저장소, 학회, 이 저장소의 관련 분석

## 1. 공식 코드와 프로젝트 (GitHub 표)

GitHub API로 확인한 값입니다(2026-11-22). 두 저장소 모두 **라이선스 파일이 없습니다**(API의 license 값이 null). 저장소를 얇게 복제해 `repo/` 아래 두고 정적으로만 읽었습니다. 실행, import, 설치는 하지 않았습니다.

| 저장소 | 내용 | 라이선스 | ★ | 마지막 push | 확인 커밋 (복제본) |
|---|---|---|---|---|---|
| [xinyu1Liu/G2D-Policy](https://github.com/xinyu1Liu/G2D-Policy) | 고수준·저수준 정책, Shadow-LEAP 어댑터, 학습·평가 스크립트, 설정 파일 | 없음 | 3 | 2026-11-03 | `80c204a` (`repo/G2D-Policy`) |
| [YingYuan0414/DexCraft](https://github.com/YingYuan0414/DexCraft) | 6개 DexCraft 과제(ManiSkill3), 키보드 텔레오퍼레이션, MimicGen 파이프라인, Dexonomy 궤적 생성기 | 없음 | 2 | 2026-10-06 | `1bed0cd` (`repo/DexCraft`) |
| [프로젝트 페이지](https://yingyuan0414.github.io/grasp2dexterity/) | 영상, 저자 링크, arXiv 링크, DexCraft 링크 | - | - | - | 2026-11-22 확인 |
| [HF 데이터셋 YingYuan0414/DexCraft](https://huggingface.co/datasets/YingYuan0414/DexCraft) | DexCraft 시연(`demos/`)과 MimicGen 생성분(`generated/`). README 기준 | 미확인 | - | - | 내려받지 않음 |
| [DexLearn](https://github.com/JYChen18/DexLearn) (Dexonomy 자산) | G2D-Pretrain 재생성에 필요한 Dexonomy 파지와 물체 자산. README가 가리킴 | 미확인 | - | - | 복제하지 않음 |

- 두 저장소 모두 2026년 가을에 만들어졌고 별이 적습니다. 공식 저장소라고 보기에는 문서가 충분하고 실제 코드가 들어 있습니다.
- 라이선스가 없어서 코드를 게시본에 옮기지 않습니다. 게시본은 링크만 적습니다.
- 복제본의 파일 목록과 대조 결과는 `04_code_verification.md`에 있습니다.

## 2. 선행연구 계보와 구성 요소별 출처

| 구성 요소 | 논문이 쓰는 출처 | 논문에서의 역할 | 비고 |
|---|---|---|---|
| 파지 데이터 | Dexonomy [2] (RSS 2025), GraspXL [3] (ECCV 2024), DexGraspNet [1], Dex1B [4] (RSS 2025) | Dexonomy는 사전학습 원천, GraspXL은 비교 사전학습 원천 | Dexonomy의 31 grasp type이 핵심 (§5.2) |
| 계층 정책 | ArticuBot [11], GHOST [12] (RSS 2026) | 4점 부목표 구조의 원형 | 논문은 이 4점이 손에 부족하다고 주장 |
| 저수준 정책 | DP [40], DP3 [41] | 확산 정책과 3D 점군 인코더 기반 | 논문은 DP3를 수정 |
| 데이터 생성 | MimicGen [38] | 키보드 시연 5개를 100개로 늘림 | DexCraft 생성 파이프라인 |
| 시뮬레이터 | ManiSkill3 [43] (RSS 2025), PartNet-Mobility [45–47], LEAP 손 [44] | 과제 환경과 자산 | 손은 xArm6 + LEAP |
| VLA 비교 | UniDex-VLA [54] (2026) | 동일 계열 손 VLA 기준선 | 논문 v2에서 추가 |
| 원격조작 보정 | DexGen [33] (DexterityGen) | 원격조작 명령을 다듬는 사전학습 제어기 | 논문은 자신과 방향이 다르다고 씀(부록 2절 관련 연구) |
| 입력 표현 | PointNet++ [48], Act3D [55] | 고수준 백본, 저수준 교차 주의 | 부록 B.2 |

- **계보의 핵심:** 파지 합성 데이터(DexGraspNet, Dexonomy)를 폐루프 파지 정책(논문 [4, 7, 8])에 쓰던 흐름에서, 관절 도구 사용이라는 더 긴 과제로 옮긴 설정입니다. 계층 구조는 평행 집게용 부목표(ArticuBot, GHOST)를 손 16점으로 바꾼 것입니다.
- **차별점:** (a) 부목표 표현을 4점에서 16점 키포인트로 바꾸고, (b) 사전학습 데이터를 파지 주석에서 만든 3단계 궤적으로 바꿨습니다. 두 요소 모두 기존 요소의 조합이라 새로움은 제한적입니다. 손 자유도가 높은 경우의 근거는 시뮬레이션 절제로 보였습니다(표 1, 부록 D.1 Cond-5 대 Cond-16은 그림 10이라 수치 표는 없습니다).
- **가장 가까운 대안:** GraspXL 사전학습(같은 구조, 다른 데이터)입니다. 표 1에서 GraspXL-PT는 18개 셀 모두에서 Ours보다 낮습니다. 다만 두 사전학습의 전이 방식이 다르므로(`03` C12) 순수한 데이터 효과로 읽기는 어렵습니다.

## 4. 학회 확인

- **arXiv 주석 (v2):** "10th Conference on Robot Learning (CoRL 2026), Austin TX, USA." (PDF 첫 쪽에도 같은 줄이 있습니다.)
- **CoRL 2026 공식 채택 목록** (`conferences/corl2026/papers.json`, 제목 정확 일치): `presentation: poster`, 세션 "Poster post-p2", 2026-11-10(화) 5:00–6:30 PM, 포스터 번호 124, `in_official: true`.
- **프로젝트 페이지:** BibTeX가 "Conference on Robot Learning (CoRL), 2026"이라고 씁니다.
- **판정:** 공식 목록으로 확인했으므로 `venue: "CoRL 2026"`, `venue_note: "포스터 (corl.org 공식 채택 목록)"`입니다.
- **주의:** 저장소 README의 BibTeX 항목에는 "To appear"가 있는 것도 있습니다. 공식 목록이 있으므로 이 표기는 참고만 합니다.

## 5. 인용 기준

본문과 게시본에서 인용할 때 쓰는 기준입니다.

- 논문 내 인용 번호는 v2 참고문헌 목록 기준입니다([1]–[55]).
- v2에 새로 들어온 인용은 [54] UniDex-VLA와 [4] Dex1B의 비교 설정입니다. v1에는 없습니다.
- GHOST [12]는 RSS 2026으로 적혀 있습니다. 확인하지 못했습니다(v2 참고문헌 기준으로만 인용).

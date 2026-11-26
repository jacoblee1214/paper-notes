# TopoRetarget 선행연구와 공개 자료

## 1. 계보

TopoRetarget은 **상호작용 메시(interaction mesh)** 계열 리타게팅을 손-물체 조작으로 옮긴 것입니다. 논문이 직접 가까운 선행으로 드는 것은 OmniRetarget [27]이고, 그 위에 손 중심 리타게팅, 접촉 인식 리타게팅, 참조 기반 RL 추적이 놓입니다. 아래 번호는 논문 참고문헌 번호입니다.

| 계보 | 대표 작업 | TopoRetarget과의 관계 |
|---|---|---|
| 상호작용 메시와 라플라시안 변형 | 캐릭터 모션의 공간 관계 보존(Ho, Komura, Tai, SIGGRAPH 2010, 논문에는 인용 없음, 분석자 지식), OmniRetarget [27] (휴머노이드 전신 조작·장면 상호작용, 접촉 인식 메시 변형) | 핵심 에너지(식 7)의 원형. OmniRetarget의 메시 목적식과 물체 표면 샘플링을 유지하고 휴머노이드 키포인트를 MediaPipe 손 키포인트로 바꾼 것을 기준선으로 씁니다 (A.2). TopoRetarget의 차이는 거리 가중 정규화(식 5), 뼈 방향 초기화·보조 항(식 1, 2), 슬랙을 둔 관통 제약(식 8)이라고 설명합니다 |
| 손 중심 리타게팅 | DexPilot [8], AnyTeleop [9], 목적함수 분석 [12](dex-retargeting 계열), GeoRT [13] | 손 자세, 손가락 끝 대응, 핀치 관계 등 **손 기준 목적**. "어디서 어떻게 물체에 닿을지"를 직접 정하지 않는다는 것이 출발점 (§1) |
| 접촉 인식 리타게팅·파지 전이 | DexFlow [15], FunGrasp [16], GenHand [18] | 파지·접촉 전이는 개선하나 접촉 모드가 자주 바뀌는 조작에서는 변화하는 관계를 보존하는 참조가 필요하다고 서술 (§2.1) |
| 물리 기반·잔차 학습 전이 | SPIDER [17], ManipTrans [7], DexMachina [6], ObjDex [14] | 같은 문제를 물리 시뮬레이션과 샘플링·RL로 풀어 실행 가능한 궤적을 만듭니다. TopoRetarget은 **물리 없이 기하 최적화만으로** 참조를 만들고 RL은 그 뒤에 따로 붙입니다 |
| 참조 기반 RL 추적 | DeepMimic [21], OmniH2O [22], DexTrack [4], Dexplore [5] | 4절의 최소 추적 제어기(PPO, 잔차 행동, 참조 상태 초기화)가 이 계열을 따릅니다 |
| 데이터 원천 | DexMV [23], Robotic Telekinesis [10], DexCap [24], DexMimicGen [25], ContactPose [26], Ho-cap [31] | 입력 사람 시연. ContactPose와 Ho-cap이 평가에 쓰임 |
| 펜 돌리기 | [3], [20] (펜 돌리기 시스템) | 평가 과제 (Pen-Spin) |

## 2. 공식 코드와 관련 저장소 (2026-11-26, `gh api`)

| 저장소 | 관계 | 라이선스 | 최근 push | 별 | 상태 |
|---|---|---|---|---|---|
| [wuji-technology/toporetarget](https://github.com/wuji-technology/toporetarget) | **공식 코드.** 프로젝트 페이지 Code 단추, README에 논문 링크와 저자 8명 명단 | MIT | 2026-10-26 | 53 | 커밋 `039c54022016a882ef8d3a6d34b4578aee7cb51f`(2026-10-26 "docs(readme): Updates the project page and paper links")를 `repo/`에 얕은 clone(서브모듈 제외). 커밋 3개, 첫 공개 `9c4c33e`(v0.1.0, 2026-10-23). 풀이기(`solver/`), 가져오기(GRAB, TACO, OakInk2, HOT3D, 캡처 묶음), 예제 클립 `pen_spin_029`, 시험, wuji-mjlab용 변환기. 실행하지 않음 |
| [wuji-technology/wuji-mjlab](https://github.com/wuji-technology/wuji-mjlab) | 공식 RL 훈련·배포 저장소(공식 README가 "RL training and deployment"로 연결). 펜 돌리기 과제 `pen_spin`, 체크포인트 5세트, 클립 40개(5그룹) | Apache-2.0 | 2026-11-17 | 277 | 릴리스 v2026.10.22(`b2b98d7054a8d1b0f5cf295d5acaebc4358bc6a9`)의 과제 설정 파일 12개만 `data/wuji_mjlab_b2b98d7/`에 받아 읽음(설정 파일 전체 크기 약 107 KB). 설치·실행·가중치는 받지 않음 |
| [wuji-technology/wuji-description](https://github.com/wuji-technology/wuji-description) | Wuji Hand 2 모델, toporetarget의 서브모듈(`v2026.10.12`) | MIT | 2026-11-18 | 56 | 서브모듈이라 clone하지 않음 |
| [TopoRetarget2026/TopoRetarget](https://github.com/TopoRetarget2026/TopoRetarget) | 프로젝트 페이지 파일(`toporetarget2026.github.io`) | 없음 | 2026-10-23 | 2 | 웹사이트 파일. 설명에 "Interaction-Centric Retargeting … (CoRL 2026 submission)" |
| [Tsinghua-MARS-Lab/toporetarget-web](https://github.com/Tsinghua-MARS-Lab/toporetarget-web) | 같은 프로젝트 페이지의 연구실 사본 | 없음 | 2026-10-06 | 0 | 확인만 |
| [0iui0/toporetarget](https://github.com/0iui0/toporetarget) | 비공식 구현("Unofficial implementation", arXiv v2 기준) | 없음 | 2026-06-23 | 22 | 확인만. 공식 저장소와 무관 |
| [Key-Zzs/TopoRetarget-Repro](https://github.com/Key-Zzs/TopoRetarget-Repro) | 비공식 재현("paper-traceable reproduction"), Arti-MANO와 임의 손으로 리타게팅 | GPL-3.0 | 2026-10-23 | 2 | 확인만 |
| [yijie21/toporetarget_reimp](https://github.com/yijie21/toporetarget_reimp) | 비공식 재구현, MANO에서 Wuji 손으로 보는 뷰어 | MIT | 2026-09-15 | 0 | 확인만 |
| [Hydrofoooil/TopoRetarget](https://github.com/Hydrofoooil/TopoRetarget) | 비공식 구현 | 없음 | 2026-08-15 | 0 | 확인만 |
| [kevinzakka/mink](https://github.com/kevinzakka/mink) | 기준선 Mink [30] | Apache-2.0 | 2026-11-19 | 1,624 | 확인만 |
| [amazon-far/holosoma](https://github.com/amazon-far/holosoma) | OmniRetarget 계열 리타게팅 포함(논문의 OmniRetarget과 동일성은 미확인) | Apache-2.0 | 2026-11-26 | 1,740 | 확인만 |
| [facebookresearch/GeoRT](https://github.com/facebookresearch/GeoRT) | 기준선 GeoRT [13] | NOASSERTION | 2025-09-02 | 241 | 확인만 |
| [dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting) | 손 중심 리타게팅 [12], 공식 코드가 이 저장소의 `landmarks.py` 일부를 가져옴(THIRD_PARTY_NOTICES) | MIT | 2025-08-21 | 1,322 | 확인만 |
| [facebookresearch/ContactPose](https://github.com/facebookresearch/ContactPose) | 표 1의 평가 데이터 [26] | MIT(코드. 데이터 이용 조건은 별도 확인 필요) | 2025-05-07 | 440 | 확인만. 데이터는 받지 않음 |

- 공식 코드의 서브모듈(`wuji-description`)은 얕은 clone에 포함하지 않았습니다. 그래서 `repo/toporetarget/assets/wuji-description/`는 비어 있고, 손 모델 파일은 읽지 못했습니다.
- 프로젝트 페이지(`toporetarget2026.github.io`)의 BibTeX는 `@inproceedings`, CoRL 2026입니다.
- 공개된 릴리스 첨부 `toporetarget-assets.zip`은 6,291,571 바이트(GitHub API 메타데이터), 펜 돌리기 원시 캡처 묶음이며 MIT입니다. 받지 않았습니다.

## 4. 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| Delaunay 사면체 분할 연결 | 상호작용 메시 (Ho 외 2010, OmniRetarget [27]) | 공개 코드는 SciPy `Delaunay`, 간선 길이 문턱은 기본으로 끔(`use_edge_threshold: false`) (`04` §2) |
| 거리 가중 라플라시안 | 상호작용 메시의 균일 가중을 $\exp(-\kappa d)$로 바꿈 | $\kappa$ = 30. 가중치는 사람 쪽 꼭짓점으로 계산해 로봇에 재사용 |
| 뼈 방향 항(식 1, 2) | 저자 정의. 손 키포인트 방향 일치는 dex-retargeting 계열의 아이디어 | 공개 코드의 형태는 논문과 다름 (`04` D2) |
| 슬랙 관통 제약(식 8) | 저자 정의. 슬랙 페널티는 표준 소프트 제약 | 공개 코드는 MuJoCo `mj_geomDistance` 기반 선형화 (`04` D11) |
| QP 풀이 | DAQP(Active-set QP 풀이기) | 논문에 풀이기 언급 없음 (`04` D5) |
| 손목 프레임 추정 | dex-retargeting의 `estimate_frame_from_hand_points` | 공개 코드 THIRD_PARTY_NOTICES에 명시 |
| 추적 RL | PPO [29], 잔차 행동, 참조 상태 초기화 (DeepMimic [21] 계열) | 코드는 mjlab(MuJoCo Warp)와 RSL-RL |
| 손 모델 | Wuji Hand (논문), Wuji Hand 2 (공개 코드) | 논문에 손 버전 표기 없음 (`04` D12) |

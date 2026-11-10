# 02 선행연구, 가까운 대안, 저장소 내 관계, GitHub

## 1. 선행연구 계보

| 계보 | 대표 연구 (논문 참고문헌 번호) | 이 논문과의 관계 |
|---|---|---|
| 데이터 스케일링 | RT-1 [1], RT-2 [2], OXE [3], DROID [4], Octo [5], OpenVLA [6] | 다양성을 "태스크·객체·embodiment" 같은 거친 속성으로 다룸. H_G는 궤적 기하에 한정 |
| 데이터 품질 | Belkhale et al. [12] (행동 발산과 전이 다양성), Mandlekar et al. [15] | 품질 축은 관측, 라벨, 범위. 궤적 기하 축은 다루지 않음 |
| 수집과 큐레이션 | DAgger [13], DART [14], 가중치·필터 [16], Scizor [17], CUPID [18], 온라인 경험 큐레이션 [19] | 선택과 가중이 목적. H_G는 감사(진단) 용도 |
| 데이터 생성 | MimicGen [21], DemoGen [22], ManiBox [23], FieldGen [25], MOVE [26] | 생성으로 커버리지를 넓힘. H_G는 넓힌 결과의 측정 도구로 쓸 수 있음 |
| 다중모드 정책 | GAIL [27], BeT [28], IBC [30], 잠재 행동 생성 [31], BAKU [32] | 다중모드를 정책 쪽에서 모델링. 논문은 이 정책 클래스의 차이를 실험하지 않음 |
| 다양성 평가 | D3IL [33], 점유 측도 매칭 [34], "다양성이 필요한가" [35] | [35]이 "다양성 효과가 변수에 따라 다르다"를 보임. 논문과 같은 질문 |
| 궤적 유사도 | DTW [36] | 쌍별 정렬. H_G는 집단 스펙트럼 |

**가까운 선행 연구 중 이 논문이 비교하지 않은 것.** [35] Shi et al. (`Is diversity all you need`)은 같은 질문에 다른 변수(장면·객체 다양성)로 답한다. 논문은 이 연구를 언급만 하고 H_G와 비교하지 않는다.

## 2. 가까운 대안 (기준선)

| 대안 | 정의 | H_G와 차이 | 이 저장소 검증 |
|---|---|---|---|
| 평균 분산 σ²_avg | 시점별 분산의 평균 | 순서와 상관을 무시 | 논문 Fig. 6 (그림만) |
| LogDet | log det(Σ+αI) | 정규화 α 필요, 낮은 차원에서 조건 나쁨 | 우리 toy 미실시 |
| 참여율 PR | (Σλ)²/Σλ² | 스펙트럼 정보는 쓰지만 단일 요약 | 논문 Fig. 6 |
| kNN KL | Kozachenko–Leonenko | 대역폭 k, D = 150에서 표본 크기 drift | 우리 toy T5 (k에 매우 민감) |
| 공분산 유효 차원만 | exp(H(p)) | 크기 항 없음 | toy T1c에서 d_eff로 분해 |

## 3. 이 저장소의 다른 분석과의 관계

논문의 핵심 질문(데이터의 어떤 속성이 학습에 도움이 되는가)은 이 저장소의 여러 폴더와 맞닿아 있습니다. 아래는 이 논문을 읽을 때 참고할 연결점입니다.

- **`nestdex/` (NestDex, arXiv:2608.13362).** 텔레옵 시연을 수집하는 방식이다. 수집 성공률(Table II, Fisher 검정)과 수집자의 노력이 데이터 품질을 좌우한다. H_G는 수집 품질(성공률)을 재지 않으므로, H_G 감사와 수집 품질 감사는 따로 해야 한다. 논문 Eq. 3이 성공 궤적만 쓰는 것도 같은 선택 편향 문제다(`04` §2의 12번).
- **`chronospolicy/` (Chronos Policy, arXiv:2610.13929).** 공개 코드가 없다("Coming soon", 2026-10-28 확인). 이 논문과 같이 "공개 코드가 없는 IL 논문에서 수치만 재계산하는" 사례다. 교훈: 재계산과 toy로 검증할 수 있는 범위를 먼저 정한다.
- **`do_as_i_do/` (Do as I Do, arXiv:2606.19333).** 일상 영상에서 손 궤적을 복원해 로봇 손으로 리타겟한다. 시연이 사람 영상에서 오므로, H_G의 목표 프레임(R_T)과 전이 구간 분할을 영상에서 얻는 문제가 생긴다(`04` C1). 영상 유래 데이터의 감사 도구로 H_G를 쓸 때 가장 먼저 확인할 조건이다.
- **`beyond_implicit_force/` (Beyond Implicit Force, arXiv:2607.14578, IROS 2026).** ACT를 ALOHA식 원격조작 데이터로 학습한다. 이 논문의 실물 실험(ACT, N = 100, 3 수집자)과 같은 정책 계열이다. 데이터 양과 수집 조건의 통제 방법(`beyond_implicit_force/analysis/04_claim_verification.md`)을 참고할 수 있다.
- **`contact_disassembly/` (CONTACT, arXiv:2603.08560, IROS 2026).** 시뮬 데모 수가 과제마다 90, 30으로 달라졌고, 에포크는 `int(50000/n_demo)`로 자동 계산되어 학습량이 같이 바뀐다(`contact_disassembly/analysis/04_code_verification.md` D1–D2). 이 논문의 N 스윕(100–1000)도 학습 예산을 통제하지 않았다면 같은 교란이 있다(`04` §2의 2번). 이 패턴이 이 논문 검증의 가장 중요한 참고 사례다.
- **`dexforce/` (DexForce, arXiv:2501.10356).** 키네스테틱 시연에서 접촉력으로 행동 목표를 만든다. 데이터의 "품질"을 힘 정보로 정의한다는 점에서 H_G(기하 다양성)와 다른 축이다. 두 축을 같이 감사하면 데이터 큐레이션이 더 정확해질 것이다.

## 4. GitHub 조사 (2026-11-10 확인)

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | 별 | 최근 커밋 (해시) |
|---|---|---|---|---|---|
| (논문 관련 코드) | `gh search repos "geometric entropy imitation learning"`, `"geometric-entropy"`, 저자 3명 계정(`qianqianlo`, `zrllrz`, `YanchaoYang`) 검색 | — | — | — | **찾지 못함.** 관련 없는 "geometric entropy" 저장소만 검색됨 |
| `geometric-entropy/geometric-entropy.github.io` | 프로젝트 페이지(`index.html`, `style.css`, 논문 PDF). 코드 없음 | 공식 (프로젝트 페이지) | 없음 | 0 | `d292168` (2026-11-07) |
| `qianqianlo/qianqianlo.github.io` | 제1저자 개인 사이트 | 저자 개인 | 확인 안 함 | — | 2026-11-07 푸시. 코드 아님 |
| `mani-skill/ManiSkill` | 시뮬레이터(PandaArm 플래너, StackCube-v1, PegInsertionSide-v1) | 논문이 사용 | Apache-2.0 | 3504 | `62ff3a5` (2026-08-02) |
| `real-stanford/diffusion_policy` | Diffusion Policy 구현 | 논문이 사용(정책) | MIT | 4721 | `5ba07ac` (2024-12-24) |
| `tonyzhaozh/act` | ACT 구현 | 논문이 사용(실물) | MIT | 2335 | `742c753` (2024-01-28) |
| `Physical-Intelligence/openpi` | π0.5 구현(미세조정 레시피) | 논문이 사용(VLA) | Apache-2.0 | 14603 | `215abfb` (2026-08-24) |
| `huggingface/lerobot` | SO-100 계열 및 π0.5 PyTorch 포트 | 참고용 | Apache-2.0 | 28790 | `81b8c5b` (2026-11-09) |

판단: 이 논문의 H_G 계산 코드와 시연 생성기는 공개되지 않았습니다. 프로젝트 페이지 저장소는 웹사이트이고 코드가 없습니다. 실험 기반(ManiSkill, DP, ACT, openpi)은 모두 공개되어 있으므로, 실험 재현의 뼈대는 가능합니다. 다만 논문이 쓴 설정(경유점 생성, 시연 수, 학습량)은 우리가 다시 정해야 합니다.

## 5. 저작권과 인용 메모

- 저자 소속은 대학(University of Hong Kong)만 인용합니다. 공개 문서에 기관명을 더 쓰지 않습니다.
- 인용 형식: Luo, Liu, Zhou, Zhou, Yang, "Geometric Entropy: When Trajectory Diversity Helps and Hurts in Imitation Learning", arXiv:2606.20871 (v1), IROS 2026.

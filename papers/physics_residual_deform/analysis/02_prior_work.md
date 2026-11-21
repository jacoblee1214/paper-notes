# 02. 선행연구와 관련 저장소

이 문서는 PGRD의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드와 권리 상태가 어떤지를 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 권리와 공개 범위

- **라이선스**: arXiv abs 페이지의 "view license" 링크는 `http://arxiv.org/licenses/nonexclusive-distrib/1.0/`입니다. **arXiv non-exclusive 라이선스이고 CC BY 4.0이 아닙니다.** 분석 시점은 2026-11-21입니다.
- **PDF 본문 고지**: pdftotext로 전문을 검색했습니다. ©, copyright, All rights reserved, 회사 고지는 없습니다. 첫 쪽에는 arXiv 스탬프만 있습니다. 참고문헌의 "Google DeepMind Blog"(ref [64])는 인용 출처일 뿐 권리 고지가 아닙니다.
- **판정**: 권리 고지는 없지만 라이선스가 CC BY가 아닙니다. 따라서 **논문 그림은 공개 산출물(`report_public.pdf`, `publish/`)에 넣지 않습니다.** 그림 판독과 그림 재구성은 개인용 분석에서만 쓰고, 공개본은 본문 수치와 우리가 그린 그래프만 씁니다. 리포트에도 논문 그림은 넣지 않았습니다.
- 공식 코드 저장소에는 LICENSE 파일이 없습니다(§2 표). 코드를 복사해 쓰지 않으므로 권리 문제는 생기지 않지만, 재사용하려면 저자에게 확인해야 합니다.

## 2. 공식 코드와 관련 저장소

| 저장소 | 관계 | 라이선스 | 규모·최근 활동 (2026-11-21 조회) | 상태 |
|---|---|---|---|---|
| [shivanshpatel35/pgrd](https://github.com/shivanshpatel35/pgrd) | 공식 구현. 학습·평가 코드, CMA-ES 파라미터 최적화, 3DGS 렌더링 (논문 저자 1인 계정) | **LICENSE 파일 없음** (GitHub API `license: null`) | 별 14, 생성 2026-04-18, 마지막 push 2026-07-15 (커밋 `e294d96`, 2026-07-15). 파일 109개 | `repo/`에 얕은 복제. 정적 읽기만(`04_code_verification.md`) |
| [kywind/pgnd](https://github.com/kywind/pgnd) | README가 "built on"이라고 밝힌 기반 코드(PGND, 저자 Kaifeng Zhang) | MIT (API) | 별 95, 마지막 push 2026-09-03 | 복제하지 않음. 데이터 구성 참고용으로만 언급됨 |
| [pgrd-robot.github.io](https://pgrd-robot.github.io/) | 프로젝트 페이지. 논문 PDF(`static/pdf/paper.pdf`)와 영상 링크 | - | - | HTML만 받아 링크 확인 |
| Google Drive 데이터 폴더, 체크포인트 폴더 | README가 가리키는 가공 데이터와 체크포인트 | 확인 안 됨 | - | **받지 않음** (가중치·데이터 다운로드 금지) |

- 공식 저장소의 README는 `diff-gaussian-rasterization-w-depth` 하위 모듈을 설치하라고 씁니다. 그러나 `.gitmodules`가 없고 하위 모듈 코드도 저장소에 없습니다. 설치 경로가 불완전합니다(`04` D12).
- 데이터 수집 파이프라인(RealSense, CoTracker, Grounded SAM 2, RaySt3R, 계획·언어 파이프라인, AnyGrasp)은 저장소에 없습니다(`04` D10, D11).

## 3. 선행연구 계보 (구성 요소별 출처)

| PGRD의 요소 | 논문 출처 | 관계 |
|---|---|---|
| 스프링-질량 백본 | 스프링-질량 모델 [4 Baraff·Witkin 1998, 45 Liu 2013, 41 Liang 2019] | 표준 모델. 논문은 효율과 해석 가능성을 이유로 택함(§II) |
| 백본 파라미터 식별 | PhysTwin [34] (미분 가능 스프링-질량, 영상에서 식별) | 기준선 Spring-Mass와 Diff. Spring-Mass의 출처. PGRD는 미분 대신 CMA-ES를 씀 |
| 미분 없는 최적화 | CMA-ES [25 Hansen·Ostermeier 2001] | 접촉 때문에 기울기가 깨진다는 이유로 채택(§III-A). 근거는 `mpm_gradient_reliability/`의 FD·기울기 신뢰성 분석과 같은 문제의식 |
| 잔차 학습 개념 | 로봇 잔차 모델 [22 Golemo 2018, 23 Gruenstein 2021, 7 Bauersfeld 2021, 14 Dong 2026], 물리 시뮬을 잔차로 보정 [2 Ajay 2018, 19 Gao 2024] | 논문은 로봇 모델 잔차는 쉬운 편이고, 환경 잔차를 다룬 [2]는 저차원 평면 밀기라 직접 비교가 안 된다고 씁니다(§II). **고차원 입자 상태의 잔차는 처음이라는 주장**입니다 |
| 속도 기반 적분 | 충격 기반 시뮬 [54 Mirtich·Canny 1995, 72 Stewart·Trinkle 2000] | 위치 보정이 속도와 어긋나 불안정하다는 근거. 정량 실험 없음(`03` K2) |
| 입자 기반 학습 동역학 | GBND [94], PGND [93], AdaptiGraph [92], 입자·그래프 신경 동역학 [39, 40, 67] | 학습 기준선. PGND는 공식 최신 학습 기준선이고 GBND는 GNN 기준선 |
| 점 기반 인코더 | PTv3 [83], PointNet++ [62] (절제) | 인코더 교체 실험 대상 |
| 시간 게이팅 | GRU [11 Cho 2014], LSTM [27] | "게이트 덧셈 대 게이트 블렌딩" 절제의 근거 |
| 계획 | MPC 서베이 [20 Garcia 1989]. MPPI 원논문은 인용되지 않음 | 계획 알고리즘 MPPI의 원출처가 빠져 있음(`01` §7 빠진 정보) |
| 영상 렌더링 | 동적 3DGS [47 Luiten 2024], 3DGS 원논문 미인용 | 3DGS 위치·공분산만 갱신하는 설계는 [47]과 같은 계열 |
| 분할·추적 | Grounded SAM 2 [65], CoTracker3 [35] | 데이터 파이프라인 |
| 목표 영상·깊이·파지 | Nano Banana Pro [64] (블로그), Depth Anything V2 [88], AnyGrasp [18], DINOv2 [59] | 언어 조건 계획 파이프라인. 정량 평가 없음 |

## 4. 가까운 대안과 위치

- **PhysTwin [34] 계열(미분 가능 스프링-질량)**: 물리 백본을 같이 쓰는 가장 가까운 대안입니다. 논문도 Diff. Spring-Mass로 직접 비교합니다. Diff. Spring-Mass는 PhysTwin의 **저자가 쓴 구현이 아니고**, 저자들이 "canonical spring graph를 에피소드 초기 관측에 등록해 맞춘" 변형으로 구현했습니다(§IV-C). 따라서 PhysTwin 원본 성능과 같다고 볼 수 없습니다.
- **PGND [93]**: 가장 강한 학습 기준선. 논문 결과에서 PGRD와의 격차는 대부분 SD 안에 있습니다(`03` §2).
- **GBND [94]**: GNN 기준선. Teddy에서 PGND보다 낫게 나온 셀이 있습니다(`03` K1).
- **Ajay 2018 [2]**: 환경 잔차를 확률 신경망으로 붙인 평면 밀기 연구. 가장 가까운 개념 선행연구입니다.

## 6. 새로움의 위치 (판단)

- 잔차 학습 자체는 새롭지 않습니다(§3의 로봇·환경 잔차 연구).
- 고차원 입자 변형체에 **속도 잔차 + 게이트 시간 집계 + 다중 스텝 학습**을 묶은 것이 본 논문의 기여입니다. 이 묶음은 구성 요소 각각의 선행이 있고, 조합의 효과만 절제 표(표 III, 시드 하나)로 보입니다.
- "가장 가까운 선행이 직접 비교 불가"라는 논문의 주장(§II)은 맞습니다. 그러나 비교 가능한 가장 가까운 대안은 Diff. Spring-Mass와 PGND이고, 둘 다 같은 데이터에서 논문이 구현했습니다.

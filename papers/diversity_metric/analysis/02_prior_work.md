# 02. 선행연구와 관련 저장소

## 1. 선행연구 계보 (구성 요소별 출처)

| 구성 요소 | 논문에서의 역할 | 출처 (논문 참고 문헌 번호) |
|---|---|---|
| 다양성을 샘플 유사도로 재는 발상 | 분포 없이 샘플 자체의 Gram 행렬로 엔트로피를 정의 | Vendi score, Friedman and Dieng [1] (arXiv:2210.02410). 엔트로피 값을 유효 랭크로 해석 |
| 시그니처 변환과 커널 | 궤적을 경로로 보고 길이가 달라도 비교 | 시그니처 입문, Chevyrev and Kormilitzin [13]; 시그니처 커널, Lee and Oberhauser [12] (arXiv:2305.04625); *The Signature of a Path* (Springer 2007) [27] |
| 시그니처 커널의 PDE 계산 | 커널 트릭으로 무한 차수 커널 계산 | Salvi et al. [23] (Goursat PDE) |
| 순서 데이터 커널의 동적계획법 | 절단 시그니처 커널 | Király and Oberhauser [24] (JMLR 2019) |
| 무작위 푸리에 시그니처 특징(RFSF) | 대규모 계산을 위한 근사 (FAKTUAL 기본) | Tóth, Oberhauser, Szabó [25] (SIAM 2025) |
| 부분집합 선택의 탐욕 근사 | 최대 피복 문제의 NP-hard 성질과 greedy, stochastic greedy | Karp [31] (NP-hard), Khuller et al. [32] (budgeted max coverage); stochastic greedy 보장은 부록 F.2 |
| 전역 정렬 커널 (대조 커널) | 시계열 커널 비교 대상 | Cuturi et al. [41] (2007) |
| 로봇 데이터 품질 추정 (보조 모델) | 기준선 | DemInf, Hejna et al. [19] (상호정보 추정) |
| 데이터 영향 함수 기반 큐레이션 | 기준선 | CUPID, Agia et al. [22] (2025) |
| 온라인 경험으로 시연 자기 큐레이션 | 기준선 | Demo-SCORE, Chen et al. [20, 21] (arXiv:2503.03707). **참고 문헌에 같은 논문이 두 번 실림(§5 참조)** |
| 데이터 품질 영향 | 동기 | Belkhale et al. [26] |
| 다양성이 일반화에 도움 | 동기 | 부록 A에서 Yu et al. (NLP), Zhang et al. (영상 산란), Jung et al. [66] (G-Vendi)를 인용 |
| 다양한 행동을 가진 IL 벤치마크 | 관련 벤치마크 | Jia et al. [42] |
| 행동 복제와 정책 구조 | 실험 설정 | BC-RNN, Mandlekar et al. [16] (RoboMimic); 확산 정책, Chi et al. [38]; SmolVLA, Shukor et al. [39] |
| 과제 | 실험 | RoboMimic (Can, Square, Transport); MetaWorld, Yu et al. [33] (Door open, Stick push, Shelf place); 실로봇 SO-ARM101 [40] |

- 논문의 주장: 기존 "데이터 품질 기반" 큐레이션(DemInf, CUPID, Demo-SCORE)은 보조 모델이나 온라인 롤아웃을 쓰므로 비용이 크다(§1). FAKTUAL은 이 비용을 피한다는 것이 차별점입니다. 다만 계산 비용의 실측 비교는 논문에 없습니다(01 §3.5).
- 엔트로피 측정과 최적화(greedy) 자체는 기존 기법의 조합입니다. 새로운 부분은 **궤적 집합에 시그니처 커널의 엔트로피를 쓰는 것**과 **이를 큐레이션 목적으로 쓰는 것**입니다.

## 2. 코드와 저장소 (GitHub 표)

| 저장소 | 관계 | 라이선스 (GitHub API / 저장소 파일) | 기준 | 상태 |
|---|---|---|---|---|
| [S-sirigiri/FAKTUAL](https://github.com/S-sirigiri/FAKTUAL) | 공식 구현 (`kernel_diversity_lib/`, 저자 계정) | **없음** (API `license: null`, 루트에 LICENSE 파일 없음) | 커밋 `7aa0e6e3` (2026-11-09), 스타 8, 생성 2025-06-04, 마지막 push 2026-11-09 | 얕은 복제 `repo/` 정적 읽기 (실행, 설치 안 함) |
| 같은 저장소 안의 `robomimic/` | 벤더링 (로봇 학습 프레임워크) | MIT (Stanford Vision and Learning Lab, 2021) | 위와 같음 | 정적 읽기만 |
| 같은 저장소 안의 `baselines/cupid/` | 벤더링 (CUPID 기준선) | MIT (Christopher Agia, 2025) | 위와 같음 | 정적 읽기만 |
| 같은 저장소 안의 `baselines/demonstration-information/` | 벤더링 (DemInf 기준선) | MIT (Joey Hejna, 2025) | 위와 같음 | 정적 읽기만 |
| [S-sirigiri/sigkernel](https://github.com/S-sirigiri/sigkernel) | 요구 사항 (`requirements.txt`, git 설치) | Apache-2.0 (포크 원본 crispitagorico/sigkernel, 스타 66, 마지막 push 2025-11-04) | 포크, 마지막 push 2025-02-18 | 읽지 않음 (포크와 원본의 차이 미확인) |
| [S-sirigiri/KSig](https://github.com/S-sirigiri/KSig) | 요구 사항, RFSF 구현 | Apache-2.0 (포크 원본 tgcsaba/KSig, 스타 53, 마지막 push 2025-06-06) | 포크, 마지막 push 2025-06-18 | 읽지 않음. RFSF의 실제 구현은 여기에 있음 |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | 서브모듈 (`lerobot/`) | Apache-2.0 (스타 29,006, 마지막 push 2026-11-21) | 복제에서 초기화하지 않아 비어 있음 | 읽지 않음 |
| 프로젝트 페이지 | 찾지 못함 | — | 논문 본문, PDF 링크 주석, GitHub 검색 기준 | 없음 |

- 코드 저장소 주의점: 루트에 라이선스가 없으므로 저작권은 저자에게 있고, 재배포 권한이 명시되지 않았습니다. 이 저장소는 코드를 복사하지 않고 설명만 씁니다.
- 기준 커밋의 이력(3개): "Initial public release" (2026-10-13), "Revise citation format" (2026-10-19), "Update README to include hyperlink for paper" (2026-11-09). v2 개정 날짜(2026-10-13)와 맞습니다.
- 논문 PDF의 URI 주석 문자열을 찾아본 결과 arXiv 링크만 있었습니다. 본문에는 "this GitHub repository"라고만 적혀 있습니다. URL은 GitHub 검색으로 찾았고, 저장소 README의 인용 항목과 논문 저자 목록(Sirigiri)이 일치합니다.
- 공식 저장소의 README는 "현재 커널과 메트릭 파이프라인은 CuPy를 쓰고 NVIDIA GPU와 CUDA 환경이 필요하다"고 적습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- 데이터 품질과 커버리지가 정책 성능을 함께 결정한다는 가정(§1, Mandlekar et al. [16] 등). 이 가정 아래에서 다양성은 "필요하지만 충분하지 않은" 지표입니다. 논문도 결론에서 품질, 국소 밀도, 다양성을 함께 봐야 한다고 씁니다(§8).
- 시그니처 커널의 재매개변수화 불변성과 선형 함수의 조밀성(§4, Stone–Weierstrass 인용). 이론적 배경이며 본 분석에서 검증하지 않았습니다.
- 엔트로피 포화(그림 2): 본문은 Can에서 포화되고 Transport에서는 포화되지 않는다고 씁니다. 포화 여부는 엔트로피 값이 있어야 판단할 수 있고, 표에는 없습니다.

## 5. 참고 문헌 목록의 점검

- 참고 문헌 [20]과 [21]은 같은 논문(arXiv:2503.03707, Chen et al.)입니다. 두 항목의 제목이 같고 저자가 같습니다. 중복입니다.

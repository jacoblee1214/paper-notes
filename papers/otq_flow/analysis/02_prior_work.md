# 02. 선행연구 계보와 GitHub

분석일 2026-11-22. 인용 번호는 OTQL 논문의 참고문헌 번호입니다.

## 1. 계보

OTQL은 세 줄기를 한 목적함수에 묶습니다. (가) 보상 가중·어드밴티지 가중 회귀로 정책을 기울이는 줄기, (나) flow·diffusion 정책을 RL로 후훈련하는 줄기, (다) 미니배치 OT로 flow 경로를 곧게 펴 NFE를 줄이는 줄기입니다.

| 줄기 | 핵심 작업 | OTQL이 가져온 것 |
|---|---|---|
| 기울임 `π̃ ∝ π e^{λA}` | RWR [41], REPS [42], AWAC [43] | 닫힌 해와 어드밴티지 가중. §2의 "KL 정규화 문제의 해" |
| flow·diffusion의 오프라인 RL | Diffusion-QL [20], IDQL [52], QSM [51], EWFM [22], Guided Flow Policy [21], Diffusion Actor-Critic [77] | 어드밴티지(에너지) 가중 흐름 매칭. EWFM [22]은 OTQL의 가장 가까운 가중 FM 대조군(Fig. 1, Fig. 3) |
| 한 스텝 정책 | FQL [25] (증류 + Q 손실), 일관성 정책 [27], One-Step Diffusion Policy [26] | "증류 없이"의 대조 대상 |
| 노이즈 공간 RL | DSRL [16] | 실로봇 후훈련 기준선. 블랙박스 정책 가능(OTQL은 불가, §5) |
| 액션 청킹 RL | Q-Chunking (QC) [23] | 청크 Q 함수, TD 손실의 청크 보상 합, 거절 샘플링(N=32) |
| adjoint matching | QAM, QAM-F [24] | OGBench 실험 설정과 **기준선 값 전체**(표 1). 부록 D.1이 "[24]와 같은 평가 설정"이라 밝힘 |
| OT 결합으로 곧은 경로 | OT-CFM [10], multisample FM [44], COT 흐름 [32], curse of conditions [33], 같은 저자의 COT Policy [28] | 미니배치 OT, 조건 비용 증강 `식 (3)–(4)` (이 부분은 [28]에서 그대로) |
| 실로봇 경험 기반 VLA 개선 | π*0.6 [46] (어드밴티지 조건), 잔차 RL [18, 19], DynaGuide [17] | 희소 보상 `r = 0 / −1`과 에피소드 라벨링 방식 [46] |
| VLA 대상 | SmolVLA [39], LeRobot [50] | 실험 대상과 코드 기반 (부록 D.3) |

**독창성의 위치.** 새로운 것은 "에너지 가중 목표 질량을 OT 결합의 한계분포에 직접 넣는다"는 한 줄입니다(`식 (5)`). 가중 흐름 매칭(EWFM)은 손실에 가중치를 곱하는 반면 OTQL은 결합 자체를 기울여 표본 분산을 줄이고 경로를 곧게 한다는 것이 주장입니다(§3.1, Fig. 1). 조건 증강과 가속 아이디어는 같은 저자의 COT Policy [28]입니다. 즉 **가속은 이전 논문의 것이고, 새 논문은 거기에 가중 질량으로 RL 신호를 얹은 것**입니다.

## 2. 가까운 대안과 차이

| 방법 | 정책 접근 | 가속 | 실로봇 | OTQL과의 차이 |
|---|---|---|---|---|
| FQL [25] | flow → 한 스텝 가우시안으로 증류 | 1 NFE | OGBench | 증류 필요, 표 1 평균 0.44 |
| QAM / QAM-F [24] | adjoint matching(경량) | QAM-F는 1 NFE | 없음 | OGBench 기준선, QAM-F는 증류형 |
| DSRL [16] | 노이즈 공간에서 RL | 정책은 그대로 | 있음 | 블랙박스 가능, 정책 가속 없음 |
| QC [23] | 청크 Q + 거절 샘플링 | NFE 10, N=32 | 없음 | 후보 수와 NFE가 많음 |
| EWFM [22] | 가중 FM 손실 | NFE 10 | 없음 | 결합을 안 기울임, OTQL의 직접 대조군 |
| π*0.6 [46] | 어드밴티지 조건 | 없음 | 있음 | 대규모 VLA, 코드 없음 |

## 4. GitHub와 공개 자료

모두 2026-11-22에 `gh api`로 확인했습니다. 복제는 OTQL 공식 코드가 없어 하지 않았습니다(관련 저장소도 복제하지 않았습니다).

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| **공식 코드: 없음** | 프로젝트 페이지가 "Code (Coming Soon)" | | | | 논문 PDF에도 저장소 링크가 없음. GitHub 검색(otql, optimal transport q-learning)에서 저자 계정의 코드 저장소 없음 |
| [ansocho/otql-flow](https://github.com/ansocho/otql-flow) | 프로젝트 페이지 저장소(HTML, 영상, `OTQL.pdf`). 코드 아님 | 없음 | 0 | 2026-07-08 | 마지막 커밋 `11e8c19` "add arxiv". README는 OTQL_CoRL2026_appendix.pdf를 언급하나 저장소에는 `OTQL.pdf`만 있음 |
| [ansocho/cot_policy](https://github.com/ansocho/cot_policy) | 선행 COT Policy(arXiv:2505.01179, CoRL 2025)의 공식 코드 | README 배지는 MIT, LICENSE 파일 없음(API: 없음) | 14 | 2025-07-16 | 조건 비용 증강 결합의 구현이 있을 가능성(미확인) |
| [seohongpark/fql](https://github.com/seohongpark/fql) | FQL 공식 | MIT | 349 | 2025-07-21 | 기준선, OGBench 코드 기반 |
| [ColinQiyangLi/qc](https://github.com/ColinQiyangLi/qc) | QC 공식 | MIT | 422 | 2026-02-05 | 청크 Q, 거절 샘플링 |
| [ColinQiyangLi/qam](https://github.com/ColinQiyangLi/qam) | QAM, QAM-F 공식 (OGBench 평가 기준) | MIT | 135 | 2026-05-11 | 표 1 기준선 값의 출처 |
| [ajwagen/dsrl](https://github.com/ajwagen/dsrl) | DSRL 공식 | 없음 | 244 | 2025-08-05 | 실로봇 기준선 |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | SmolVLA·학습 코드 기반 (부록 D.3) | Apache-2.0 | 29,021 | 2026-11-22 | OTQL이 LeRobot 위에서 구현됨 |

- 프로젝트 페이지(`ansocho.github.io/otql-flow/`)에는 방법 설명, 영상(cups, door, tape, chess 성공·실패 롤아웃), 표·Fig. 요약이 있고 BibTeX 키는 `sochopoulos2026optimaltransportqlearningflow`입니다. 페이지에만 있는 정보가 하나 있습니다: **학습 스텝당 flow 평가 횟수** (크리틱 손실용 2, 가치 추정용 2, 흐름 매칭 손실 1, OT 계산 환산 1). 논문 본문에는 없는 수치라 근거는 페이지 한 곳뿐입니다.
- 프로젝트 페이지의 "OTQL.pdf"는 arXiv v1과 본문이 같습니다(`01` §0).

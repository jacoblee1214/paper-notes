# ContactBench 1.0: 선행연구, GitHub 저장소, v1 → v2 차이

조사일 2026-11-10. GitHub 수치는 같은 날 `gh api`로 확인했습니다. 학회 정보는 논문 본문, arXiv 초록 페이지, 프로젝트 페이지에서만 확인했고, 학회 채택은 확인하지 못했습니다.

## 1. 선행연구 계보

논문 Sec. II의 분류를 따르고, 이 분석이 뽑은 가까운 대안을 덧붙였습니다.

**(a) 재현 가능한 물리 벤치마크**
- 표준 하드웨어: YCB(Calli et al., 2015), NIST 과제 보드(Kimble et al., 2020), ManipulationNet(Chen et al., 2026; 분산 실물 평가 인프라).
- 인쇄 설계: FurnitureBench(Heo et al., IJRR 2025), Fabrica(Tian et al., CoRL 2025), FMB(Luo et al., IJRR 2025), POMDAR(Liconti et al., RA-L 2026).
- 조건을 체계적으로 바꾸는 것: MatchMaker(Wang et al., 2025), **Peg-in-Bench**(Delgado et al., 2026, arXiv:2609.00906; 삽입 전용, 가장 가까운 대안), BusyBox(Fortier et al., 2026; 외형 일반화).
- ContactBench의 차별점은 논문 Sec. II-A의 주장대로 "기술 8개 범위"와 "인쇄 스프링이 힘을 정하는 설계"입니다.

**(b) 시뮬레이션 벤치마크**
- 과제 묶음: RLBench, MetaWorld, robosuite, ManiSkill2/3(GPU 병렬), CALVIN, LIBERO. 장기 과제와 언어 조건은 CALVIN과 LIBERO.
- 촉각 시뮬레이션: Taxim(렌더링), Tactile Gym 2.0(센서 비교), roto 2.0(Robot Tactile Olympiad, 2026), ManiFeel(2025), UniVTac(2026), OpenViTac(2026). 이들은 촉각 정책 평가 플랫폼이지만, 물리 자산과 접촉 행동을 연결하지는 않습니다(논문 Sec. II-B).

**(c) 평가한 정책**
- Diffusion Policy(Chi et al., IJRR 2025): 확산 정책의 표준 기준선.
- π0.5(Black et al., CoRL 2025): 비전-언어-행동 모델. 과제마다 미세조정(단일 과제 전문가로 씀).
- Reactive Diffusion Policy(Xue et al., 2025): 느린 시각 정책 + 빠른 촉각 반응 정책.
- Foundation Tactile Policy, FTP-1(Yuan et al., arXiv:2606.13102, 2026): π0.5 기반 범용 촉각 정책.
- 힘/접촉 정책 참고: ForceMimic(ICRA 2025), FoAR(RA-L 2025)(논문 참고문헌 [8], [9]). 힘 기반 기준선으로는 이 논문이 쓰지 않았습니다.

**(d) 일반화 분류**: Star-Gen(Gao et al., RA-L 2026)의 행동 변이와 시각+행동 변이 구분을 씁니다.

## 2. v1 → v2 차이

v1은 2026-10-17, v2는 2026-10-23입니다. 두 PDF(v1 16.99 MB 19쪽, v2 46.07 MB 23쪽)를 받아 본문 텍스트와 Table III 이미지를 대조했습니다.

**수치는 같습니다.** Table III(방법별 과제별 성공 수, 진행률, >40 N, 시간 배수)은 v1과 v2가 같습니다(이미지로 확인). Fig. 9(v1)과 Fig. 11(v2)의 단계 퍼널 수치도 같습니다. 캡션도 같습니다.

**달라진 것**

| 항목 | v1 | v2 |
|---|---|---|
| 용어 | "vision-based", "vision-tactile" 8회·12회 | "vision-only", "visual-tactile"로 통일(각각 1회·17회) |
| Table I | 과제 예시(Examples of reference tasks) | 변이 목록(Task variations) |
| Table II | 변이 목록 | 정책 평가 6개 과제의 사양(성공 기준 포함) |
| 부록 | A(자산 모델링) 한 편 | A 확장(Fig. A1, 촉각 시뮬 식 A1–A3, Fig. A2), **B 신설**(학습 커밋 해시, Table B1 학습 설정, Table B2 시연 통계와 언어 지시), **C 신설**(36개 과제 전체 사양, Table C1·C2) |
| 초기 조건 | "체계적으로 샘플링" | "**고정 난수 시드**로 물체 초기 자세를 뽑고, 사람이 그 위치에 둔다"는 문장 추가 |
| 초기 조건 그림 | 없음(PyBullet 렌더 오버레이를 본문에만 설명) | Fig. 10 신설(샘플링 위치의 렌더 오버레이와 벨크로 보드) |
| 그림 번호 | Fig. 8 "여섯 과제의 사람 시연 예"(v1) | 삭제. Table II의 "실행 순서" 열로 대체 |
| 문손잡이 변이 | Table IV(회전량·직경별 성공률) | 성공률은 Fig. 14 왼쪽으로 옮김. v2의 Table IV는 그리퍼 폭(v1 Table V) |
| 단위 | "5.5 cm", "6 cm" | "55 mm", "60 mm" |
| 롤아웃 시간 | "at most one minute" | "has a one-minute time limit" (표현만 바뀜) |
| 시뮬 그림 | Fig. 4–7 | Fig. 6–9 (번호만 바뀜) |

**해석**: v2의 실질적 추가는 (1) 재현 정보(학습 커밋, 학습 설정, 과제 사양), (2) 초기 조건 샘플링의 시드, (3) 문손잡이 변이 결과의 이동입니다. 결과 수치는 바뀌지 않았습니다. v2는 "시드가 있다"는 점을 명시했지만, 시드 값과 에피소드별 기록은 여전히 없습니다(`01` §5).

## 3. 이 저장소의 다른 논문과의 관계

이 분석은 같은 저장소의 아래 분석들과 방법과 질문을 공유합니다. 각 항목은 해당 폴더의 README와 분석 문서 요약을 보고 적었습니다. 판단은 원문 재확인 없이 요약 수준입니다.

| 폴더 | 논문 | 관계 | 이 분석에 주는 것 |
|---|---|---|---|
| `contact_disassembly/` | CONTACT, arXiv:2603.08560 (촉각 분해 정책) | 같은 문제: 접촉 정책의 시뮬·실물 통계 | 같은 가정 문제. 그 분석은 "1500 독립 rollouts"(CMH p = 2.6e-9)와 "3 시드 × 초기 조건 50"(p = 0.062)의 차이를 보였습니다. 이 분석의 `03` §7(토이 P2)은 초기 조건 공유보다 **학습 실행 분산**이 거짓 양성을 키움을 보입니다 |
| `restacvla/` | ResTacVLA, arXiv:2607.03387 (π0.5 + 촉각 잔차) | FTP-1과 같은 계열(π0.5 기반 촉각 확장) | 같은 통계 틀(Fisher, Wilson, CMH)을 씁니다. 이 분석의 `03` §3에서 π0.5 → FTP-1 비교의 CMH 결과(p = 0.0097)와 비교할 수 있습니다 |
| `insertion_wm/` | 세계모델 삽입, arXiv:2609.28258 | 벤치마크 점수의 객체 간 분산 | 보고된 "56%" 대 "7%"는 같은 조건이 아니라는 점, 객체 분산을 넣으면 구간이 ±15–19 pp로 넓어진다는 점이 이 논문 표 III의 30회 칸(Wilson 구간 폭 11–34 pp)과 같은 문제입니다 |
| `stiffness_copilot/` | Stiffness Copilot, arXiv:2603.14068 | "안전" 주장의 힘 지표 검정 | 최대 접촉력 차이를 과제별로 검정하는 방식은 이 논문의 >40 N 지표를 검정할 때 참고가 됩니다. 이 논문은 힘 지표 검정을 하지 않았습니다 |
| `dexforce/` | DexForce, arXiv:2501.10356 | 촉각 힘을 **행동 목표**에 넣는 Diffusion Policy | 이 논문의 기준선(촉각 입력을 관측에 넣는 RDP, FTP-1)에 없는 설계 축입니다. 힘을 목표로 바꾸는 방식은 벤치마크에 추가할 후보입니다 |
| `beyond_implicit_force/` | Beyond Implicit Force, arXiv:2607.14578 | 힘 인지의 원천(관절각 편차 대 서보 전류) | 이 논문은 F/T를 자산 검증에만 쓰고 정책 입력에는 넣지 않았습니다. 힘 입력 기준선이 없다는 점(`01` §8)의 대조군입니다 |
| `pacevla/` | PaceVLA, arXiv:2610.25961 | VLA 평가(공식 코드 없음) | 과제별 단일 전문가 평가 설정과 비교할 수 있습니다 |
| `thinkworld/` | ThinkWorld, arXiv:2610.37666 | VLA 평가와 공식 리더보드 대조 | 벤치마크 점수와 실물 성능의 연결을 따지는 질문이 같습니다. 이 논문은 정책 수준의 연결 증거를 내지 않습니다 |

## 4. GitHub 저장소 표

| 저장소 | 무엇인가 | 공식? | 라이선스 | 별 | 최근 push | 고정 커밋 |
|---|---|---|---|---|---|---|
| [Michaelwyx/contactbench-sim](https://github.com/Michaelwyx/contactbench-sim) | ContactBench 1.0 시뮬레이션. force-bench 패키지, 과제 22종(73행), 스크립트 정책, 원격조작, GelSight 시뮬, 자산 키트. **정책 학습과 평가 파이프라인은 제외**(README 명시) | **공식**(논문 Simulation code 링크, 프로젝트 페이지 링크) | MIT (LICENSE, 저작권 "2026 Yixuan Wang") | 9 | 2026-11-05 | `ccd864501275` (2026-11-04 커밋 "ContactBench 1.0 simulation"). 얕은 클론 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0.5 구현(논문 commit `15a9616`) | 의존(논문: 공식 구현 사용) | Apache-2.0 | 14,604 | 2026-08-24 | `15a9616` (2026-06-16) 존재 확인 |
| [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | Diffusion Policy 구현(논문 commit `5ba07ac`) | 의존(공식 구현) | MIT | 4,721 | 2024-12-24 | `5ba07ac` (2024-12-24) 존재 확인 |
| [xiaoxiaoxh/reactive_diffusion_policy](https://github.com/xiaoxiaoxh/reactive_diffusion_policy) | RDP 구현(논문 commit `8ed82a3`) | 의존(공식 구현) | GitHub 자동 판정 NOASSERTION(LICENSE 파일 미확인) | 396 | 2026-04-12 | `8ed82a3` (2025-12-12) 존재 확인 |
| [michaelyuancb/ftp1-policy](https://github.com/michaelyuancb/ftp1-policy) | FTP-1 구현(논문 commit `89fa681`) | 의존(공식 구현) | Apache-2.0 | 145 | 2026-07-15 | `89fa681` (2026-07-15) 존재 확인 |
| [newton-physics/newton](https://github.com/newton-physics/newton) | 시뮬 엔진. force-bench는 `newton==1.5.0`에 고정 | 의존 | Apache-2.0 | 5,933 | 2026-11-10 | 버전 고정 1.5.0 |
| [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) | 솔버. force-bench는 `mujoco-warp==3.11.0`에 고정 | 의존 | Apache-2.0 | 1,626 | 2026-11-10 | 버전 고정 3.11.0 |

**해석**: 공식 시뮬 저장소는 별 9개이고 최근 push는 2026-11-05(5일 전)입니다. 커뮤니티 활동은 아직 작습니다. 정책 구현 네 개는 모두 논문이 고정한 커밋이 존재합니다. 그러나 **이 저장소에 정책 학습 스크립트, 데이터셋, 체크포인트, 평가 스크립트가 들어 있지 않아** Table III를 재생성할 수 없습니다(`04` D8).

## 5. 이 논문이 다른 벤치마크와 다른 점(요약)

- 평가 대상: 정책 중심(DP, π0.5, RDP, FTP-1). 힘 입력은 기준선에 없습니다.
- 물체: 인쇄 가능한 물체(PLA + M3 나사)와 같은 물체의 시뮬 모델. 스프링의 힘을 K와 τ0로 맞춥니다(검증은 두 물체 종류).
- 과제: 36개를 정의하지만, 시뮬에서 채점 가능한 것은 22개 기본 과제(73행)와 과제 2의 장면입니다(`04` D1).
- 통계: 30회, 한 학습 실행, 집계 표만 공개(`03`).

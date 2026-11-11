# ULTRA: 선행연구와 관계

## 1. 버전 차이

| 버전 | 날짜 | 쪽 수 | 내용 차이 |
|---|---|---|---|
| v1 | 2026-03-03 | 13 | 초록과 서론이 더 길고, 용어가 "skill latent"(잠재 기술 공간)입니다. 참고문헌 번호가 다릅니다(PHC는 [15], GMR은 [1]). 표 I–IV의 수치는 v3와 같습니다. |
| v2 | 2026-09-18 | 12 | 본문을 다시 씀. 저자 목록과 제목 쪽 배치가 바뀌었습니다. 수치 변화는 텍스트 추출로 확인하지 못했습니다(표 확인은 v3 기준). |
| v3 | 2026-10-17 | 12 | v2와 텍스트 차이는 두 줄뿐입니다(arXiv 스탬프 날짜, 프로젝트 주소 `ultra-humanoid.github.io` → `ultrahumanoid.github.io`). |

- v1과 v3의 **표 수치는 같습니다**(표 I, II, III, IV를 텍스트 추출과 렌더로 확인했습니다). 표 IV의 열 배치(Vertical, Lateral)도 두 버전에서 같습니다.
- v1 대 v2 텍스트 차이는 1,497줄로 대부분 재작성입니다. 결과 수치를 바꿨는지는 표로 확인했습니다.
- 원본 PDF: v1(`.scratch/v1.pdf`), v2(`.scratch/v2.pdf`)는 `ultra_humanoid/.scratch/`에 있습니다(gitignore). 최신 v3는 `paper/paper.pdf`입니다.

## 2. 계보

### 2.1 리타게팅

| 흐름 | 대표 연구 (논문 참고문헌) | ULTRA와의 관계 |
|---|---|---|
| IK 기반 애니메이션 리타게팅 | Lee & Shin, SIGGRAPH 1999 [8] | 운동학만 씀. 물리 일관성 없음 |
| 신경 리타게팅 | Villegas et al., CVPR 2018 [9] | 상각된 매핑. 접촉 무시 |
| 운동학 휴머노이드 리타게팅 | PHC 계열 [4], GMR [5], OmniRetarget [2] | 기준선. 논문 Table II와 I의 대상 |
| 물리 기반 리타게팅 (샘플링·RL) | DexPlore [11], SPIDER [12] | 물체 한 개씩 최적화. ULTRA는 한 정책으로 모든 동작을 처리한다고 주장 |

ULTRA의 위치는 "물리 기반이지만 한 정책으로 대량 처리"입니다. 이 점이 새롭다고 주장하지만, 이전 연구가 같은 목적으로 RL 리타게팅을 쓴 경우와의 직접 비교는 없습니다.

### 2.2 휴머노이드 추적과 기반 모델 방식

- 추적 제어: OmniH2O [13], ASAP [15], HOVER [16], TWIST [18], GMT [19], UniTracker [20], SONIC [21], BeyondMimic [22], BFM-Zero [27]
- 잠재 사전 분포: LeVERB [24], StyleLoco [25], AMP 계열 [26]. 논문은 이들이 대규모 이질 데이터로 확장된 적이 없다고 봅니다.
- 교사-학생 증류: DAgger [38], MaskedMimic [6]. ULTRA의 증류는 DAgger 방식과 잠재 변수 손실(VAE 유사)을 합칩니다.

### 2.3 물체 상호작용과 loco-manipulation

- 같은 저자 계열: InterMimic [3] (CVPR 2025, 사람-물체 추적 교사), InterPrior [7] (CVPR 2026, 증류 후 RL 미세조정 절차의 출처). ULTRA는 InterPrior의 두 단계 절차를 따른다고 씁니다(IV-C).
- 물체 상호작용 추적·조작: HDMI [1], ResMimic [31], DemoHLM [32], PhysHSI [34], DreamControl [33], VisualMimic [17]
- 계층형 설계: 계획기가 궤적을 만들고 추적기가 따름 [17], [33]. ULTRA는 계층 오차 누적을 피하려고 단일 목표 조건부 정책을 씁니다.

## 3. 이 저장소의 관련 분석과 관계

| 폴더 | 대상 | ULTRA와 관계 |
|---|---|---|
| `warp_retargeting/` | WARP(arXiv:2606.29940, CoRL 2026): 사람 상체 시연을 로봇 전신으로 바꾸는 오프라인 기하 리타게터 | **가장 가깝습니다.** 같은 리타게팅 문제를 기하 해법으로 풉니다. ULTRA는 RL로 물리 일관성을 얻습니다. 두 방법의 접촉 품질 지표(관통, 미끄럼, 접촉 뜸)를 맞대어 볼 수 있습니다. |
| `reasoning_retargeting/` | 블로그 "Retargeting Through Reasoning": 에이전트가 MuJoCo 피드백으로 HOI 리타게팅 기준을 만듦 | 리타게팅을 최적화 문제가 아니라 설계 과정으로 봅니다. ULTRA의 보상 항과 대조할 수 있습니다. 블로그라 심사된 근거가 없습니다. |
| `gnr/` | GNR(arXiv:2610.12440): 샘플링 MPC 궤적에 잔차를 더하는 생성 리타게터(손 로봇) | 방법은 다르지만 "리타게팅 궤적을 물리 필터로 거른다"는 점이 같습니다. ULTRA는 손이 없는 G1을 씁니다. |
| `cross_embodiment_wm/` | 교차 형상 세계 모델(IROS 2026) | 학생의 모달리티 마스크와 교차 형상 문제(여러 입력을 한 모델로)가 관련됩니다. |
| `dreamx/` | DreamX(arXiv:2611.05377): 영상 확산 모델 잠재 행동 | 사람-로봇 공통 행동 인터페이스라는 목표는 같지만, ULTRA는 물리 정책입니다. |
| `motion_latent_action/` | Motion-Focused Latent Action(IROS 2026) | 잠재 행동을 교차 형상 VLA에 쓴다는 점에서 ULTRA의 잠재 변수와 비교할 수 있습니다. |
| `contactbench/` | ContactBench(arXiv:2610.24512): 접촉 과제 벤치마크 | 실물 검정력과 표 재계산 방식을 같이 씁니다(`03` §6). |

## 4. GitHub 저장소 표

확인일 2026-11-11. 별 수와 최근 활동은 `gh api`로 확인했습니다. 라이선스는 저장소의 감지 결과입니다.

| 저장소 | 무엇 | 공식? | 라이선스 | 별 | 최근 활동 (push) | 비고 |
|---|---|---|---|---|---|---|
| [Sirui-Xu/ULTRA](https://github.com/Sirui-Xu/ULTRA) | ULTRA 코드, 교사 추론 가중치, 증강 데이터 링크 | 공식(저자 계정) | Apache-2.0 (InterMimic 파생 MIT 포함) | 56 | 2026-10-17, 커밋 `146ddd5f6398` ("Fix project page link in README") | 학생 가중치, 배포 스택, 실제 데이터 없음. `repo/ULTRA/`에 얕게 클론 |
| [Sirui-Xu/InterMimic](https://github.com/Sirui-Xu/InterMimic) | 사람-물체 전신 추적(CVPR 2025), ULTRA 데이터 형식 원본 | 공식 | MIT | 558 | 2026-04-21 | ULTRA가 데이터 형식과 OMOMO 수정 부분집합을 가져옴 |
| [YanjieZe/GMR](https://github.com/YanjieZe/GMR) | 실시간 CPU 운동학 리타게팅(ICRA 2026) | 논문 [5] 구현으로 보임 | MIT | 2,821 | 2026-10-22 | 기준선. 공식 여부는 저자 목록으로 추정 |
| [ZhengyiLuo/PHC](https://github.com/ZhengyiLuo/PHC) | PHC(ICCV 2023) 공식 구현 | 공식 | 감지 안 됨 (NOASSERTION, 원문 확인 필요) | 1,308 | 2025-08-21 | 기준선 [4] |
| [LeCAR-Lab/HDMI](https://github.com/LeCAR-Lab/HDMI) | HDMI(인간 영상 학습) | 공식 | 없음 | 680 | 2026-01-17 | 기준선 [1]. 라이선스가 없어 코드를 가져다 쓸 수 없음 |
| [project-instinct/omniretargeting](https://github.com/project-instinct/omniretargeting) | OmniRetarget(arXiv:2509.26633) 재구현 | **재구현**(논문 저자 공식 구현이 아님) | MIT | 85 | 2026-11-03 | 공식 저장소는 찾지 못함. 이 재구현으로 OmniRetarget 결과를 재현한 것은 아님 |
| [OmniRetarget/omniretarget.github.io](https://github.com/OmniRetarget/omniretarget.github.io) | OmniRetarget 프로젝트 페이지 저장소 | 공식 추정 | 없음 | 2 | 2026-07-11 | 코드 저장소로 보기 어려움 |
| [facebookresearch/spider](https://github.com/facebookresearch/spider) | SPIDER(arXiv:2511.09484) 물리 기반 손 리타게팅 | 공식 추정 | NOASSERTION | 598 | 2026-09-22 | 기준선 아님. 물리 기반 리타게팅의 다른 구현 |

**공식 코드가 있는 것은 ULTRA 하나입니다.** 기준선 중 PHC와 GMR은 공식 구현이 공개되어 있고, HDMI는 공개되었지만 라이선스가 없습니다. OmniRetarget의 공식 구현은 찾지 못했습니다.

## 5. 한 줄 정리

ULTRA는 "사람-물체 MoCap을 물리 RL로 대량 리타게팅 → 교사 → 모달리티 마스크 학생 → RL 미세조정"이라는 파이프라인을 한 논문에 묶은 시스템 논문입니다. 새 알고리즘 하나보다 **단계들의 결합**이 기여입니다. 그래서 개별 단계의 검증 수준(`03`, `04`)을 따로 봐야 합니다.

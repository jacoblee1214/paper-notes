---
title: "RoboSimStudio 읽기: 쓸 만한 Real2Sim 도구, 그런데 스펀지는 손에 붙어 있었다"
date: 2026-10-29
tags: [robotics, simulation, real2sim, coding-agents, paper-review]
paper: https://arxiv.org/abs/2610.18835
code: https://github.com/robosimstudio/robosimstudio
project: https://robosimstudio.github.io/
summary: "사람과 코딩 에이전트가 함께 시뮬레이션 물성을 고치는 Real2Sim 도구 RoboSimStudio를, 공식 저장소와 대조하고 데모 수치를 다시 계산하며 읽었습니다."
---

> **논문:** Chen, Xie et al. *RoboSimStudio: An Interactive Toolkit for Human and Agent Rapid Real2Sim of Robot Manipulation Tasks.* arXiv:2610.18835 (2026). UC Berkeley, NVIDIA.
> 논문 그림은 CC BY 4.0에 따라 출처를 밝히고 실었습니다(© 저자). 나머지 그래프는 공식 저장소(Apache-2.0, `81f57c9`)의 측정값과 직접 계산한 값으로 그렸습니다.

![논문 Fig. 1](images/paper_fig1_overview.png)

## 한 줄 요약

시뮬레이터의 solver, 접촉 모델, 물성 값을 **편집 가능한 항목**으로 드러낸 Newton 기반 도구입니다. 사람은 브라우저 화면과 힘 피드백 리더 암으로, 코딩 에이전트(Claude Code, Codex)는 저장하면 다시 빌드되는 live 페이지로 같은 장면을 고쳐 가며, 실제 영상처럼 움직이는 다중 물리 시뮬레이션을 만듭니다.

## 1. 무엇이 새로운가

기존 Real2Sim 연구는 대부분 재구성이나 생성 파이프라인 하나를 제안합니다. RoboSimStudio는 파이프라인이 아니라 **작업대**입니다.

- 물체는 "무엇인지"(강체, 관절, 천, 케이블, 스펀지)만 선언하고, 장면이 "어느 solver가 맡는지"를 선언합니다. solver를 바꿔도 과제 코드는 그대로입니다.
- 기본 배치는 관절은 MuJoCo(-Warp), 스펀지는 SuperDex FEM, 나머지는 VBD이고, 외부 엔진과 접촉 wrench를 주고받습니다.
- 사람용 인스펙터는 물성을 단위·범위와 함께 보여 주고, 실행 중 바꿀 수 있는 값과 장면을 다시 만들어야 하는 값을 구분합니다.

## 2. 결과는 파일럿 하나

정량 결과는 거의 없습니다. Real2Sim2Real의 근거는 양팔 스펀지 닦기 한 과제입니다. 코딩 에이전트가 스마트폰 영상 하나로 약 1시간 만에 시뮬레이션을 만들고, Graph-as-Policy(GaP)가 14번 반복으로 "닦인 면적"을 0%에서 100%로 올린 뒤 실제 로봇에서 실행했습니다.

![논문 Fig. 7](images/paper_fig7_sponge.png)

읽을 때 걸러야 할 점이 두 가지 있습니다.

- **곡선의 단위가 그림마다 다릅니다.** Fig. 1은 y축을 "completion rate"라고 쓰고, 본문은 "닦인 면적"이라고 씁니다. 29%와 71%는 30개 배치의 성공률로는 나올 수 없는 값입니다(k/30은 26.7, 30.0, 70.0, 73.3%). 연속량으로 읽어야 합니다.
- **Table 1은 기능 목록에 가깝습니다.** RoboSimStudio는 10개 행이 모두 ✓이고, RoboSimStudio만 ✓인 5행은 모두 이 논문이 정의한 워크플로입니다.

![Table 1 집계](images/table1_tally.png)

## 3. 저장소를 열어 보니: 스펀지는 손에 붙어 있었다

공식 저장소는 논문보다 훨씬 구체적이고 솔직합니다. 공개된 스펀지 닦기 twin의 README에 이런 기록이 있습니다.

- 마찰만으로 잡으면 닦는 동안 폼이 손 안에서 69–226 mm 미끄러지고, 닦인 비율이 0.69–0.76에 그칩니다. 마찰, 그립 힘, 누름 힘, 속도를 바꿔 30번 더 시도해도 미끄럼을 10 mm 아래로 줄이지 못했습니다.
- 그래서 잡기가 안정되면 패드 근처 폼 노드 106개를 손에 **고정(latch)** 합니다. 그러면 닦인 비율이 0.96이 됩니다. 막대도 블록에 용접되어 있습니다.
- 영상 속 부드러운 부엌 스펀지(약 20 kPa)는 집게 사이로 흘러나가 들리지 않아서, 80 kPa 폼으로 바꿨습니다.

![latch 효과](images/latch_effect.png)

흥미로운 점은 같은 저장소가 나중에 추가한 에이전트 작업 규칙입니다. 이 규칙은 완성된 twin의 조건으로 "손가락 접촉으로만 물체를 움직일 것, latch·weld·attachment 금지"를 둡니다. 논문 시점의 데모가 이 규칙을 지키지 않는 셈입니다. 논문의 강성 변형 범위(6·20·100 kPa)도 절반은 이 집게 조건으로 스펀지를 들 수 없는 값입니다.

## 4. 영상으로 맞출 수 있는 것과 없는 것

"영상과 맞을 때까지 고친다"가 물성에서는 얼마나 가능한지 간단한 모델로 계산해 봤습니다. 4 N으로 누른 스펀지의 압축량을 1080p 영상에서 0.5 px 정밀도로 읽는다고 가정했습니다.

![식별성 toy](images/identifiability.png)

- 강성이 약 100 kPa를 넘으면 E 추정의 상대 오차가 50%를 넘습니다. 공개 twin의 80 kPa와 100 kPa는 0.4σ 차이라 영상으로는 같은 값입니다.
- 힘 관측은 오차가 강성과 무관하게 일정해서, 단단한 물체일수록 유리합니다.

논문이 강성을 "맞추기"보다 넓게 "변형"하고 힘 피드백을 따로 둔 선택을 정량적으로 뒷받침하는 결과입니다. 반대로 에이전트가 "영상과 맞는 시뮬레이션"을 만들었다는 말은, 기하와 운동에는 강하고 물성에는 약하다는 뜻으로 읽어야 합니다.

## 5. 직접 쓸 수 있나

| 경로 | 판단 | 메모 |
|---|---|---|
| 도구로 쓰기 | 쓸 수 있음 | Apache-2.0, 설치 문서·예제·테스트. Linux x86-64 + Python 3.12 기준. NVIDIA GPU가 없으면 CPU 경로로 자동 전환 |
| 논문 결과 재현 | 불가 | 18개 과제 정의, GaP 닦기 정책, 30개 배치, UR5e가 공개 저장소에 없음 |
| GPU 없는 노트북 | 일부 | 강체·관절 장면과 에이전트 루프는 됨. 천·로프는 다른 solver로 대체, 테이프·종이·알갱이는 거부, 렌더러는 collider만 그림 |

![논문 Fig. 5](images/paper_fig5_tasks.png)

## 6. 정리

- 코딩 에이전트에게 시뮬레이션을 고치게 하는 **하네스**로서는 잘 만든 도구입니다. 저장하면 다시 빌드되는 페이지, 사진 대비 수치, gate, 장면 해시가 붙은 로그 구조는 다른 시뮬레이터로 옮겨 쓰기도 쉽습니다.
- 논문의 Real2Sim2Real 주장은 파일럿 하나이고, 공개 twin은 물리적 잡기를 latch로 우회합니다.
- Real2Sim에서는 **무엇을 영상으로 맞추고 무엇을 힘이나 랜덤화로 다룰지** 먼저 정해야 합니다.

분석 문서, 재계산 스크립트, 리포트 PDF는 함께 공개한 저장소에 있습니다.

## 자료

- [통합 리포트 PDF](report/report_public.pdf)
- [01. 논문 분석: RoboSimStudio (arXiv:2610.18835)](analysis/01_paper_analysis.md)
- [02. 선행연구와 계보](analysis/02_prior_work.md)
- [03. 수치 재계산과 toy 해석](analysis/03_analysis.md)
- [04. 코드 검증: 논문 ↔ 구현 대조, 수치 재계산, 구현 가능성](analysis/04_code_verification.md)

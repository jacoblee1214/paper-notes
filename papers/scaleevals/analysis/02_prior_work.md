# 02. 선행연구와 관련 자료: ScaleEvals

논문 번호 [n]은 ScaleEvals의 참고문헌 번호입니다. 확인일은 2026-11-26입니다.

## 1. 계보

ScaleEvals는 네 갈래를 한 파이프라인으로 묶습니다.

1. **실물 평가의 자동화.** AutoEval [3]은 과제별 성공 분류기와 자동 재설정으로 사람 개입을 줄였지만 이진 성공과 재설정 정책에 묶여 있습니다(논문 §2). 자율 개선 [11]과 Tether [15]는 기초 모델이 과제를 제안하는 연속 놀이를 정책 개선에 썼습니다. ScaleEvals는 같은 구조를 **평가**에 씁니다(§2 "Autonomous play"). PlayWorld [10]는 놀이 데이터로 세계 모델을 학습합니다.
2. **쌍대 비교 순위.** Bradley–Terry [19]와 LLM 쌍대 평가 [20]를 로봇 평가에 옮긴 것이 RoboArena [21]입니다. RoboArena는 사람이 두 정책의 롤아웃을 비교한 선호로 Elo를 매기고, ScaleEvals는 같은 BT 틀에 **자동 채점 점수에서 만든 소프트 목표**를 넣습니다(§3.3). 논문은 RoboArena를 경쟁이 아니라 로컬 평가 엔진으로 위치시킵니다(§2).
3. **VLM 채점.** 연속 진행도를 VLM으로 주는 Robometer [16]는 미세조정 모델이고, "Score the steps" [17]는 하위 목표별 VLM 채점을 제안했습니다. ScaleEvals의 AutoGrader는 후자에 가깝습니다: 마일스톤마다 yes/no VQA를 하고 가중합을 TP로 씁니다(§3.2).
4. **시뮬레이션·세계 모델 평가.** SIMPLER [1], RoboLab [2], PolaRiS [5], Ctrl-World [6], Interactive World Simulator [7]은 실물 평가를 대체하려는 쪽입니다. ScaleEvals는 이들이 구현체·카메라·접촉 영역에 민감하다는 점을 들어 실물 평가 자체의 확장을 택합니다(§1–2).

진단 쪽에서는 INT-ACT [4](의도는 알지만 실행이 실패한다는 분해 벤치마크)와 "jagged frontier" [22]를 동기로 삼고, 정책 비교의 A/B 검정 [23]은 같은 과제 분포를 전제하는 방법으로 대비시킵니다.

## 2. 구성 요소별 출처

| 구성 요소 | 가져온 곳 | ScaleEvals에서 바뀐 점 |
|---|---|---|
| 연속 놀이로 과제 생성 | 자율 개선 [11], Tether [15], AutoEval [3] | 정책 개선이 아닌 평가 목적. 과제·난이도·스킬 태그·채점표를 한 번에 생성(§3.1) |
| 쌍대 비교 → 전역 순위 | Bradley–Terry [19], RoboArena [21] | 사람 선호 대신 TP로 만든 소프트 목표 y = (P_i + ε)/(P_i + P_j + 2ε)(§3.3) |
| 일대일 매칭 | 헝가리안 할당(고전 알고리즘) | 난이도·동사·태그·주 물체 유사도를 가중치로(§3.3) |
| 마일스톤 VQA 채점 | "Score the steps" [17] | Gemini 3 Flash, 3 fps·10프레임 배치·한 장 건너 한 장, 마일스톤마다 독립 질의(§3.2, C.1) |
| 태그별 진단 | 일원 ANOVA와 BH 보정(일반 통계) | 정책 효과 η²를 스킬 태그별로. 한 롤아웃이 여러 태그에 들어감(§3.3) |
| 조건부 예측 | Borda 집계(일반) | 신뢰도·지지도 가중. 동사·스킬·물체 조건부 BT 순위를 합침(§5, B.3) |
| 정책 | GR00T N1 계열 [25], DreamZero [26], MolmoAct2 [27], π0.5 [24] | 모두 DROID Franka 위에서 평가(§4.1) |
| 플랫폼 | DROID [28] | YAM 양팔을 두 번째 구현체로 추가(§4.2) |


## 4. GitHub와 코드 자료 (확인 2026-11-26)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드** | 확인하지 못함 | - | - | 프로젝트 페이지 버튼이 "Code (soon)". GitHub 검색과 흔한 이름 후보로도 찾지 못함 |
| 프로젝트 페이지 | https://scaleevals.github.io/ | 페이지 하단에 템플릿 CC BY-SA 4.0 표기 | - | JS 앱. 수치와 시연 영상, 인터랙티브 데모. 템플릿(Academic Project Page)을 사용했다고 밝힘 |
| AutoEval (비교 대상 [3]) | https://github.com/zhouzypaul/auto_eval | MIT | 별 109 / 2026-03-26 | 자동 재설정과 성공 분류 |
| RoboArena (비교·보완 대상 [21]) | https://github.com/robo-arena/roboarena | MIT | 별 117 / 2026-04-28 | 분산 쌍대 평가 리더보드 |
| PolaRiS ([5]) | https://github.com/arhanjain/polaris | MIT | 별 252 / 2026-07-13 | 실제-시뮬 평가 |
| SIMPLER ([1]) | https://github.com/simpler-env/SimplerEnv | MIT | 별 1,191 / 2026-09-30 | 시뮬레이션 평가 |
| GR00T (평가된 정책, [25]) | https://github.com/NVIDIA/Isaac-GR00T | Apache-2.0 | 별 8,423 / 2026-11-17 | N1.6·N1.7이 이 저장소에 있는지는 확인하지 않음 |
| DreamZero ([26]) | https://github.com/dreamzero0/dreamzero | Apache-2.0 | 별 2,788 / 2026-04-19 | 평가된 정책 |
| MolmoAct2 ([27]) | https://github.com/allenai/molmoact2 | Apache-2.0 | 별 838 / 2026-10-02 | 평가된 정책 |
| openpi (π0.5, [24]) | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 별 14,772 / 2026-08-24 | 평가된 정책 |
| DROID ([28]) | https://github.com/droid-dataset/droid | LICENSE 메타데이터 없음 | 별 474 / 2025-09-15 | 평가 플랫폼 |

논문 속 "Gemini 3 Flash"는 호출형 상용 모델이고 AutoGrader 점수는 이 모델에 묶여 있습니다(논문 §6에서도 정확한 재현의 한계로 적음). 수동 평가 20과제의 초기 프레임, 롤아웃 영상, 채점 표는 공개 위치를 찾지 못했습니다.

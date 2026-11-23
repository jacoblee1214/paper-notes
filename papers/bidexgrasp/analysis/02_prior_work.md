# 02. 선행연구, 구성 요소의 출처, GitHub 자료

## 1. 선행연구 계보

이 논문은 세 줄기가 만나는 자리에 있습니다. 번호는 v2 참고문헌 번호입니다.

1. **한 손 합성 파지 데이터와 최적화.** Liu 외의 미분 가능한 force-closure([14]), DexGraspNet([10], 1.32 M 파지), BODex([25], ICRA 2025, 이중 최적화와 QP 에너지), GraspQP([26]), Dexonomy([27]). 이 논문의 합성 파이프라인은 BODex 위에 얹혀 있고(v1 §IV-A "built upon [2]", v2 부록 B.2의 ReLU-QP와 cuRobo 에너지), 6방향 교란 QP 에너지와 사전 파지(접촉 거리 1 cm)를 그대로 가져옵니다.
2. **양손 파지 합성·학습.** BimanGrasp([19], RA-L 2024, 0.15 M 파지, 물체 900개), DHAGrasp([20], 텍스트 조건 양손 파지, 1.30 M), UltraDexGrasp([23], 2026, 합성 데이터로 양손 범용 파지), ArtiGrasp([21])와 Bi-DexHands([22])의 강화학습 계열. 논문의 위치: 데이터와 크기 범위를 키운 쪽(표 1).
3. **생성 모델로 파지 자세 만들기.** UniDexGrasp([16]), DexGraspNet 2.0([11]), SceneDiffuser([17]), DGTR([43], 같은 연구실의 CVPR 2024), Grasp as You Say([35]), UGG([34]), DexGrasp Anything([36]). 프레임워크는 DDPM([41])에 뷰 쌍 예측과 앵커 상대 표현을 더한 구조이고, §2.3은 "drawing inspiration from prior methods [11]"이라고만 적습니다(어느 요소가 [11]에서 왔는지는 명시하지 않음).

**이 논문이 채우는 자리:** 양손 데이터가 작고 크기 범위가 좁았던(최대 40 cm) 자리를 대규모(80 cm)로 넓히고, 양손 뷰 쌍 예측으로 두 손의 조율을 명시적으로 모델링합니다. 방법 자체는 BODex의 양손 확장(영역 쌍 초기화 + 손별 에너지 분리)과 DexGraspNet 2.0 계열의 생성 모델 구성을 참고한 조합이라, 새로운 부분은 **영역 쌍 선택 → 손별 분리 에너지 → 뷰 쌍 조건부 표집** 구성에 있습니다.

## 2. 구성 요소별 출처 (논문 인용 기준)

| 구성 | 출처 (논문 번호) | 비고 |
|---|---|---|
| QP 에너지, 사전 파지(1 cm), 쥐기 자세 2G−G_pre, 평가 지표(SUC, PD, SPD, CDC) | BODex [25] | 지표 정의("Following [25]")와 평가 임계값(5 cm, 15°)이 BODex 평가 도구와 같음 (`03` V12) |
| GWB 추정기 | TDG, Chen 외 2023 [39] | 영역 쌍 점수(식 3) |
| 하위 QP 풀이 | ReLU-QP [47] | PyTorch ADMM. v2 부록 B.2 |
| 관절 한계, 자기 관통, 손 사이 관통 에너지 | cuRobo [48] | 부록 B.2 "directly adopt" |
| 물체 | DexGraspNet [10] 2,397개, Objaverse [37] 3,954개 | v1 §IV-A |
| 시뮬레이터 | MuJoCo [38], Shadow Hand | 마찰 0.6/0.02 |
| 점군 인코더 | PointNet++ [40] | |
| 생성 | DDPM [41], Chamfer 손실 [42] | |
| 뷰 후보와 정답 뷰 확률 | §2.3이 [11]에서 영감을 받았다고만 표기 | 단일 뷰와 쌍 뷰 확률 라벨의 설계는 이 논문 서술 기준 |
| 기준선 | SceneDiffuser [17], DGTR [43] | 부록 F.1: 출력 차원을 두 배로 확장해 양손으로 개조 |
| 실로봇 | Grounded-SAM [44], Hyper3D [45], FoundationPose [46], 재타게팅 [35, 6] | 3D 재구성은 상용 서비스 |

## 4. GitHub와 자료 표

조회일 2026-11-23 (`gh api`). 이 논문 공식 코드는 아직 없습니다.

| 저장소 / 자료 | 관계 | 라이선스 | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [BiDexGrasp 프로젝트 페이지](https://frenkielm.github.io/BiDexGrasp.github.io/) | 공식 프로젝트 페이지(저장소 frenkielm/BiDexGrasp.github.io) | 없음 | 0 | 2026-10-12 | 페이지에 "Dataset and Code (Coming soon)". 코드 링크 없음 |
| 공식 코드 저장소 | 없음 | – | – | – | 찾지 못함(저자 계정 frenkielm, wyl2077, Rain-Shuoyu 조회) |
| [maxtensor33/bidexgrasp](https://github.com/maxtensor33/bidexgrasp) | **동명의 다른 논문** ("Bimanual Dexterous Grasping of Single and Dual Objects", IROS 2026 익명 저장소, README에 "rejected by IROS 2026") | 파일 없음 | 1 | 2026-07-16 | 이 논문과 무관. Isaac Gym 기반. 혼동 주의 |
| [JYChen18/BODex](https://github.com/JYChen18/BODex) | 합성 파이프라인의 기반(cuRobo 포크) | NVIDIA License (API: NOASSERTION) | 156 | 2026-04-21 | 정적 확인만 (`06b9a3c`) |
| [JYChen18/DexGraspBench](https://github.com/JYChen18/DexGraspBench) | BODex 평가 도구(MuJoCo). 마찰 0.6/0.02, 5 cm, 15°가 설정에 있음 | 없음 | 67 | 2025-09-21 | 설정 파일만 읽음 (`d9ea6cf`) |
| [JYChen18/DexLearn](https://github.com/JYChen18/DexLearn) | BODex 계열 학습 코드 | 없음 | 52 | 2026-10-19 | 참고 |
| [yanming03/BimanGrasp-Generation](https://github.com/yanming03/BimanGrasp-Generation) | 기준선 BimanGrasp의 오픈소스 구현 | NOASSERTION | 24 | 2025-12-02 | 비교 대상 |
| [yanming03/BimanGrasp-Dataset](https://github.com/yanming03/BimanGrasp-Dataset) | BimanGrasp 데이터셋 | MIT | 24 | 2025-09-08 | 비교 대상 |
| [wyl2077/DGTR](https://github.com/wyl2077/DGTR) | 기준선 DGTR(같은 연구실) 공식 코드 | MIT | 0 | 2025-07-28 | 기준선 재구현 근거 |
| [Sage-PhysicalAI/UltraDexGrasp](https://github.com/Sage-PhysicalAI/UltraDexGrasp) | 양손 합성 데이터 파지(ICRA 2026) | Apache-2.0 | 93 | 2026-03-06 | 가장 가까운 공개 비교 대상 |
| [PKU-EPIC/DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) | 한 손 대규모 데이터 | 없음 | 483 | 2025-01-06 | 물체 원천 |
| [NVlabs/curobo](https://github.com/NVlabs/curobo) | BODex의 기반 라이브러리 | Apache-2.0 | 2,039 | 2026-11-21 | 참고 |
| DHAGrasp (arXiv:2509.22175) | 표 1의 비교 데이터셋 | – | – | – | 공개 저장소를 찾지 못함 |

- 라이선스 전망: 이 논문 합성 코드가 BODex(cuRobo 포크)에서 파생되었다면 공개 시 NVIDIA License 또는 비상업 조건을 이어받을 가능성이 있습니다. 공개 전이라 확정할 수 없습니다.
- 이 저장소의 clone은 만들지 않았습니다(공식 코드 없음). 참고 저장소는 `gh api`로 설정 파일 두 개만 읽었습니다.

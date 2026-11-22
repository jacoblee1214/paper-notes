# 02. 선행연구와 관련 저장소

이 문서는 Kamino의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드와 관련 저장소가 어떤 상태인지를 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References, v2)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| Kamino의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 최대 좌표 표현 | Haug 1989 [21], Brüdigam & Manchester [13] (선형 시간 변분 적분기), Dojo [14] (내점법 NCP 솔버, 미분 가능) | 같은 표현. Dojo는 미분 가능성과 내점법, Kamino는 GPU 병렬과 ADMM을 택함 |
| 구속 동역학과 쿼터니언 | Maloisel 외 [15] (SIGGRAPH 2025) | 쿼터니언 기반 구속 강체 동역학 정식화. 같은 연구실 |
| 포워드 다이내믹스 솔버 비교 | Tsounis, Grandia, Bächer [16] (arXiv:2504.19771) | 식 유도와 솔버 비교의 본체. 본문은 "철저한 수학적 설명은 [16]"이라고 위임 |
| 접촉 법칙 | Moreau [31, 33], Glocker [32], Signorini 조건, Coulomb 마찰과 최대 소산 원리 | 부록 A.1.1의 서술 근거 |
| de Saxcé 보정 (이중 쌍포텐셜) | de Saxcé & Feng [25, 34], Acary 외 [26] | 접선 슬라이딩 속도로 법선 속도를 보정해 비연합 쿨롱 법칙을 원뿔 상보성으로 만듦 |
| 시간 이산화 | 반암시적 오일러, Moreau–Jean 중점 [36], Stewart–Trinkle [35] | 부록 A.1.2 |
| 구속 안정화 | Baumgarte [27] | 가중 α(관절), β(한계), γ(접촉) |
| ADMM | Boyd 외 [17], Tasora 외 [18] (변분 부등식으로 비매끄러운 동역학) | 일반 이론 |
| 근위 접근 | Carpentier 외 [19] (RSS 2021), [20] (RSS 2024: 컴플라이언트에서 강체 접촉으로) | PADMM의 직접 선행. [20]이 NSOCP와 NCP의 동치 증명 스케치를 제공(부록 A.1.2) |
| 가속과 재시작 | O'Donoghue & Candès [37] | 부록 A.2.1 |
| matrix-free Delassus | Sathya 외 [38] (RA-L 2026), Macklin 외 [11] (켤레 잔차) | 큰 시스템의 CR 경로 |
| 비교 솔버의 접촉 모델 | MuJoCo [3] (볼록 완화, 소프트 구속), PGS 계열(Isaac Gym [1], PhysX), XPBD | §1의 대비 대상 |
| 루프 있는 로봇 | DR Legs [9] (Gim 외 2018), 굴삭기 [10], 역기구학 [12] | 대상 로봇과 FK 솔버 |
| 인프라 | Warp [23], Newton [22], MJWarp [5], MJX [4], Isaac Lab [24] | 실행 기반과 학습 연결 |
| 학습 | PPO [39], Rudin 외 [6], Orbit [7] | RL 설정 |
| 대상 로봇 | BDX [28] (RSS 2024), Olaf [29] (RA-L 2026), Ant [30] | 같은 연구실 로봇과 표준 과제 |

**가까운 대안.**
- **MJWarp(+ 등식 구속).** 루프를 트리 + 닫힘 구속으로 근사합니다. 논문이 가장 길게 비교하는 상대이며 튜닝(`eq_solref`, 시간 상수)이 필요하다고 주장합니다(§3.1, §3.3).
- **XPBD(Newton).** 위치 수준 반복으로 하드 구속에 가깝지만 질량비와 접촉에서 반복이 많이 필요합니다.
- **Featherstone 계열(Newton의 `featherstone` 솔버), PhysX.** 트리를 전제하는 최소 좌표 방식입니다.
- **Dojo, Brax류 미분 가능 엔진.** 최대 좌표를 쓰지만 GPU RL 처리량 쪽은 아닙니다.

**같은 저장소 안의 형제 솔버.** 현재 코드에는 PADMM 외에 SCC(구 DVI), LOX 솔버가 같이 들어 있고 문서는 "SCC는 더 빠르지만 부등식을 덜 정확하게 푼다"고 설명합니다(`04` D2). 논문은 PADMM만 다룹니다.

## 2. 코드와 저장소 (GitHub 표)

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|---|
| [newton-physics/newton](https://github.com/newton-physics/newton) (`newton/_src/solvers/kamino`, `newton/examples/kamino`) | 공식 구현. 프로젝트 페이지의 Code 링크 | Apache-2.0 | 5,977 | 2026-11-21 | `f39d8199f18d5117f9dd8073c1eb71c9a196f020` (2026-11-21). 솔버 디렉터리 최근 커밋 `21228c19` (2026-11-20), 디렉터리 이력 214개 | 얕은 sparse 복제(`repo/`, 20 MB)로 정적 읽기만. 실행, 설치, import 안 함 |
| Newton 릴리스 v1.8.0 | 최신 릴리스 | Apache-2.0 | — | 2026-11-11 게시 | — | 받지 않음 |
| [kamino.disneyresearch.com](https://kamino.disneyresearch.com/) | 프로젝트 페이지 | 페이지 자체는 명시 없음 | — | — | 2026-11-22 읽음 | 예제 링크, 사용 예(DR Legs USD는 `newton.utils.download_asset("disneyresearch")`) |
| [NVIDIA/warp](https://github.com/NVIDIA/warp) | 실행 기반 (Python → CUDA/CPU 커널) | Apache-2.0 | 7,338 | 2026-11-22 | — | 설치 안 함 |
| [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) | 비교 대상 MJWarp | Apache-2.0 | 1,649 | 2026-11-20 | — | 읽지 않음 |
| [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) | 비교 대상 MuJoCo CPU | Apache-2.0 | 18,385 | 2026-11-21 | — | 읽지 않음 |
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | RL 연결 (Newton 솔버를 물리 백엔드로 지원) | GitHub API는 NOASSERTION | 8,699 | 2026-11-22 | — | 읽지 않음 |
| BDX, Olaf, Iron Man 모델과 RL 환경 | 논문의 주요 실험 대상 | 공개되지 않음(프로젝트 페이지 "proprietary") | — | — | — | 코드나 자산을 찾지 못함 |

- 별과 push 시각은 `gh api`로 2026-11-22에 읽은 값입니다.
- 별도 Kamino 저장소는 없습니다. 논문 본문 [22]도 Newton(Apache-2.0)을 인용합니다.
- 솔버 README는 "BETA 1, 사용을 권하지 않음, 기여는 받지 않음, BETA 2는 2026년 여름 예정"이라고 적습니다. 이 글을 쓰는 시점(2026-11-22)에도 BETA 2 안내는 갱신되지 않았습니다(`04` D1).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **PADMM이 PGS보다 병렬 GPU 환경에 맞다는 주장**은 [16, 20]의 CPU 결과가 근거입니다. 이 논문의 새로운 부분은 GPU 구현과 이종 세계 지원, 솔버 구성의 정리이지 알고리즘 자체의 수렴 이론이 아닙니다.
- **de Saxcé 보정이 필요하다는 서술**은 [25, 26, 34]의 이중 쌍포텐셜 이론에 기댑니다. 이 논문의 MJWarp 비교는 "MJWarp가 이 항이 없어 슬라이딩에서 울렁인다"는 가설을 상자 미끄럼 한 예로만 보여 줍니다.
- **Nesterov 재시작**은 [37]의 일반 규칙이고, 이 코드는 재시작 임계를 0.999로 둡니다(`04` D7).

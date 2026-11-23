# 02. 선행연구, 가까운 대안, 코드 저장소

## 1. 계보

### 1.1 평가 쪽 (벤치마크)

| 벤치마크 | 대상 | HumanoidToolBench와 다른 점 (논문 표 I, §II 기준) |
|---|---|---|
| LIBERO, RoboTwin 2.0 | 단일 팔, 양팔 조작 VLA | 휴머노이드 아님, 도구 선택을 변수로 두지 않음 |
| HumanoidBench (RSS 2024) | 휴머노이드 전신 이동·조작, 숟가락·창문 닦기 포함 | 기능이 대조되는 도구 중 고르는 평가가 없음 |
| SIMPLE, WOLF-VLA | 휴머노이드 VLA 평가 | SIMPLE은 실패 위치 단계화는 하나 요구되는 판단을 바꾸지 않고, WOLF-VLA는 사람 시연을 공개하지 않음 |
| PhysToolBench | 도구 이해 VQA | 행동하지 않음 |
| DexCraft(「From Grasps to Dexterity」), RoboWits | 고정 팔 도구 파지, 양팔 도구 선택 | 단계적 수준이 없음, RoboWits만 판단을 변화 |

차별점은 (i) 휴머노이드, (ii) 선택·정지 사용·이동 사용 3단 수준, (iii) 표준/디코이 쌍, (iv) 사건 기록입니다. 시뮬 환경과 제어 스택은 SIMPLE(MuJoCo 휴머노이드 시뮬)에서 코드를 가져왔고(`LICENSE` 저작권 표기, 코드 주석), 다리 제어는 GR00T-WholeBodyControl에서 가져왔습니다(§III-C, 저장소 설명). L0 성공 정의(0.1 m 중 80%)는 SIMPLE의 pick 정의를 그대로 씁니다(코드 주석, `04` D3).

### 1.2 평가 대상 정책

IL: ACT, Diffusion Policy. VLA: π0.5, Ψ0(SIMPLE 팀의 휴머노이드 기반 모델), GR00T N1.7. WAM(세계-행동 모델): Cosmos 3 Edge Policy, FastWAM(테스트 시 미래 상상 없이 행동). 정책 구현이나 학습 코드는 이 저장소에 없습니다(`04` D1).

### 1.3 가까운 대안

- **도구 사용 계열:** SimToolReal(제로샷 정교 도구 조작, 고정 손), DexCraft류 도구 파지 사전학습. 휴머노이드 선택-사용 통합은 이 논문이 처음이라는 주장은 표 I의 7개 비교에 근거합니다(저자 확인 범위 밖의 연구는 검토하지 않았습니다).
- **LLM 에이전트 도구 사용:** 페이지의 에이전트 실험은 Code as Policies류(CAP, 한 번에 프로그램)와 한 명령씩 실행하는 방식(RUA)입니다. 논문 v1에는 없습니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 |
|---|---|
| 시뮬 엔진과 과제 틀 | MuJoCo, SIMPLE에서 파생(MIT) |
| 전신 제어 | GR00T-WholeBodyControl(SONIC 계열, 논문 참고문헌 [19]) Balance/Walk 정책 |
| 도구 메시 | Objaverse → MolmoSpaces(자산 164개 해시 고정, `resources/evaluation_assets.json`) |
| 원격조작 | Meta Quest |
| ACT·DP 학습 | SIMPLE/Ψ0 계열 학습 틀(모델 카드의 `psi0_source_revision`) |
| 텍스트 조건 | 고정 CLIP ViT-L/14 |
| 데이터 형식 | LeRobot |

## 4. GitHub와 프로젝트 자원

| 저장소·자원 | 관계 | 라이선스 | 별 | 최근 푸시 | 기준 | 상태 |
|---|---|---|---|---|---|---|
| [SNU-PI/HumanoidToolBench](https://github.com/SNU-PI/HumanoidToolBench) | 공식 코드(평가 도구) | MIT (`LICENSE` 파일. GitHub 자동 감지는 NOASSERTION: SIMPLE 저작권 문구 추가) | 25 | 2026-11-22 | `b53c7aaad4499eb8a23f7c2defb2dce4d162c227` (2026-11-23 얕은 복제, 커밋 18개) | 정적 읽기만 (`repo/`) |
| [프로젝트 페이지](https://snu-pi.github.io/HumanoidToolBench/) | 논문 초록이 가리키는 주소 | 페이지 하단: 코드 MIT, ToolBook CC BY-NC 4.0 | | 2026-11-23 갱신 | | 읽음 (`paper/project_page.html`) |
| [snupilab/humanoidtoolbench-teleop](https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop) | ToolBook | CC BY-NC 4.0 | 좋아요 16, 다운로드 24,905 | 2026-11-19 | `1ef5be08` | 받지 않음(저장 76.5 GB, 메타데이터만) |
| [snupilab/humanoidtoolbench-act-sim-3003](https://huggingface.co/snupilab/humanoidtoolbench-act-sim-3003) | ACT 체크포인트 (논문의 ACT) | 모델 카드에 표기 없음 | 0 | 2026-10-02 | `6cf41e6a` | 모델 카드·메타만 읽음 (가중치 209 MB, 받지 않음) |
| [snupilab/humanoidtoolbench-dp-sim-3003](https://huggingface.co/snupilab/humanoidtoolbench-dp-sim-3003) | DP 체크포인트 | 모델 카드에 표기 없음 | 1 | 2026-10-02 | `e79b28b7` | 받지 않음 (가중치 339 MB ×2 파일) |
| [physical-superintelligence-lab/SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE) | 시뮬 틀의 원천 | MIT 표기 (GitHub NOASSERTION) | 375 | 2026-11-16 | | 확인만 |
| [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) | 다리 제어기 원천 | 자체 조항(GitHub: Other) | 3,940 | 2026-11-03 | | 확인만 |
| [allenai/molmospaces](https://github.com/allenai/molmospaces) | 도구 메시 출처 | 자체 조항(GitHub: Other) | 527 | 2026-11-14 | 저장소가 고정한 `5f802a46` | 확인만 |
| [SNU-PI/decoupled_wbc](https://github.com/SNU-PI/decoupled_wbc) | 제어기 포크(설치 스크립트가 가져옴) | 표기 없음 | 0 | 2026-09-05 | | 확인만 |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | GR00T N1.7 | Apache-2.0 | 8,407 | 2026-11-17 | | 확인만 |

- 공식 코드가 공개하는 것은 평가 도구이고, π0.5, Ψ0, GR00T N1.7, Cosmos 3, FastWAM의 체크포인트와 모든 정책의 학습 코드, 그림 6·7 탐침 코드, 에이전트 실험 코드는 없습니다(`04` D1).
- 데이터셋 저장소에는 `raw/`, `hardware/`, `trimmed/`, `rendered/`, `rendered_images/`, `viewer/`가 있고 전체 76.5 GB입니다. 실로봇 `hardware/`의 에피소드 수는 29 + 21 + 20 + 21 = 91로 표 II와 맞습니다(트리 API로 파일 수만 셈).

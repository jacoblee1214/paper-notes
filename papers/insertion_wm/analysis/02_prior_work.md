# 02. 선행연구와 관련 저장소

확인일: 2026-11-10. 논문 본문의 참고문헌 번호는 [번호]로 씁니다.

## 1. 계보

| 축 | 대표 연구 | 이 논문과의 관계 |
|---|---|---|
| 조립 시뮬레이션과 데이터 | Joinable [4], Assemble-them-all [5] (CAD 조립 자산과 역분해 시연) | 자산 100쌍과 역분해 시연의 출처 |
| 특수 정책 (RL, 저차원 상태) | IndustReal [6] (PPO, 접촉 보상), Force-guided exploration [7] | 보상 형태(식 5)를 가져온 출처 |
| 생성 정책 (CAD 점군 조건) | **AutoMate [1]** (RSS 2024, arXiv:2407.08028) | 주 기준선. 특수 PPO 학습, 생성 정책은 BC·DAgger·RL 증류 |
| 잠재 세계 모델 + 계획 | TD-MPC [30], **TD-MPC2 [31]**, 다과제 TD-MPC2 [32] | 이 논문의 기반 알고리즘 |
| 시각 세계 모델 (조작) | MoDem-V2 [18], Finetuning offline WM [19] | 같은 저자 계열의 선행 |
| 능동 지각 조립 | Fabrica [42] (arXiv:2506.05168) | 저자들이 향후 방향으로 언급 |

**이 논문의 위치.** 방법 자체는 TD-MPC2를 그대로 쓰고, 새로운 점은 (1) 깊이 + 고유감각 인코더(식 6), (2) 90개 조립품 다과제 학습, (3) 미세조정 실험입니다. 알고리즘 기여가 아니라는 점은 저자들도 씁니다(I절).

## 2. 가까운 대안 (읽은 것과 제목만 확인한 것)

- **AutoMate [1]:** 논문 본문과 웹 검색으로 확인. 프로젝트 페이지는 bingjietang718.github.io/automate. 공식 코드는 찾지 못했습니다(§4).
- **EasyInsert (arXiv:2505.16187):** 데이터 효율적이고 일반화되는 삽입 정책이라는 검색 결과의 제목만 확인했습니다. **읽지 않았습니다.** 이 논문의 "unseen 삽입" 주장과 비교하려면 따로 읽어야 합니다.
- **Fabrica [42]:** 본문에서 능동 지각 방향으로 언급만 합니다. 읽지 않았습니다.

## 3. 이 저장소의 다른 분석과의 관계

| 폴더 | 관계 | 이 논문에서 가져올 것 |
|---|---|---|
| `openwam/` (arXiv:2609.07398) | 세계 행동 모델(WAM). 영상 생성 prior 기반이라 잠재 모델과 목적이 다름 | "세계 모델이 무엇을 예측하는가"의 대비 (영상 대 잠재) |
| `agentic_wam/` (arXiv:2610.30136) | 영상·행동을 한 모델에서 함께 denoise하는 WAM | 같은 질문: 예측 대상이 영상이어야 하나 |
| `contact_disassembly/` (CONTACT, arXiv:2603.08560) | 접촉이 많은 분해 과제, 시각 + 촉각 비교 | 실물 접촉 삽입에서 시각만으로 충분한지 |
| `dexforce/` (arXiv:2501.10356) | 손끝 힘을 목표에 더하는 시연 정책 | 힘 정보 대 시각 정보 |
| `reactive_grasp/` (arXiv:2609.30082) | 접촉력 분배 QP, 힘 제어 | 접촉 모델 기반 제어의 난이도 |
| `stiffness_copilot/` (arXiv:2603.14068) | 손목 영상 정책이 임피던스 행렬을 고르는 공유 제어 | 삽입의 임피던스 설계 대 학습 정책 |
| `chronospolicy/` (arXiv:2610.13929) | 시간 정보 주입, 센서 지연 무작위화 | 실물 전환에서 지연 처리 |

## 4. GitHub와 허깅페이스 (논문 본문에서 찾은 것과 검색으로 찾은 것)

논문 본문에는 URL, 프로젝트 페이지, 코드 링크가 **없습니다**(`paper/paper.txt` 전체 검색). arXiv 초록 페이지에도 코드 링크가 없습니다. 아래는 `gh search repos`와 `gh api`로 확인한 결과입니다.

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | 별 | 최근 커밋 |
|---|---|---|---|---|---|
| (InsertionWM 코드) | `gh search repos`, 코드 검색("2609.28258", "InsertionWM"), 웹 검색 모두 결과 없음 | — | — | — | — |
| [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) | TD-MPC2 공식 구현(방법의 기반) | 방법 원저자 공식 | MIT | 999 | `332f19e` (2026-10-20) |
| [bingjietang718/automate](https://github.com/bingjietang718/automate) | AutoMate **프로젝트 웹사이트 소스**(README에 CC BY-SA 4.0 웹사이트 표기). 정책 학습 코드가 아닙니다 | 저자 계정, 코드 아님 | 명시 없음(웹사이트 CC BY-SA 4.0) | 0 | `ce95071` (2024-10-26 push) |
| [isaac-sim/IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) | Isaac Gym 강화학습 환경 모음. AutoMate 환경이 이 계열에 기반했는지는 확인하지 못함 | NVIDIA 공식 | NOASSERTION (LICENSE 파일 확인 필요) | 2,950 | `aeed298` (2024-10-26 push) |
| [ShaneFlandermeyer/tdmpc2-jax](https://github.com/ShaneFlandermeyer/tdmpc2-jax) | TD-MPC2의 JAX 구현(커뮤니티) | 비공식 | MIT | 82 | push 2026-07-28 |

- **AutoMate 코드:** 논문 VI-A는 "All AutoMate results are reproduced using the official codebase"라고 쓰고, VI-B 각주는 "released codebase"를 말합니다. 그러나 `bingjietang718/automate`는 웹사이트 소스뿐입니다. **논문이 말하는 공식 학습 코드는 이 검색으로는 찾지 못했습니다.** 따라서 AutoMate 수치를 우리 조건에서 재현할 방법이 없습니다.
- **TD-MPC2 기본 설정(`tdmpc2/config.yaml`, commit `332f19e`):** 호라이즌 3, 샘플 512, 엘리트 64, 반복 6, 정책 궤적 24, 온도 0.5, 잠재 차원 512, 가치 bins 101. 이 논문의 설정이라고 확정할 수 없습니다. 논문에 명시가 없기 때문입니다(`03_analysis.md` §7).
- **Isaac Gym:** 논문 시뮬레이션의 기반입니다(III절 "we leverage Isaac"). CUDA GPU가 있어야 하므로 이 맥에서는 실행할 수 없습니다.
- **체크포인트와 데이터:** 확인된 것이 없습니다. 자산 100쌍의 원본은 AutoMate 측이 공개했는지 확인하지 못했습니다.
- **HF:** 이 논문 전용 모델이나 데이터셋은 검색되지 않았습니다.

**결론.** 공식 코드가 없습니다. 재현 가능한 자산은 TD-MPC2 공식 구현(MIT)과 Isaac Gym 환경 모음뿐입니다. 이 맥에서 TD-MPC2 코드를 그대로 쓰는 것은 가능하지만(torch 설치 필요), 조립 환경과 자산은 직접 만들어야 합니다.

## 5. 이 논문을 읽을 때 같이 볼 것

1. AutoMate [1]의 실험 설정과 생성 정책 수치(7%의 출처).
2. TD-MPC2 [31]의 계획 설정과 MPPI 세부(샘플, 호라이즌).
3. TD-MPC2 다과제 [32]: 다과제 확장에서 학습 규모 효과가 어떻게 보고되는지.
4. Fabrica [42]: 능동 지각 방향과 이 논문의 한계(VII절)의 연결.

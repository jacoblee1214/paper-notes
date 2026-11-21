# 02. 선행연구와 관련 저장소

이 문서는 TacInno의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드가 있는지를 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| TacInno의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 촉각을 입력으로 넣는 visuo-tactile VLA | VLA-Touch [2], OmniVTLA [11], Tactile-VLA [16], AT-VLA [22], TAP-VLA [32], FTP-1 [40], N0-VTLA [26], VTLA [42], TacVLA [43], TacForcing [49] | TacInno가 비판하는 첫 번째 방향. 행동 손실만으로 촉각 사용을 정하므로 이득이 일관되지 않다고 논문은 봅니다 (§1) |
| 촉각을 예측 목표로 쓰는 정책 | SimpleTouch [38], Representation-aligned tactile grounding [9], DeCAL [13], Physical words [24], TacDyn-WAM [33], ContactWorld [46] | 두 번째 방향. 예측 목표 전체가 "예측 가능한 부분"을 많이 담는다고 봅니다 (§1) |
| 변화량으로 손실을 가중하는 정책 | τ-VLA [10], Dream-Tac [25], Tactile-WAM [36], HiTac-WAM [37], ME-Dex 1.0 [45] | 세 번째 방향. 저자들은 가중치가 큰 변화가 오히려 예측 가능한 부분이라 도움이 안 된다고 봅니다 (§1, Fig. 4) |
| 가장 가까운 선행 연구 | ResTacVLA [44] | 촉각 표현과 영상 예측의 잔차를 양자화해 **정책 입력**으로 씁니다. TacInno는 같은 잔차 개념을 **학습 타깃**으로만 씁니다 (§2) |
| 잔차 개념의 기원 | Kailath [17] (1968), Kalman [18] (1960) | 추정 이론의 innovation, 즉 관측과 예측의 차이라는 개념을 가져왔습니다 (§2) |
| 잠재 동역학을 이산 코드로 학습 | VQ-VAE [31] (van den Oord 2017), Genie [6], LAPA [39], UniVLA [7] | TacInno 토크나이저는 인코더-디코더 구조와 VQ를 잠재 행동 모델 [6, 7, 39]에서 가져옵니다 (§3.2) |
| 인과 시간 주의와 공간-시간 Transformer | Genie [6] | 프레임 간 인과 주의, 프레임 내 공간 주의 (§3.2) |
| 촉각 센서 | GelSight [41], DIGIT [20] | 시뮬레이션은 GelSight 계열을 Taxim [30]으로 렌더링합니다. 실로봇은 Xense 센서입니다 (§3.1, §4.1) |
| 영상 특징 | DINOv2 [28] | 동결된 영상 특징을 토크나이저에 씁니다 (§3.2) |
| 기반 VLA | π0 [4], π0.5 [3], GR00T N1 [27], OpenVLA [19], RT-2 [5] | TacInno-VLA는 π0.5 (PaliGemma + Gemma 300M 행동 전문가)를 씁니다 (§3.3) |
| 시뮬레이션 벤치마크 | UniVTAC [8] | 시뮬레이션 6과제와 평가 프로토콜의 출처입니다 (§4.1). 논문 [8]의 학회 표기는 "CoRL 2026"입니다 (이 논문의 학회가 아닙니다) |
| 기준선 ACT, Diffusion Policy | ACT [48], Diffusion Policy [12] | UniVTAC-ACT 기준선의 원형입니다 (§2, Table 1) |
| 원격조작 수집 | GELLO [35] | 실로봇 시연 수집에 씁니다 (§4.1) |

**가장 가까운 대안과의 차이.** TacInno와 ResTacVLA [44]는 둘 다 영상으로 설명되는 촉각 잔차를 다룹니다. 차이는 위치입니다. ResTacVLA는 잔차를 정책의 입력으로 넣고, TacInno는 잔차를 학습 목표로만 씁니다 (§2, "our code encodes the future tactile change ... and serves only as a training target"). 논문의 절제 실험은 영상만으로 조건을 준 토크나이저(vision-only conditioning, 이는 ResTacVLA의 시각 잔차에 해당)가 21 pp 중 7.5 pp만 회복한다고 보여줍니다 (표 2, `03` §2).

## 2. 공식 코드와 저장소 (GitHub 표)

| 저장소 | 관계 | 라이선스 | 대조 기준 | 상태 |
|---|---|---|---|---|
| [LisavilaLee/TacInno](https://github.com/LisavilaLee/TacInno) | 공식 구현. 프로젝트 페이지의 Code 링크. 토크나이저, TacInno-VLA, UniVTAC 평가 도구, 설정 파일 | Apache-2.0 (LICENSE 파일, GitHub API `apache-2.0`). `NOTICE`에 openpi 등 제3자 라이선스 목록 | 커밋 `e3c81f9` (2026-11-20, 얕은 복제). 저장소 생성 2026-11-18, 최근 push 2026-11-20, 별 4개 | **정적 읽기만** (`repo/`, gitignore). 실행, 설치, import 안 함 |
| [TacInno/tacinno.github.io](https://github.com/TacInno/tacinno.github.io) | 프로젝트 페이지 소스 (GitHub Pages) | 라이선스 없음 (API `null`) | 최근 push 2026-11-20, 별 1개 | 복제 안 함. 프로젝트 페이지 HTML만 읽음 (`tacinno.github.io`) |
| Hugging Face `Lisavila/TacInno-VLA` | 체크포인트 (시뮬 6과제, 실로봇 4과제, 토크나이저). 약 53 GB | README와 model zoo에 따르면 π0.5에서 미세조정. Gemma 이용 약관을 따른다고 적음 | — | **받지 않음** (가중치 다운로드 금지) |
| Hugging Face dataset `Lisavila/TacInno-UniVTAC-Demos` | 시뮬 시연 데이터 8과제 중 100개씩. 약 112 GB | 저장소 README에 적힌 정보만 확인 | — | **받지 않음** |
| [UniVTAC](https://github.com/univtac/UniVTAC) | 시뮬레이션 벤치마크와 Isaac Sim 평가 환경 (외부 의존) | 외부 | — | 설치 안 함 |
| openpi (Physical Intelligence) | π0.5 PyTorch 구현 (`third_party/openpi`, 저장소 안에 포함) | Apache-2.0 | — | 정적 읽기만 |
| FTP-1 코드 | 학습 코드의 출발점(README, Acknowledgements). 같은 저장소에 포함되어 있는지 확인 안 함 | — | — | 확인 안 함 |

공식 저장소의 README는 "Reproduce the simulation results with the released checkpoints"를 목표로 한다고 적고, 체크포인트 평가 결과 표를 제공합니다 (`repo/README.md`). 이 표와 논문 표 1의 값은 같습니다 (`04` §2).

## 4. 이 논문의 주장이 기대는 선행 결과

- **촉각 가리기 검사.** 시뮬레이션에서 촉각을 가리면 TacInno-VLA만 Lift Can에서 크게 떨어집니다 (표 3). 이 검사는 Fig. 1의 "입력이 예측 가능한 부분만 담으면 정책은 입력을 무시한다"는 가설과 맞습니다.
- **정보 이론의 innovation.** 정의식은 추정 이론의 innovation [17, 18]과 같은 형태이지만, 논문은 조건 기대를 데이터에서 추정하는 방법을 보이지 않습니다. 이 부분은 `04`의 수치 확인(§3, 식 2)에서 다룹니다.
- **보조 손실의 근거.** 코드 손실이 행동 전문가의 촉각 토큰을 바꾼다는 결과(§4.5)는 SimpleTouch [38]의 "예측 목표" 방식과 대비되는 설계입니다. SimpleTouch는 미래 촉각 전체를 예측합니다.

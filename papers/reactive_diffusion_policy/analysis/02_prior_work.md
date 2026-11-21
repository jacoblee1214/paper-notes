# 02. 선행연구와 관련 자료

이 문서는 RDP가 어떤 계보에 놓이는지, 구성 요소별로 어디서 왔는지, 공식 코드와 프로젝트 자료가 어디에 있는지 정리합니다. 논문 번호 [n]은 `01_paper_analysis.md`와 같은 원문 참고문헌 번호입니다.

## 1. 계보

| 계보 | 대표 연구 | RDP와의 관계 |
|---|---|---|
| 확산 정책 (행동 청크) | Diffusion Policy [10] (IJRR 2024), 3D Diffusion Policy [68] | RDP의 느린 정책(LDP)은 DP를 잠재 공간에서 다시 쓴 것입니다. 입력과 U-Net은 DP와 같은 틀입니다 |
| 행동 청크와 시간 앙상블 | ACT (ALOHA) [70], HATO [40] ("Learning visuotactile skills with two multifingered hands", arXiv:2404.16823) | 청크 길이와 시간 앙상블을 바꿔 열린 고리 문제를 완화하려는 두 방법. RDP는 표 V에서 둘 다 부작용이 크다고 보고 계층 구조로 대체합니다 |
| 잠재 행동과 압축 | Lai 외 "Action chunking as policy compression" [36], 잠재 확산 모델 LDM [49] | 청크를 잠재 공간에 두고 확산을 거는 방법. RDP는 AT로 잠재를 만듭니다 |
| 상대 엔드이펙터 행동, 지연 매칭 | UMI [11] | 상대 궤적과 지연 매칭을 가져와 촉각 정책에 맞게 씁니다(논문 §IV-B-3) |
| 쌍방향 제어 기반 행동 청크 | Bi-ACT [9], Kobayashi 외 [34] | 힘 피드백을 주는 양방향 원격조작과 청크 학습. 하드웨어가 같은 구조여야 해서 RDP는 이를 피한다고 설명합니다 |
| 촉각을 관측에 넣는 정책 | 3D-ViTac [27] (법선 힘만), MimicTouch [64], Bogert 외 [8] (GelSlim 전단력), Pattabiraman 외 VISK [45] (촉각 피부 + 시간 앙상블) | 대부분 관측 쪽에 촉각을 넣고 행동은 청크 방식입니다. RDP의 차이는 행동 쪽 고주파 보정입니다 |
| 힘 기반 반응 정책 | FoAR [24], ForceMimic [41], ACP [26] | 힘과 컴플라이언스를 쓰는 대안입니다. 논문은 이들이 특수 하드웨어나 자세 추정 오차를 가진다고 설명합니다 |
| 광학 촉각 표현 | GelSight [66], MCTac [48], 마커 변형장 추적 (점수 기반 추적, 논문 [21]), Helmholtz–Hodge 분해 [69] | 변형장 F와 PCA 특징은 이 계열의 표준 표현을 따릅니다 |
| 조건 주입 | FiLM [46] | LDP의 조건 입력 방식 |
| 변분 오토인코더 | Kingma [33] | AT의 KL 손실 |

## 2. 가까운 대안 (논문 §II 기준)

- **접촉을 명시적으로 쓰는 정책**: 촉각 데이터로 학습하는 접촉 역학 모델이나 임피던스 계열 정책. RDP는 이들과 직접 비교하지 않았습니다.
- **원격조작 하드웨어**: TactAR가 비교 대상입니다. 기존 원격조작은 VR 진동 피드백(coarse), 같은 구조의 리더-팔(haptic), 힘 센서 기반 핸드헬드입니다.
- **ImplicitRDP** (Chen-Wendi/ImplicitRDP, RA-L 2026): RDP 공식 README가 같은 팀의 후속 연구로 소개합니다. 저장소 설명에 따르면 같은 slow-fast 구조를 end-to-end로 단순화합니다. 이 저장소에서 분석하지 않았습니다.

## 4. 공식 코드와 프로젝트 자료

| 저장소 | 역할 | 라이선스 | 기준일 | 상태 |
|---|---|---|---|---|
| [xiaoxiaoxh/reactive_diffusion_policy](https://github.com/xiaoxiaoxh/reactive_diffusion_policy) | RDP 공식 구현 (학습, 추론, 데이터 수집, 실로봇 배포) | 최상위 `LICENSE`는 MIT (저자 Han Xue, Wendi Chen, Yuan Fang, 2025). 같은 파일에 Diffusion Policy, UMI 등 제3자 라이선스 고지가 함께 있습니다. GitHub API는 NOASSERTION으로 분류 | 별 397, 최근 push 2026-04-12, 커밋 `824c5e8` (2026-04-12) | 얕은 복제 `repo/`, **정적 읽기만**, 실행과 설치 안 함 |
| [xiaoxiaoxh/TactAR_APP](https://github.com/xiaoxiaoxh/TactAR_APP) | Meta Quest 3용 TactAR AR 앱 (Unity) | 라이선스 파일 없음 (API license null) | 별 93, 최근 push 2025-07-11 | 복제하지 않음. 촉각 시각화와 AR 지연은 논문 부록 B만 근거로 씁니다 |
| [Chen-Wendi/ImplicitRDP](https://github.com/Chen-Wendi/ImplicitRDP) | 후속 end-to-end 구현 (RA-L 2026) | NOASSERTION (other) | 별 64, 최근 push 2026-11-06 | 복제하지 않음 (후속 연구이므로 범위 밖) |
| [프로젝트 페이지](https://reactive-diffusion-policy.github.io) | 영상, 코드 링크, 보충 결과 | — | 확인일 2026-11-21 | 본문의 "website" 참조 대상 |
| Hugging Face `WendiChen/reactive_diffusion_policy_dataset`, `WendiChen/reactive_diffusion_policy_model` | 데이터셋, 체크포인트 (README에 링크) | 확인 안 함 | — | **받지 않음** (가중치와 데이터 다운로드 금지) |

- 공식 코드 README의 TODO에는 TactAR 코드, RDP 코드, 데이터, 체크포인트, 맞춤 가이드가 모두 완료(체크)로 표시되어 있습니다.
- 공식 README는 RDP 학습에 "GPU 워크스테이션(예: RTX 3090)"이 필요하고, Ubuntu 22.04와 ROS 2 Humble을 전제합니다.

## 5. 이 논문을 읽을 때 놓치기 쉬운 선행 자료

- Diffusion Policy의 "행동 청크"와 "DDIM 추론 단계" 설정(같은 저자 계열).
- UMI의 상대 행동과 지연 매칭 설명(논문 [11]).
- 3D-ViTac [27]은 법선 힘만 쓰는 방식으로 RDP 부록 G에서 절제 대조군으로 쓰입니다.

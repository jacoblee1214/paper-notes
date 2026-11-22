# 02. 선행연구와 관련 저장소

이 문서는 Being-H0의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드와 모델이 어디에 있는지 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.pdf`의 References, p.28–37)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| Being-H0의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 사람 영상으로 로봇 정책 사전학습 | 암묵 학습: R3M [21], MAE [22], GR00T 잠재 행동 [23]. 명시 학습: EgoDex [151] 등 | 저자들은 암묵 방식이 대규모 이득을 내지 못한다고 봅니다(§1, p.3). 손 모션을 **예측 대상으로 직접** 쓰는 것이 차이입니다 |
| 시각-언어 모델을 행동 모델로 | RT-1/2 [3, 4], OpenVLA [5], π0 [6], GR00T N1.5 [23], Octo [11], RDT-1B [121], FAST [120] | 기준선 GR00T N1.5와 InternVL3 [57]가 이 계열입니다 |
| 시각 명령 튜닝 (물리 명령 튜닝의 모체) | LLaVA [24, 27], InternVL [57, 140] | "물리 명령 튜닝"은 여기서 확장한 이름입니다 (§3.1) |
| 백본 | InternVL3 [140] (InternViT-300M, 2층 MLP), 동적 고해상도 [141] | 1B, 8B, 14B 세 크기 |
| 모션을 언어 토큰처럼 다룸 | 대규모 모션 모델 [66] (MotionLib, 같은 저자 그룹) | "손 모션을 외국어처럼" 취급 (§3.2). 모션 어휘 확장 방식도 [66]을 따릅니다 |
| 잔차 양자화 | RQ [28], 그룹 잔차 VQ(GRQ, 오디오 코덱) [29], VQ-VAE [76] | 손목과 손가락을 나누는 부분별 토크나이저는 [66]의 방식입니다 (§4.1.2) |
| MANO 손 모델 | MANO [139] | 손 파라미터 표준. 라이선스는 별도 등록 필요 (`04` §1) |
| 손 자세 추정 | HaMeR [102] | 3D 관절이 없는 데이터의 MANO 추정 (§5.2.1). 3D 손 재구성의 표준 방법 |
| 동적 손 복원과 카메라 추적 | Dyn-HaMR [106] | 관련 연구로 언급만 (§2) |
| 손-물체 상호작용 생성 | MEgoHand [118], 여러 모캡 데이터 [98, 99, 97, 148–150, 156, 157] | 기준선 GR00T 적응에 MEgoHand의 절차를 따름 (§6.2). UniHand 구성 데이터 |
| 자기중심 데이터 | EgoDex [151] (Apple Vision Pro, 829 h), Taste-Rob [154], HD-EPIC [152] (인용 번호 문제, `03` K6) | UniHand에서 가장 큰 출처 |
| 생성 지표 | TMR [158] (텍스트-모션 검색, FID와 M2T 계산에 씀) | 지표 모델의 체크포인트는 공개되지 않음 (`04` D17) |
| 로봇 원격 조작 | Gello [159], AnyTeleop [160], Being-0 [161] | 실로봇 시연 수집 (§6.1.3) |

## 2. 코드와 저장소 (GitHub 표)

| 저장소 / 자원 | 관계 | 라이선스 | 기준 (확인 날짜 2026-11-22) | 상태 |
|---|---|---|---|---|
| [BeingBeyond/Being-H0](https://github.com/BeingBeyond/Being-H0) | **공식 코드.** 추론, 토크나이저, 로봇 정렬 평가. 학습 코드 미공개 | MIT (LICENSE 파일, API 확인) | 별 61, 포크 0, 생성 2026-01-19, 마지막 푸시 2026-05-04, 커밋 `ae92e46` (README "ICML 2026" 채택 공지 2026-05-01) | **얕은 복제 `repo/`** 정적 읽기만. 설치, 실행 안 함 (`04`) |
| [beingbeyond.github.io/Being-H0](https://beingbeyond.github.io/Being-H0) | 프로젝트 페이지 (아카이브 링크: 논문, 코드, 모델) | 사이트 템플릿 CC BY-SA 4.0 (논문 라이선스와 별개) | 정적 확인 | 학회 표기 없음 |
| [research.beingbeyond.com/being-h0](https://research.beingbeyond.com/being-h0) | GitHub 저장소의 homepage 항목 | — | API 확인 | 읽지 않음 |
| HF `BeingBeyond/Being-H0-GRVQ-8K` | 모션 토크나이저 (wrist/finger 각 한 개) | MIT (LICENSE 파일) | 2025-08-02 | **설정 JSON만 받음** (`motion_args.json`, `meta_mask.json`). 가중치(0.21 GB) 미다운로드 |
| HF `BeingBeyond/Being-H0-1B-2508` | 사전학습 VLA, 2.37 GB | MIT (LICENSE 파일) | 2025-08-02 | 미다운로드 (`config.json`은 8B만 받음) |
| HF `BeingBeyond/Being-H0-8B-2508` | 사전학습 VLA, 16.05 GB | MIT | 2025-08-02 | `config.json`만 받음 |
| HF `BeingBeyond/Being-H0-14B-2508` | 사전학습 VLA, 30.86 GB | MIT | 2025-08-02 | 미다운로드 |
| HF `BeingBeyond/Being-H0-8B-Align-2508` | 로봇 정렬 후학습 VLA, 16.05 GB | MIT | 2025-08-02 | `config.json`만 받음 (행동 헤드 `action_token`) |
| HF datasets `BeingBeyond/h0_post_train_db_2508` | 후학습 데이터 (lerobot 형식 이미지 포함), 1.17 GB, 10,820 파일 | 태그 MIT | 2025-08-02 | 미다운로드 |
| [GitHub 설명 "ICML 2026"](https://github.com/BeingBeyond/Being-H0) | 채택 표기 | — | — | ICML 공식 목록과 일치 (`01` §0) |

- **관련 후속 연구.** README 인용 항목에 Being-H0.5 (arXiv:2601.12993)와 Being-H0.7 (arXiv:2605.00078)이 있습니다. 같은 저자 그룹이 쓴 것으로 README에 적혀 있습니다. 이 분석에서는 읽지 않았습니다.
- **학습 코드 상태.** README 체크리스트에서 "Training code and scripts"는 아직 미완료입니다(`04` D1).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **손 모션 토큰화가 정밀하다는 주장**은 GRQ-VAE [29]와 [66]의 부분별 토큰화에 기대고 있습니다. 재구성 정밀도 5.7 mm (표 5)는 그 계열에서 기대할 만한 수준입니다.
- **사람 영상이 로봇 전이에 쓰인다는 주장**은 EgoDex [151], HOI4D [98], TACO [156] 같은 데이터가 손 행동의 다양성을 준다는 전제에 기대고 있습니다. 로봇 손과 사람 손의 기구학 차이는 MLP 투영 한 개로 해결합니다(§4.3). 이 부분은 저자들이 "향후 과제"로 적었습니다.
- **시점 정렬 전략**(식 11–14)은 Dyn-HaMR [106]과 카메라 불일치 문제를 다룬 연구들의 계보입니다. 논문은 이를 데이터 전처리 쪽에서 해결합니다.
- **기준선 GR00T N1.5**는 명시적인 논문이 없습니다. 참고문헌 [23]은 GR00T N1입니다. 본문 표기와 인용 번호가 다릅니다(`03` K19).

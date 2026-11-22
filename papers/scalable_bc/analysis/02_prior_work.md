# 02. 선행연구와 관련 저장소 (ABC, arXiv:2606.27375)

## 1. 선행 계보

논문은 새 알고리즘이 아니라 **기존 구성 요소를 한 스택으로 묶고 공개해서 절제 실험을 가능하게 하는** 기여입니다. 그래서 계보도 구성 요소별로 읽습니다.

| 구성 요소 | 논문이 쓰는 것 | 출처 (논문 참고문헌 번호) | ABC에서 달라진 점 |
|---|---|---|---|
| 확산/흐름 정책 | DiT 행동 헤드, 조건부 확률 흐름(속도 예측) | DiT [2], Diffusion Policy [9], LBM [10] | LBM의 CLIP + adaLN을 기준선으로 두고 교차 어텐션·DINOv3로 확장(§3.1) |
| 비전 인코더 | DINOv3 ViT-B, CLIP ViT-B | DINOv3 [17], CLIP [16] | 카메라 3대가 인코더를 공유, 카메라별 12질의 풀링(B.2) |
| VLA 백본 | Gemma 3 4B + SigLIP | Gemma 3 [19], π0/π0.5 [12, 13], GR00T N1 [11], RT-2 [3] | 확산 손실로 VLM 전체 학습. 연결부 3종(교차 어텐션, FAST 병행, 풀링 adaLN) 비교(§3.2) |
| 연결부 설계 | 교차 어텐션, "knowledge insulating", FAST 토큰 | Driess et al. [20], FAST [21], π0.5의 KV 재사용 [13] | LBM-2의 레지스터 토큰 대신 QK-Norm 어텐션 풀링 [22] |
| 분산 감소 | 같은 VLM 특징에 독립 (잡음, 시간)을 k개 | (명시적 인용 없음. 확산 모델의 "multiple draws" 계열 관행) | 4B VLM에 비해 헤드가 작은 구조를 활용(§3.2) |
| 실시간 청킹 | 훈련 시 행동 prefix 조건화 | Training-time action conditioning for efficient RTC [15] | 사전학습 중 prefix 0–7 균등, 평가에서는 1–4 비교(H.2) |
| 조건화 | 조종자 ID, 하위 과제 프롬프트 | SARM의 단계 인식 보상 모델에서 영감받은 하위 과제 분류기 [42] | 데이터셋에 조종자 ID와 1,552 h의 하위 과제 주석을 함께 공개(§2, 부록 H) |
| DAgger | 개입 데이터 수집과 재학습 | DAgger [1] | 수동 GELLO 리더만으로 개입(SE(3) 델타 + mink IK) [41], 80:10:10 혼합(§5.3, 부록 F) |
| 원격조종 | GELLO 리더와 VR | GELLO [25], ALOHA 계열 [30–32] | YAM 양팔 + GELLO + (시뮬에서) Quest/Vision Pro |
| 데이터셋 | 대규모 공개 데이터 비교 | DROID [4], BridgeData-V2 [6], RH20T [29], Open X-Embodiment [5], AgiBot-World [7], MolmoAct2 BimanualYAM [8] | 3,553 h, 195 과제, 8,000달러 로봇 하나로 통일(§2, §6.1) |
| 시뮬 | MuJoCo + MJWarp, Blender 재렌더 | MuJoCo [24] | 실제 로봇을 재구성한 과제 10개(프로젝트 페이지는 20개)와 VR 데이터(§4) |

- **가장 가까운 대안:** LBM(Toyota Research Institute의 Large Behavior Model [10])은 같은 문제를 다루지만 코드와 데이터가 비공개이고, 논문도 "대규모 시스템의 코드는 대부분 닫혀 있다"(§6.2)고 씁니다. ABC의 위치는 "LBM류 recipe를 열린 데이터와 코드로 재현 가능하게 만든 것"입니다. π0.5, GR00T N1, MolmoAct2는 모델 가중치는 공개하지만 학습 데이터 전체나 학습 코드가 완전히 공개되지는 않습니다(논문 §6.2의 주장이며 이번에 개별 확인하지 않았습니다).
- **새로움의 정도:** 방법론적 새로움은 작습니다(풀링 adaLN VLA 연결부와 다중 draw가 가장 새로움). 공헌의 중심은 **스케일과 공개 범위**입니다. 3.5K 시간 규모에서 비교 가능한 공개 BC 데이터가 거의 없으므로 그 자체로 가치가 있지만, 표 1의 결론은 사내 7K 시간 데이터에 의존합니다(`01` §6, `03` V2–V3).

## 2. GitHub와 공개 자원 (2026-11-23 확인)

| 저장소 / 자원 | 관계 | 라이선스 | 별 | 마지막 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|---|
| [amazon-far/abc](https://github.com/amazon-far/abc) | **공식 코드**(학습, 평가, 시뮬, 배포) | Apache-2.0 | 401 (포크 46, 열린 이슈·PR 21, 기여자 6) | 2026-09-26 | `d0832d1` (2026-09-25) | 정적 읽기만(`repo/`), 실행·설치 안 함 |
| [abc.bot](https://abc.bot) | 프로젝트 페이지 (PDF는 arXiv v1과 동일) | - | - | - | - | HTML만 읽음 |
| [XDOF/ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) | 공식 데이터셋(MCAP) | apache-2.0 (카드 표기) | 좋아요 148 | 2026-09-24 | - | gated(자동 승인), 크기 1 TB 초과. 받지 않음 |
| 공식 체크포인트 (`prepare.py`로 받음) | `bottles_75k.pt`, `abc_dit_xl_200k_model.pt`(각 약 8.1 GB), VLA, DAgger 5종 | DiT와 DAgger는 Apache-2.0(DINOv3 약관 적용), VLA는 Gemma 약관 | - | - | - | 받지 않음 |
| [wuphilipp/gello_software](https://github.com/wuphilipp/gello_software) | GELLO 리더 소프트웨어(논문 [25]) | MIT | 566 | 2026-11-10 | - | 읽지 않음 |
| [i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt) | YAM 팔 드라이버 | MIT | 183 | 2026-11-20 | - | 읽지 않음 |
| [ArthurAllshire/i2rt-drivers](https://github.com/ArthurAllshire/i2rt-drivers) | 저자 포크(고정 커밋 `852f6ff`, "기본 이득이 xdof 데이터에 맞지 않다"는 주석) | MIT | 0 | 2026-04-14 | - | 읽지 않음 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 웹소켓 추론 클라이언트 코드 채택 | Apache-2.0 | 14,732 | 2026-08-24 | - | 읽지 않음 |
| [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | DINOv3 비전 백본 | Meta 라이선스 (API: NOASSERTION) | 11,787 | 2026-11-02 | - | 가중치는 약관 동의 후 별도 |
| [kevinzakka/mink](https://github.com/kevinzakka/mink) | DAgger 개입 IK | Apache-2.0 | 1,619 | 2026-11-19 | - | 읽지 않음 |

- 공식 코드의 최신 커밋은 논문 제출(2026-06-25)보다 3개월 뒤입니다. README는 "v3 릴리스(action-alignment 수정)"로 체크포인트를 다시 학습했다고 쓰며(`abc_sim/README.md`), 이 수정은 논문 v1에 없습니다. 논문 결과와 현재 공개 가중치가 같은 설정의 결과인지는 보장되지 않습니다(`04` D10).
- LBM, π0, π0.5, GR00T N1의 학습 코드는 이 조사에서 확인하지 않았습니다.

## 4. 한 줄 평가

방법이 아니라 인프라 논문이고, 가치의 대부분은 공개된 데이터와 코드에 있습니다. 논문 표 1의 모델 선택은 신뢰구간 없는 한 번의 평가(과제 3개)에 의존하고(`03` V2–V3), 공개 코드는 그 비교의 절반을 담지 않으며(`04` D9), 시뮬-실제 상관은 엔진 버전에 민감할 수 있습니다(`04` D14). 소비자 입장에서는 "ABC-130K와 8.1 GB 체크포인트로 출발하는 미세조정 기준선"으로 보는 것이 안전합니다.

# 02. 선행연구, 자료 지도, 관련 분석: Being-H0.5 (arXiv:2601.12993)

분석일 2026-11-22. 참고문헌 번호는 `paper/paper.txt`의 [1]–[139] 기준입니다. 다른 논문의 수치는 해당 논문의 분석 결과를 인용했고 원 논문을 다시 열지는 않았습니다.

## 1. 연구 계보 (논문이 스스로 쓰는 위치)

| 흐름 | 대표 문헌 (논문 번호) | 이 논문과의 관계 |
|---|---|---|
| 흐름 매칭 VLA | π0 [8], π0.5 [34], GR00T-N1 [9], X-VLA [11] | 행동 전문가와 정류 흐름의 출처. 실로봇 기준선은 π0.5, LIBERO·RoboCasa 비교표에 GR00T-N1, π0, X-VLA가 들어감 |
| MoT와 통합 모델 | BAGEL [123], MoT [124] | 이해-생성 전문가 분리 구조. 공식 코드의 학습 프레임워크도 BAGEL 계열(README 감사의 글) |
| 사람 영상 행동 사전학습 | EgoVLA [90], Being-H0 [10], EgoDex [64] | MANO 손 모션을 행동 공통어로 쓰는 직계. Being-H0의 모션 토크나이저를 이산 채널로 재사용(§5.2.3) |
| 잠재 행동 사전학습 | LAPA [70], UniVLA [71], GR-1/2 [73, 74] | 논문이 "간접 대리"로 분류. 직접 비교 실험은 없음 |
| 다중 임베디먼트 통합 | OXE [7], AgiBot World [48], RoboMIND [49, 50], RoboCOIN [51], X-VLA [11] | 데이터 출처이자 "임베디먼트별 헤드" 비판 대상(§5.1.1은 GR00T [9]를 지목) |
| 비동기 청킹 | Training-Time RTC [131] | UAC의 직접 출처. UAC는 지연을 임베디먼트별로 만든 확장 |
| 게이팅과 조건 신뢰도 | DiG-Flow [128] | MPG의 출처. "출력 게이팅" 대신 "입력 게이팅"이라고 차별화(§5.3.2) |
| 휴대용 수집 | UMI [55–57], UniCraftor(자체, §4) | 수집 시스템 비교 |

## 2. 공식 코드, 프로젝트, 자료 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 별 / 포크 | 최근 push | 비고 |
|---|---|---|---|---|---|
| 공식 코드 (Being-H, H0.5는 `Being-H05/`) | https://github.com/BeingBeyond/Being-H | Apache-2.0 | 1,159 / 60 | 2026-09-27 | **정적으로만 읽음**, 얕은 복제 커밋 `27ec67b`. 열린 이슈·PR 23. 학습·후학습·시뮬 평가 코드는 있고 사전학습 스크립트와 사람 모션 데이터 코드는 없음 (`04`) |
| 프로젝트 페이지 | https://research.beingbeyond.com/being-h05 | 해당 없음 | | 2026-01-20 게시 | "Accepted: CoRL 2026 (Oral)" 표기. 논문·코드·HF 링크 |
| 모델 Being-H05-2B (사전학습 "preview") | https://huggingface.co/BeingBeyond/Being-H05-2B | Apache-2.0 | 좋아요 15 | 2026-01-21 | safetensors 28.1억 파라미터, 파일 11.2 GB(fp32). **내려받지 않음**, config.json만 읽음 |
| 모델 LIBERO 전문가 / RoboCasa 전문가 / 범용 | `BeingBeyond/Being-H05-2B_libero`, `_robocasa`, `_libero_robocasa` | Apache-2.0 | 3 / 3 / 5 | 2026-01-21–22 | 같은 크기. 학습 후 설정은 `04` D12 |
| 데이터 UniHand_Preview | https://huggingface.co/datasets/BeingBeyond/UniHand_Preview | 카드에 라이선스 없음 | 다운로드 639 | 2026-05-13 | 사전학습 혼합의 부분집합(2026-03-20 공개). 저장소 크기 약 606 GB, EgoDex·ARCTIC·DexYCB 등 재배포 포함. 내려받지 않음 |
| 데이터 robocasa_human_single_lerobot | https://huggingface.co/datasets/BeingBeyond/robocasa_human_single_lerobot | CC-BY-4.0 | | | 약 4.9 GB. 내려받지 않음 |
| PND Adam-U 후학습 데이터 | HF 컬렉션 `pnd-adam-u-data` (9개 데이터셋) | 확인 안 함 | | 2026-01-24 공지 | 논문 실로봇 과제(꽃꽂이, 물건 전달, 화이트보드 닦기 등)와 같은 이름이 있음. 시연 수는 확인 안 함 |
| Being-H-EDU 튜토리얼 | 같은 저장소 `tutorials/Being-H-EDU/` | Apache-2.0 | | 2026-06-09 | SO101 교육용 로봇 후학습·배포 예제 |
| 관련: Being-H0 | https://github.com/BeingBeyond/Being-H0 | MIT | 61 | 2026-05-04 | 선행 모델(ICML 2026) |
| 관련: Being-H0.7 | 같은 저장소 `Being-H07/`(README만), arXiv:2605.00078 | | | 2026-04-14 공개 | 후속 WAM(NeurIPS 2026 채택, 저장소 뉴스). 코드·체크포인트는 "coming soon" |
| 관련: 파생 연구 | Rethink_VLA(Apache-2.0, 15★), TTP(Apache-2.0, 11★), BeTTER(MIT, 39★), JALA(14★), VIPA-VLA(27★) | | | | 저장소 README의 "Projects Based on Being-H" 목록. 이 분석에서는 열지 않음 |

논문 서론은 "모델 가중치, 학습 파이프라인, 시뮬레이션 스크립트에 더해 실세계 배포 인프라와 1,000 GPU-hour 사전학습 레시피를 공개한다"고 썼습니다(§1). 2026-11-22 기준 저장소 README의 TODO에서 "Complete pretraining scripts and documentation"과 "Out-of-the-box real robot pretrained checkpoints"는 미완료로 남아 있습니다(`04` D10).

## 4. 구성 요소별 출처와 이 폴더의 검증

| 구성 요소 | 출처 | 이 폴더의 검증 |
|---|---|---|
| 통합 행동 공간(200차원 슬롯, 축-각, 정규화 없음) | §5.1.1, 자체 (Being-H0 [10]의 물리 의미 정렬) | 코드 문서의 슬롯 표와 LIBERO·RoboCasa 설정이 일치. 정규화는 EEF에 없음, 그리퍼는 이진(`04` D6) |
| MoF 라우팅 | §5.1.2, MoE [12] | 공개 코드에 없음(`04` D1) |
| 이산 마스크 모션 예측과 하이브리드 목적 | §5.2.3, Being-H0 [10] | 공개 코드에 없음, `hmg_dataset.py`는 라이선스 헤더만(`04` D2) |
| ESA 슬롯별 어댑터 | §5.3.1 | 공개 코드에 없음(`04` D3) |
| MPG | §5.3.2, DiG-Flow [128] | 구현 있음. SWD는 평균 제곱 거리와 같고 LayerNorm이 크기를 지움(`03` §1.8, `04` D5) |
| UAC | §5.3.3, RTC [131] | 학습 쪽 구현은 있으나 임베디먼트별이 아니라 전역 지연, 모든 공개 레시피에서 꺼짐(`04` D4) |
| LIBERO·RoboCasa 수치 | 표 4, 5 | 평균 재계산, 기준선 집계 규칙 불일치(`03` §1.2, §1.3) |
| 사전학습 이득 | 표 6, 7 | 재계산, 기준선 변동 27%p(`03` §1.5, §1.6) |
| 실로봇 | 그림 10, 13 | 막대 라벨 판독, 시도 수 시나리오별 검정(`03` §1.7) |

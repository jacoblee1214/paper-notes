# 02. 선행연구와 관련 저장소

이 문서는 NativeMEM의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드가 있는지를 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| NativeMEM의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 기반 정책: SigLIP + PaliGemma 2B + 300M 행동 전문가, 흐름 정합 | π0 [3], π0.5 [4] (공개 구현 openpi) | 그대로 가져옵니다. 기억 토큰은 이 입력 시퀀스의 끝에 붙습니다 (식 8) |
| 단일 프레임 기반 정책의 한계 | RT-1 [47], RT-2 [1], OpenVLA [2], X-VLA [57] | 논문은 이들이 현재 관측과 짧은 이력만 쓴다고 봅니다 (Sec. 2). X-VLA는 표 1의 기반 VLA 기준선입니다 |
| VLM이 요약·검색하는 기억 | MemER [19] (경험 검색), MEM [20] (단기 영상 기억 + 장기 언어 기억), Mem-0 [21] (플래너 + 실행기 + 키프레임), 기타 [18, 22–24] | NativeMEM이 대비하는 첫 계열. 텍스트 변환과 주석 비용이 문제라고 봅니다 (Sec. 1). Mem-0와 MemER가 표 1, 2의 기준선입니다 |
| 정책 안의 외부 기억 모듈(순환 상태, 압축 이력, 검색 기억 은행) | ReMem-VLA [25], HAMLET [26], ContextVLA [9], HiF-VLA [27], MemoryVLA [52], MAP-VLA [53], GraSP-VLA [54], Chameleon [55] 외 [28–36] | 둘째 계열. "새로 초기화한 모듈은 사전학습 정책과 어긋날 위험이 있다"가 NativeMEM의 논거입니다 (Sec. 1). HAMLET이 표 1, 2의 기준선(재구현)입니다 |
| 이력 압축의 극단: 프레임당 토큰 1개 | HAMLET의 moment token [26], ReMem-VLA의 기억 질의 [25], GridS의 학습된 앵커 [56] | 같은 방향(토큰 압축)이지만 NativeMEM은 "VLA 자신의 비전 인코더를 복제해 질의 토큰 하나로 요약"하고, 압축기를 VLA 행동 손실로 학습합니다 |
| 비전 인코더에 시간 어텐션을 더해 최근 프레임 통합 | MEM [20] | 표 1, 2의 MEM-short* 기준선. 대규모 재학습이 전제인데 같은 시연 수로 다시 학습했다고 서술합니다 (Sec. 4.1) |
| 질의 토큰으로 패치를 요약하는 구조 | 학습 가능한 질의 토큰으로 패치 토큰을 모으는 일반 설계(논문은 별도 인용 없음) | 코드에서는 SigLIP 앞에 영으로 초기화한 질의 토큰을 붙이고 인코더 전체를 통과시킨 뒤 그 위치의 출력을 씁니다 (`04` D6) |
| 실시간 행동 청킹 | Training-time action conditioning for RTC [58] | 실로봇 표에서 π0.5 + RTC가 기준선입니다. 공개 코드에는 RTC 변형 모델이 있습니다 (`04` D12) |
| 평가 환경 | RMBench [21] (RoboTwin 2.0 기반 기억 과제 벤치마크), LIBERO | 표 1의 5과제 중 3개와 표 8이 RMBench에서 옵니다. Click Buttons 두 변형은 저자가 추가한 과제입니다 |
| 자기 지도 없이 행동 손실로 압축기 학습 | "복원 감독이 아닌 행동 감독" (논문 Sec. 3.2의 주장) | 대조 실험은 Stage 1에서 VLA를 푸는 경우와 평균 풀링 특징 사용 경우뿐입니다 (`03` §6) |

**가까운 대안과의 차이.**

- **MEM [20] (영상 인코더 + 언어 기억):** 단기는 시간 어텐션을 더한 영상 인코더(새 파라미터 없음)로 최근 수 초에서 최대 54초를, 장기는 고수준 정책이 갱신하는 언어 기억으로 처리합니다. NativeMEM은 언어 기억 없이 하나의 시각 토큰 열로 분 단위를 처리한다고 주장합니다. 두 방법의 직접 비교는 표 1, 2의 MEM-short*(재구현, 같은 시연 수)뿐입니다.
- **Mem-0 [21]:** VLM 플래너가 하위 과제를 만들고 실행기가 키프레임 기억을 씁니다. NativeMEM과 달리 두 모델 구조이고 추론 때 VLM이 필요합니다. 시뮬에서는 Swap Blocks 67%, Put Back Block 90%로 두 번째로 강한 기준선입니다.
- **HAMLET [26]:** 시점마다 moment token을 만들고 시간 대조 학습으로 다듬은 뒤 기억 모듈로 모읍니다. NativeMEM은 "새 모듈 없이" 같은 토큰 압축을 한다는 점이 다릅니다. 같은 π0.5 위에 재구현했고 표 1 평균 9.0%입니다.
- **전체 해상도 이력(Native History):** π0.5에 과거 4프레임을 256토큰 × 3시점으로 붙여 전체를 미세조정한 대조(표 6)입니다. 토큰 수 비교의 기준점입니다.

## 2. 코드와 저장소 (GitHub 표)

2026-11-26에 `gh api`로 확인했습니다(요약 지표는 그날의 값).

| 저장소 | 관계 | 라이선스 | 지표 / 대조 기준 | 상태 |
|---|---|---|---|---|
| [OpenDriveLab/NativeMEM](https://github.com/OpenDriveLab/NativeMEM) | 공식 코드(openpi 기반 JAX). 프로젝트 페이지의 Code 링크. 설명 "[CoRL 2026] Official Codebase for ..." | `LICENSE`는 Apache-2.0, `LICENSE_GEMMA.txt`(Gemma 이용 약관)도 포함 | ★19, 포크 3, 이슈 0, 마지막 push 2026-10-19, 커밋 10개. 대조 기준 `c599f81f80f5fc3b9c2c7c947f6fb2dbcd6f0315` (2026-10-19, "upload method.png") | 얕은 복제를 `repo/`에 두고 정적 읽기만 (`04`). 실행, 설치, import 안 함 |
| [Ziyeeee/NativeMEM-RMBench](https://github.com/Ziyeeee/NativeMEM-RMBench) | 공식 시뮬레이션 평가(공식 README가 링크). RMBench 코드를 고쳐 서버-클라이언트 평가와 `click_button`, `click_button_rand` 과제를 추가 | MIT | ★0, 마지막 push 2026-10-16. 대조 기준 `36cf2ca95bbea80b6f4edc1b74c9bc169e58e358` (2026-10-16) | 얕은 복제를 `data/rmbench/`에 두고 정적 읽기만. 실행, 설치 안 함 |
| [Hugging Face Ziyeeee/NativeMEM](https://huggingface.co/Ziyeeee/NativeMEM) | 체크포인트: Stage 1 토크나이저(`mem_tokenizer_pretrain/pretrain`) 6.8 GB, 시뮬 과제별 Stage 2 정책 5개(`click_button`, `click_button_rand`, `observe_and_pickup`, `put_back_block`, `swap_blocks`) 각 11.3 GB. 파일 199개 합계 약 63.6 GB | 모델 카드 apache-2.0 (API 메타데이터의 license 필드는 null) | 마지막 수정 2026-10-14. API 메타데이터의 크기만 확인 | 다운로드 안 함 (가중치는 승인 없이 받지 않음) |
| [Ziyeeee/openpi_rlt](https://github.com/Ziyeeee/openpi_rlt) | 관련 저장소. "OpenPI와 NativeMEM 위에 RL Token을 RTC와 함께 구현" (설명) | Apache-2.0 | ★0, 마지막 push 2026-10-21 | 복제 안 함. 목적이 다른 후속 구현 |
| [RoboTwin-Platform/RMBench](https://github.com/RoboTwin-Platform/RMBench) | 벤치마크 원본 (RMBench [21], Mem-0 기준선 포함) | MIT | ★235, 마지막 push 2026-09-20 | 복제 안 함 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 상위 코드 (π0, π0.5, 학습·서빙 틀) | Apache-2.0 | ★14,772, 마지막 push 2026-08-24 | 복제 안 함. 공식 저장소는 `openpi` 패키지 이름을 그대로 씀 |
| [Ziyeeee/nativemem.io](https://github.com/Ziyeeee/nativemem.io) | 프로젝트 페이지 소스로 보이는 저장소(내용 미확인) | 없음 | ★0, push 2026-07-09 | 확인 안 함 |

- 프로젝트 페이지(`opendrivelab.com/NativeMEM`)는 arXiv, Code 두 링크가 있고, 한 줄 요지는 "프레임당 토큰 1개, 160+ 프레임, 시연 100개로 미세조정, 실세계 평균 98.7%"입니다.
- HAMLET*, MEM-short*, MemER 기준선 코드는 이 저장소들에 없습니다(`04` D16). Mem-0의 구현은 RMBench 저장소 쪽에 있고 NativeMEM-RMBench에도 정책 폴더가 있습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **π0.5 행동 손실로 압축기를 학습한다는 설계는 π0.5의 공개 학습 틀을 그대로 쓴다는 전제에 기댑니다.** 공개 π0.5 코드가 흐름 정합만 쓰는 점은 NativeMEM 코드에서 확인됩니다(`04` D9, D10). Stage 1이 "원래 행동 손실"이라고 부르는 것은 이 흐름 정합 MSE입니다.
- **RMBench [21]의 Mem-0 수치:** NativeMEM 표 1의 Mem-0는 RMBench 저장소 README가 2026-07-14에 공지한 "재학습한 Mem-0 공개 가중치"(NativeMEM-RMBench 커밋 기록)와의 관계가 본문에 적혀 있지 않습니다. 같은 가중치인지는 확인하지 못했습니다.
- **RTC [58]:** 학습 때 행동 접두부를 조건으로 주는 방법입니다. 표 2의 π0.5 + RTC가 π0.5보다 크게 높은 점(14.7 → 34.7%)은 기억이 아닌 행동 연속성 효과라고 본문이 설명합니다.

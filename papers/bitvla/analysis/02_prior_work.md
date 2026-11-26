# 02. 선행연구, 구성 요소의 출처, 코드 자료

## 1. 계보

BitVLA는 세 줄기를 합칩니다. (a) 삼진 가중치 LLM, (b) LLaVA식 VLM 학습, (c) OpenVLA 계열의 VLA 학습과 OFT 미세조정.

**(a) 1비트·1.58비트 LLM**
- BitNet(Wang 등, 2023): 가중치를 ±1로 제한한 트랜스포머. 성능 격차가 파라미터 수가 커질수록 줄어든다는 경험적 관찰을 논문은 [38]로 인용합니다(II).
- BitNet b1.58(Ma 등, arXiv:2402.17764): 가중치를 {−1, 0, +1}로, 활성을 8비트로 하는 absmean·absmax 양자화. BitVLA의 식 (1)–(2)가 이 정의입니다. 저자 중 Hongyu Wang은 이 계열의 공저자입니다.
- BitNet b1.58 2B4T(Ma 등, arXiv:2504.12285): 2B 파라미터, 4T 토큰으로 처음부터 학습한 삼진 LLM. BitVLA의 LLM 백본입니다(공개 BF16 마스터 가중치 `hongyuw/bitnet-b1.58-2B-4T-bf16`를 config가 가리킴).
- OneBit(Xu 등): 지식 증류로 이진 LLM을 학습. BitVLA의 증류 아이디어의 가까운 전례로 인용됩니다(II).
- 추론 시스템: BitBLAS(커널, 논문 III-A가 사용), bitnet.cpp(CPU 추론, README가 권장), robot.cpp(BitVLA용 추론 프레임워크, 2026-11 공개).

**(b) VLM 학습**
- LLaVA(Liu 등, 2023)의 2단계(커넥터 정렬 → 지시 튜닝)를 따릅니다(III-B a). 데이터는 LLaVA-1.5-558k와 MammoTH-VL의 1,000만 샘플 단일 이미지 부분집합입니다.
- 비전 인코더는 SigLIP 계열입니다(Zhai 등, 2023). 논문은 "SigLIP-L 224"라 쓰고 OpenVLA의 선택(작은 해상도의 효율)을 따른다고 밝힙니다(III-A).

**(c) VLA 학습**
- OpenVLA(Kim 등, 2024): OXE 위의 7B VLA, 행동을 256구간 토큰으로 이산화. BitVLA의 로봇 사전학습은 이 방식을 따라 OXE 부분집합으로 합니다(III-B c).
- OpenVLA-OFT(Kim 등, arXiv:2502.19645): 병렬 디코딩, 행동 청크, L1 회귀 헤드. BitVLA의 미세조정, LIBERO 데이터, 평가 프로토콜, 코드 바탕(`openvla-oft` 포크)입니다.
- π0(Black 등, 2024): 흐름 매칭 VLA. 3B급 비교 기준입니다.

## 2. 가까운 대안과 비교 대상

| 접근 | 대표 | BitVLA와의 관계 |
|---|---|---|
| 사후 양자화(PTQ) | bitsandbytes INT8·INT4 (LLM.int8(), Dettmers 등) | 표 II의 직접 비교 대상. BitVLA는 학습 중 양자화(QAT)로 같은 메모리 대에서 비슷한 LIBERO 성능을 보고 |
| 작은 VLM 백본 | TinyVLA, NORA, SmolVLA, GR00T-N1 | 모델 크기를 줄이는 다른 길. 논문은 "서로 결합할 수 있다"고 씁니다(II). 표 I에서 같은 크기대의 비교 행 |
| 어댑터·디코딩 개선 | VLA-Adapter, EVO-1 | 성능 쪽 개선. 효율은 "첫 원리에서"(양자화와 학습의 공동 설계)라고 구별합니다(II) |
| 큰 VLA | UniVLA, UnifiedVLA, CoT-VLA, OpenVLA, OpenVLA-OFT | 표 I의 Large 행 |
| 로봇용 전용 하드웨어 | (논문 서론: 1비트 VLA 가속기 가능성) | 후속 과제로만 언급 |

## 4. GitHub와 코드 자료 (확인 2026-11-26)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (BitVLA)** | https://github.com/ustcwhy/BitVLA | MIT | 별 175, 포크 13, 열린 이슈 4, 마지막 push 2026-11-18 | 커밋 `a8d62dfe398bfcce01283278cd716d4d6eae3c5a`(2026-11-18, "Update README.md"). 커밋 30개, 얕은 clone을 정적으로만 읽음. 로봇 사전학습 코드는 "계획" 표시(README) |
| 프로젝트 페이지 | - | - | - | 별도 페이지를 찾지 못함(arXiv 초록 페이지와 GitHub 저장소가 전부, GitHub homepage 필드 비어 있음) |
| robot.cpp (BitVLA 추론) | https://github.com/Robot-cpp/robot.cpp | Apache-2.0 | 별 43, push 2026-11-26 | README가 "공식 추론 프레임워크"라 소개. 이 분석에서는 읽지 않음 |
| BitNet (bitnet.cpp) | https://github.com/microsoft/BitNet | MIT | 별 42,481, push 2026-07-27 | 오프라인 1.58비트 추론을 README가 권장 |
| BitBLAS | https://github.com/microsoft/BitBLAS | MIT | 별 771, push 2025-08-06 | 논문 III-A의 커널 |
| OpenVLA-OFT | https://github.com/moojink/openvla-oft | MIT | 별 1,451, push 2025-09-09 | BitVLA 코드가 이 저장소의 포크 안에 있음 |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | MIT | 별 2,572, push 2025-03-15 | 평가 환경 |
| lmms-eval | https://github.com/EvolvingLMMs-Lab/lmms-eval | 라이선스 필드 미판정(NOASSERTION) | 별 4,512, push 2026-11-25 | VQA 평가 도구(표 III) |
| HF 사전학습 VLA | https://huggingface.co/lxsy/bitvla-bf16 | MIT (태그) | 파라미터 3,147,925,664(BF16), 파일 6.30 GB, 갱신 2026-03-07 | 마스터 가중치. 받지 않음 |
| HF LIBERO 미세조정 4종 | `hongyuw/ft-bitvla-bitsiglipL-224px-libero_{spatial,object,goal,long}-bf16` | MIT (태그) | 파라미터 2,819,549,344, 생성 2025-06, 갱신 2025-07-03 | 받지 않음 (`04` D13) |
| HF VL 사전학습 | `hongyuw/bitvla-bitsiglipL-224px-bf16`, `hongyuw/bitvla-siglipL-224px-bf16` | MIT (태그) | 갱신 2025-06-30 | VQA 평가용. 받지 않음 |
| HF I2S GGUF 4종 | `robotcpp/bitvla-libero-{spatial,object,goal,long}-i2s` | MIT (태그) | 한 모델 합계 1,440,224,736 byte | robot.cpp용 2비트 묶음. 크기는 HF API 메타데이터에서 읽음. 받지 않음 |
| OpenReview | https://openreview.net/forum?id=mHCqiRUiHi | - | - | CoRL 2026 목록의 링크. 열어 보지 않음 |

- 가중치와 데이터셋(OXE, MammoTH-VL, LLaVA-558k, modified_libero_rlds)은 받지 않았습니다. 크기는 허브 API 메타데이터로만 확인했습니다.
- 라이선스 열은 GitHub API의 license 값 또는 허브 태그입니다. 별과 push 시각은 `gh api` 응답 기준입니다.

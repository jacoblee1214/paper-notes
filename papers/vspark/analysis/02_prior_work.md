# 02. 선행연구와 자료

## 1. 계보

| 축 | 대표 연구 (논문 참고문헌 번호) | 이 논문에서의 역할 |
|---|---|---|
| 행동 토큰화 (bin) | OpenVLA [20] | 행동 차원마다 bin 하나, 토큰 7개/스텝. 기존 완화 검증이 이 구조에 맞춰짐 |
| 행동 토큰화 (압축) | FAST [40] (π0-FAST의 토크나이저), ActionCodec [13], G0.5 [33] | 청크를 짧은 토큰열로 압축. 토큰 하나가 행동 한 값에 대응하지 않음 |
| VLA 기반 | π0-FAST, StarVLA [12], Qwen2.5-VL | 타깃 모델. StarVLA는 WidowX 체크포인트도 씀(표 A.3) |
| 투기 디코딩 기본 | SD [5, 24], 트리 검증 [4, 37] | 무손실 가속의 틀 |
| 자기회귀 드래프터 | EAGLE [26], EAGLE-2 [27], EAGLE-3 [28] | 한 토큰씩 드래프트. EAGLE-3는 특징 회귀를 뺌 |
| 병렬 드래프터 | P-EAGLE [17], PARD [1, 2], DFlash [7], DFlash2 [18], Domino [16], DSpark [10] | 한 번에 여러 위치를 예측. DSpark는 Markov 보정을 더함 |
| VLA 투기 디코딩 (완화) | Spec-VLA [44], KERV [48], HeiSD [49], WA-SpecDec [45] | 가까운 행동 bin이면 수용. 압축 토크나이저에는 적용이 어려움 |
| 확산 VLA 투기 | [34, 39] | 토큰 검증 바깥 |
| 버린 예측 재사용 (LLM) | [21, 36] | 검증 후 버려지는 타깃 은닉 상태·예측을 다음 드래프트에 씀 |
| 벤치마크 | LIBERO [32], LIBERO-Plus [15], LIBERO-Pro [51], SimplerEnv [25] | 평가 |

VSpark가 가장 가까운 선행 연구는 DSpark [10]입니다(기반 드래프터). 차이는 세 가지입니다. 잡음 롤아웃, 특징 회귀(EAGLE-3에서 뺀 것을 되살림), 연속 재사용(트리를 타깃 예측으로 채움).

## 2. 논문이 명시한 구성 요소의 출처

- DSpark: arXiv:2607.05147 (공식 저장소 README의 감사의 말). 논문 본문은 [10]으로만 인용.
- 잡음 주입: DART [22]. VSpark는 노이즈 보정 기준을 "정책과 전문가의 차이"에서 "드래프트와 타깃의 차이"로 바꿈(§3.2).
- 드래프트 샘플링 앵커: DeepSpec (MIT, `deepseek-ai/DeepSpec`)에서 가져옴(공식 저장소 `vspark/loss.py` 주석).
- 타깃 모델의 출처: LeRobot (π0-FAST), ActionCodec 공식 저장소, StarVLA.

## 4. GitHub와 자료 표 (확인 2026-11-22, GitHub API)

| 자료 | 링크 | 라이선스 | 별 / 포크 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| **공식 VSpark 코드** | https://github.com/suho-yoo/VSpark | Apache-2.0 | 7 / 0 | 2026-11-21 | 커밋 1개 (f151152 "initial commit", 2026-11-21). 생성 2026-11-17. `repo/`에 얕은 clone, 정적 읽기만. 크기 약 40 MB (영상 28 MB 포함) |
| 프로젝트 페이지 | https://suho-yoo.github.io/VSpark/ | - | - | - | 데모와 결과. BibTeX는 `@article`, 학회 없음 |
| VSpark 드래프트 체크포인트 | https://huggingface.co/suhoyoo/VSpark | 타깃 원 라이선스 따름 (README) | - | - | `vspark_pi0_fast.pt`, `vspark_actioncodec.pt`, `vspark_starvla.pt`. 받지 않음. **G0.5 체크포인트 없음** |
| π0-FAST 타깃 | https://huggingface.co/lerobot/pi0fast-libero | 미확인 | - | - | LeRobot 저장소는 Apache-2.0 (29,018 별, 2026-11-22 push) |
| StarVLA 타깃 | https://huggingface.co/StarVLA/Qwen2.5-VL-FAST-LIBERO-4in1 | 미확인 | - | - | 코드는 `starVLA/starVLA` (라이선스 NOASSERTION, 3,970 별, 2026-11-18 push) |
| ActionCodec 타깃·토크나이저 | https://github.com/ZibinDong/actioncodec | 라이선스 없음 | 81 | 2026-03-01 | 공식 VSpark는 커밋 f034eb9 고정 |
| G0.5 타깃 | (표 A.3: ZibinDong/g05-libero-ar) | 미확인 | - | - | 공식 VSpark 코드에 G0.5 항목 없음 (`04` D1) |
| 기준선: EAGLE-3 | https://github.com/SafeAILab/EAGLE | NOASSERTION | 2,583 | 2026-02-20 | 논문은 `SafeAILab/EAGLE` 사용 (표 A.3) |
| 기준선: P-EAGLE, DFlash2 | https://github.com/vllm-project/speculators | Apache-2.0 | 973 | 2026-11-20 | |
| 기준선: DFlash | https://github.com/z-lab/dflash | MIT | 6,254 | 2026-08-18 | |
| 기준선: DSpark 앵커 | https://github.com/deepseek-ai/DeepSpec | MIT | 7,325 | 2026-07-09 | |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | MIT | 2,563 | 2025-03-15 | 저장소 requirements.txt는 `libero2gym==0.1.0`과 mujoco 3.3.2 |
| LIBERO-Plus | https://github.com/sylvestf/LIBERO-plus | 라이선스 없음 | 502 | 2026-01-21 | |
| LIBERO-Pro | https://github.com/Zxy-MLlab/LIBERO-PRO | MIT | 362 | 2026-07-17 | |
| LeRobot | https://github.com/huggingface/lerobot | Apache-2.0 | 29,018 | 2026-11-22 | 공식 저장소 requirements에 커밋 e595b79 고정 |
| openpi (비교용) | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 14,727 | 2026-08-24 | VSpark 코드는 쓰지 않음. |

- 저장소 `requirements.txt`의 핀: torch 2.7.1, transformers 5.5.4, mujoco 3.3.2, lerobot은 커밋 고정. README는 ActionCodec을 **transformers 4.57.1 별도 환경**에서 돌리라고 합니다. 두 설정이 같은 저장소에 있습니다(`04` D12).
- 공식 저장소의 라이선스 파일(LICENSE, NOTICE)은 Apache-2.0이며 타깃 체크포인트는 원 라이선스를 따른다고 적혀 있습니다.
